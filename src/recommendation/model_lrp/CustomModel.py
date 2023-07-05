from transformers import BertPreTrainedModel
from transformers.utils import logging
from torch.nn import CrossEntropyLoss, MSELoss, BCEWithLogitsLoss
from typing import List, Any, Optional, Union, Tuple
import torch
from transformers.modeling_outputs import SequenceClassifierOutput

from transformers.configuration_utils import PretrainedConfig

from .Layers import *
from .Bert import BertModel
from .BertAdapters import BertLayerAdapters

class BertForSequenceClassificationAdapters(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.bert = BertModel(config)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = Dropout(classifier_dropout)
        self.classifier = Linear(1, config.num_labels)

        self.adapter_size = config.hidden_size * 2# + config.intermediate_size
        self.embedding_size = config.embedding_size

        self.user_embeddings = nn.Embedding(config.num_users, self.embedding_size)
        #add additional projection layer if adapter_size != self.embedding_size
        if self.adapter_size != self.embedding_size:
          self.user_projection = Linear(self.embedding_size, self.adapter_size)
        self.user_dropout = Dropout(config.user_dropout_prob)
        self.user_norm = LayerNorm(self.embedding_size, eps=config.layer_norm_eps)
        
        self.bert.encoder.layer[-1] = BertLayerAdapters(config)

        comparator_data = torch.zeros(config.num_classification_heads,config.hidden_size, device=self.bert.device).normal_(mean=0.0, std=config.initializer_range)
        self.sample_comparison = torch.nn.parameter.Parameter(data=comparator_data, requires_grad=True)
        self.sample_norm = LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.dot = MatMul()

        self.register_buffer("adapter_bias", torch.ones(1, self.adapter_size))

        # Initialize weights and apply final processing
        self.post_init()

    def calculate_contrastive_loss(self, current_user, positive_indices, negative_indices, margin: float = 0.5):
        """
        Computes Contrastive Loss
        """

        positive_samples = torch.cartesian_prod(current_user, positive_indices.squeeze(dim=0))
        negative_samples = torch.cartesian_prod(current_user, negative_indices.squeeze(dim=0))


        #similar labesl are zeros / disimilar labels are ones
        positive_samples_labels = torch.zeros(negative_samples.shape[0], device=negative_samples.device)
        negative_samples_labels = torch.ones(positive_samples.shape[0], device=positive_samples.device)

        all_contrastive_samples = torch.concat([positive_samples, negative_samples],dim=0)
        label  = torch.concat([positive_samples_labels, negative_samples_labels],dim=0)

        x1 = self.user_embeddings(all_contrastive_samples[:,0])
        x2 = self.user_embeddings(all_contrastive_samples[:,1])

        x1 = torch.nn.functional.normalize(x1, p=2, dim=-1)
        x2 = torch.nn.functional.normalize(x2, p=2, dim=-1)

        dist = torch.nn.functional.pairwise_distance(x1, x2)

        loss = (1 - label) * torch.pow(dist, 2) + (label) * torch.pow(torch.clamp(margin - dist, min=0.0), 2)
        loss = torch.mean(loss)

        return loss

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        users: Optional[torch.Tensor] = None,
        positive_samples: Optional[torch.Tensor] = None,
        negative_samples: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if len(input_ids.shape) > 2:
          input_ids = input_ids.squeeze(dim=0)
          attention_mask = attention_mask.squeeze(dim=0)

        adapter_values = self.adapter_bias

        if users is not None:
          users = users.squeeze(dim=0)
          user_embeds = self.user_embeddings(users)
          user_embeds = self.user_norm(user_embeds)
          user_embeds = self.user_dropout(user_embeds)
          if self.adapter_size != self.embedding_size:
            adapter_values = self.adapter_bias + self.user_projection(user_embeds)
          else:
            adapter_values = self.adapter_bias + user_embeds

        #extended_mask = self.bert.get_extended_attention_mask(attention_mask, input_ids.size())

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            user_embeds=adapter_values.unsqueeze(dim=1),
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)

        samples = self.sample_norm(self.sample_comparison)
        samples = self.dropout(samples)

        logits = self.dot((pooled_output, samples.T))

        click_prediction = self.classifier(logits)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                if input_ids.shape[0] > 1:
                  loss = loss_fct(logits.view(-1, 5), labels)

                  click_prediction_labels = torch.zeros(5, dtype=torch.long, device=click_prediction.device)
                  click_prediction_labels[labels] = torch.tensor(1, device=click_prediction.device)

                  loss_fct = CrossEntropyLoss(weight=torch.tensor([0.25,1.], device=click_prediction.device))
                  loss += loss_fct(click_prediction.view(-1, 2), click_prediction_labels)
                else:

                  loss = loss_fct(click_prediction.view(-1, 2), labels)

            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)


            if positive_samples is not None and negative_samples is not None:
              contrastive_loss = self.calculate_contrastive_loss(users, positive_samples, negative_samples)
              loss += contrastive_loss

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=click_prediction,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def relprop(self, cam=None, **kwargs):
        cam = self.classifier.relprop(cam, **kwargs)
        cam = torch.ones_like(cam)
        cam, _ = self.dot.relprop(cam, **kwargs) #added
        #print("Comparators: ", _.sum())
        #cam1 = self.dropout.relprop(cam1, **kwargs)
        #cam1 = self.sample_norm.relprop(cam1, **kwargs) #added
        #cam = self.bert.pooler.relprop(cam, **kwargs) # added
        #cam = self.adapter_layer.relprop(cam, **kwargs) # added

        #todo what todo with cam1

        cam = self.dropout.relprop(cam, **kwargs)
        cam = self.bert.relprop(cam, **kwargs)
        # print("conservation: ", cam.sum())
        return cam

class BertConfigAdapters(PretrainedConfig):
    model_type = "bert"

    def __init__(
        self,
        vocab_size=30522,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=2,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        pad_token_id=0,
        position_embedding_type="absolute",
        use_cache=True,
        classifier_dropout=None,
        num_classification_heads=2,
        embedding_size=32,
        num_users = None,
        **kwargs,
    ):
        super().__init__(pad_token_id=pad_token_id, **kwargs)

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.position_embedding_type = position_embedding_type
        self.use_cache = use_cache
        self.classifier_dropout = classifier_dropout
        self.num_classification_heads = num_classification_heads
        self.embedding_size = embedding_size
        self.num_users = num_users
