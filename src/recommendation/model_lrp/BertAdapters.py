from .BertLRP import BertLayer, BertAttention, BertSelfAttention
from .LayersLRP import *
import torch
from typing import Optional

class BertLayerAdapters(BertLayer):
    def __init__(self, config):
        super().__init__(config)
        self.attention = BertAttentionAdapters(config)
        self.config = config

    def forward(
              self,
              hidden_states,
              attention_mask=None,
              head_mask=None,
              output_attentions=False,
              user_embeds: Optional[torch.FloatTensor] = None,
      ):
        multiply_keys = user_embeds[:,:,:self.config.hidden_size] #TODO maybe clone
        multiply_values = user_embeds[:,:,self.config.hidden_size:self.config.hidden_size*2]  #TODO maybe clone

        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
            multiply_keys=multiply_keys,
            multiply_values=multiply_values,
        )
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        ao1, ao2 = self.clone(attention_output, 2)
        intermediate_output = self.intermediate(ao1)
        layer_output = self.output(intermediate_output, ao2)

        outputs = (layer_output,) + outputs
        return outputs

class BertAttentionAdapters(BertAttention):
    def __init__(self, config):
        super().__init__(config)
        self.self = BertSelfAttentionAdapters(config)

    def forward(
            self,
            hidden_states,
            attention_mask=None,
            head_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            output_attentions=False,
            multiply_keys: Optional[torch.FloatTensor] = None,
            multiply_values: Optional[torch.FloatTensor] = None,
    ):
        h1, h2 = self.clone(hidden_states, 2)
        self_outputs = self.self(
            h1,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            output_attentions,
            multiply_keys,
            multiply_values,
        )
        attention_output = self.output(self_outputs[0], h2)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs

class BertSelfAttentionAdapters(BertSelfAttention):
    def __init__(self, config):
        super().__init__(config)
        self.key_mul = Mul()
        self.value_mul = Mul()

    def forward(
            self,
            hidden_states,
            attention_mask=None,
            head_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            output_attentions=False,
            multiply_keys: Optional[torch.FloatTensor] = None,
            multiply_values: Optional[torch.FloatTensor] = None,
    ):
        self.head_mask = head_mask
        self.attention_mask = attention_mask

        h1, h2, h3 = self.clone(hidden_states, 3)
        mixed_query_layer = self.query(h1)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        if encoder_hidden_states is not None:
            mixed_key_layer = self.key_mul((self.key(encoder_hidden_states), multiply_keys))
            mixed_value_layer = self.value_mul((self.value(encoder_hidden_states), multiply_values))
            attention_mask = encoder_attention_mask
        else:
            mixed_key_layer = self.key_mul((self.key(h2), multiply_keys))
            mixed_value_layer = self.value_mul((self.value(h3), multiply_values))

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = self.matmul1([query_layer, key_layer.transpose(-1, -2)])
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = self.add([attention_scores, attention_mask])

        # Normalize the attention scores to probabilities.
        attention_probs = self.softmax(attention_scores)

        self.save_attn(attention_probs)
        attention_probs.register_hook(self.save_attn_gradients)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = self.matmul2([attention_probs, value_layer])

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        return outputs

    def relprop(self, cam, **kwargs):
        # Assume output_attentions == False
        cam = self.transpose_for_scores(cam)

        # [attention_probs, value_layer]
        (cam1, cam2) = self.matmul2.relprop(cam, **kwargs)
        cam1 /= 2
        cam2 /= 2
        if self.head_mask is not None:
            # [attention_probs, head_mask]
            (cam1, _)= self.mul.relprop(cam1, **kwargs)

        self.save_attn_cam(cam1)

        cam1 = self.dropout.relprop(cam1, **kwargs)

        cam1 = self.softmax.relprop(cam1, **kwargs)

        if self.attention_mask is not None:
            # [attention_scores, attention_mask]
            (cam1, _) = self.add.relprop(cam1, **kwargs)

        # [query_layer, key_layer.transpose(-1, -2)]
        (cam1_1, cam1_2) = self.matmul1.relprop(cam1, **kwargs)
        cam1_1 /= 2
        cam1_2 /= 2

        # query
        cam1_1 = self.transpose_for_scores_relprop(cam1_1)
        cam1_1 = self.query.relprop(cam1_1, **kwargs)

        # key
        cam1_2 = self.transpose_for_scores_relprop(cam1_2.transpose(-1, -2))
        cam1_2,_ = self.key_mul.relprop(cam1_2, **kwargs)
        cam1_2 = self.key.relprop(cam1_2, **kwargs)

        # value
        cam2 = self.transpose_for_scores_relprop(cam2)
        cam2,_ = self.value_mul.relprop(cam2, **kwargs)
        cam2 = self.value.relprop(cam2, **kwargs)

        cam = self.clone.relprop((cam1_1, cam1_2, cam2), **kwargs)

        return cam
