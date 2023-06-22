from .CustomModel import BertForSequenceClassificationAdapters, BertConfigAdapters
from transformers import AutoTokenizer
from typing import List
import torch
from rouge_score import rouge_scorer #pip install rouge-score
from huggingface_hub import hf_hub_download
import json

class ClickPredictor():
  def __init__(self, huggingface_url : str, commit_hash : str = None, device : str = None):
    """
    load model from remote if not already present on disk. once the data is downloaded it is cached on your disk
    :param
      huggingface_url (str) : url pointing to the huggingface repository for example "josh-oo/news-classifier"
      commit_has (str) : the corresponding hash if you want to use a certain version for example "1b0922bb88f293e7d16920e7ef583d05933935a9"
      device (str) : if you want to execute it on a certain device for example "cpu" or "cuda"
    """
    config = BertConfigAdapters.from_pretrained(huggingface_url, revision=commit_hash)
    self.model = BertForSequenceClassificationAdapters.from_pretrained(huggingface_url, revision=commit_hash)
    self.tokenizer = AutoTokenizer.from_pretrained(huggingface_url, revision=commit_hash)
    
    #load user mapping
    self.user_mapping = {}
    user_mapping_file = hf_hub_download(repo_id=huggingface_url, filename="user_mapping.json", revision=commit_hash)
    with open(user_mapping_file) as f:
        self.user_mapping = json.load(f)
    
    #TODO add "CUSTOM" user embedding
        
    self.device = device
    if self.device is not None:
      self.model.to(device)
      
  def _convert_tokens_to_words(self, tokens : List[str], values : torch.FloatTensor):
    """
    converts subword tokens to words and removes special tokens
    :param
      tokens : (List[str]) : the list of tokens
      values : torch.FloatTensor : the aggregated attentions for the corresponding tokens
    :return
      words : (List[str]) : a list containing all merged words
      values :  torch.FloatTensor: the modified attention map
    """
    all_words = []
    all_values = []
    current_word = ""
    current_value = 0
    for i,token in enumerate(tokens):
    if token in self.tokenizer.special_tokens_map.values():
      continue
    if not token.startswith("##") and len(current_word) > 0:
      all_words.append(current_word)
      all_values.append(current_value)
      current_word = ""
      current_value = 0
    if token.startswith("##"):
      token = token.replace("##","",1)
    current_word += token
    current_value += values[i].item()
    all_words.append(current_word)
    all_values.append(current_value)

    return all_words, torch.nn.functional.softmax(torch.tensor(all_values), dim=-1)

def _extract_word_deviations(self, tokens : List[str], non_personal_attentions : torch.FloatTensor, personal_attentions : torch.FloatTensor):
  """
  aggregates the attention map (multi-head -> single head) and transforms subword tokens into full words
  :param
    tokens : (List[str]) : the list of tokens
    non_personal_attentions : torch.FloatTensor : the full (last) attention map for the non-personal prediction
    personal_attentions : torch.FloatTensor : the full (last) attention map for the personal prediction
  :return
    word_deviations : dict : a dict containing all words of the input headline and the deviation between personal and non-personal predictions
  """
  temperature = 1/12 #1/number_of_heads

  #we are only interested in the cls token as it is used in our pooling step
  cls_attention_personal = personal_attentions[:,0]
  cls_attention_non_personal = non_personal_attentions[:,0]

  #aggregate with log summation (Adapted from: Attention Distillation: self-supervised vision transformer students need more guidance)
  personal_aggregated = torch.nn.functional.softmax(temperature * cls_attention_personal.log().sum(dim=-2),dim=-1)
  non_personal_aggregated = torch.nn.functional.softmax(temperature * cls_attention_non_personal.log().sum(dim=-2),dim=-1)

  words_personal, values_personal = self._convert_tokens_to_words(tokens, personal_aggregated)
  words_non_personal, values_non_personal = self._convert_tokens_to_words(tokens, non_personal_aggregated)

  deviation = (values_personal - values_non_personal) / values_non_personal

  all_deviations = {}
  for deviation, word in zip(deviation, words_personal):
    all_deviations[word] = deviation.item()

  return all_deviations

  def calculate_scores(self, headlines : List[str], user_id : str = "CUSTOM"):
    """
    calculate scores for a list of headlines for a given user
    :param
      headlines : (List[str]) : the list of headlines
      user_id (str) : either the user_id corresponding to the MIND dataset (for example) to calculate scores for historic users
                      or "CUSTOM" to calculate scores for the current new user
      compare : bool : if True -> compare results against non-personalized predictions (needed for wordclouds); if False -> calculate personal scores without comparison
    :return
      scores : (List[float]) : a score for each headline specifying a probability for a click event (1.0 = 100%)
      word_level_deviations (List[dict]): a list of dicts containing the headlines words and the deviation compared to the unpersonalize net
      personal_deviations (List[float]) : a list of floats indicating the deviation of the personal click probability compared to the non-personalized net
    """
    assert user_id == "CUSTOM" or user_id in self.user_mapping.keys(), "Given user id is not available"

    #the personal user embedding is saved at the last embedding matrix index
    user_index = torch.tensor([len(self.model.user_embeddings.weight) -1])

    if user_id != "CUSTOM":
      user_index = torch.tensor([self.user_mapping[user_id]])
    inputs = self.tokenizer(headlines, return_tensors="pt", padding='longest')
    if self.device is not None:
      inputs = inputs.to(device)
      user_index = user_index.to(device)
    model.eval()
    with torch.no_grad():
      personalized_outputs = model(**inputs, users=user_index.unsqueeze(dim=0), output_attentions=compare)
      if compare:
        non_personalized_outputs = model(**inputs, users=None, output_attentions=True)

    personal_probs = torch.nn.functional.softmax(personalized_outputs.logits, dim=-1)
    personal_scores = personal_probs[:,1].detach().numpy()

    personal_deviations = None
    word_deviations = None

    if compare:
      non_personal_probs = torch.nn.functional.softmax(non_personalized_outputs.logits, dim=-1)
      non_personal_scores = non_personal_probs[:,1].detach().numpy()

      personal_deviations = np.abs(personal_scores-non_personal_scores)

      personal_attention = personalized_outputs.attentions[-1].squeeze(dim=0)
      non_personal_attentions = non_personalized_outputs.attentions[-1].squeeze(dim=0)

      word_deviations = []
      for i, headline in enumerate(inputs['input_ids']):
        current_tokens = tokenizer.convert_ids_to_tokens(headline)

        word_deviations.append(_extract_word_deviations(None, current_tokens,non_personal_attentions[i], personal_attention[i]))

    return personal_scores, word_deviations, personal_deviations

  def update_step(self, new_headline : str, new_label : int):
    """
    update the users embedding vector in this online learning step
    :param
      new_headline : (str) : the new headline
      new_label (int) : the label (in  {0,1}) 0 if the user did not like the new_headline, 1 if the user did like the new_headline
    """

    #the personal user embedding is saved at the last embedding matrix index
    user_index = torch.tensor([len(self.model.user_embeddings.weight) -1])

    #TODO implement online learning
    pass

  def get_historic_user_embeddings(self):
    """
    :return: all embeddings in the shape (embedding_dim, total_number_of_users)
    """
    return self.model.user_embeddings.weight[:-1].numpy()

  def get_personal_user_embedding(self):
    """
    :return: personlized embedding in the shape (embedding_dim, 1)
    """
    return self.model.user_embeddings.weight[-1].numpy()



class RankingModule():
  def __init__(self, click_predictor : ClickPredictor):
    """
    initialize the ranking module
    :param
      the clickpredictor used to rank the headlines
    """
    self.click_predictor = click_predictor
    self.similarity_scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

  def rank_headlines(self, ids : List[int], headlines : List[str], user_id : str = "CUSTOM", take_top_k : int = 10, exploration_ratio : float = 0.2):
    """
    get the k top ranked (distinct) articles including some exploration articles
    :param
      ids (List[int]) : list of ids to re-identify returned headlines and avoid duplicated "user-impressions"
      headlines (List[str]) : the candidate headlines to rank
      user_id (str) : the user
      take_top_k (int) : the number of articles to return
      exploration_rati (float) : the ratio of articles which are used for exploration (in the range of 0.0 and 1.0)
    :return: list of tuples containing the sorted candidate strings, the id and the score for example: [("Lorem ipsum ...", 2, 0.78)]
    """
    assert len(headlines) > take_top_k
    assert exploration_ratio > 0.0 and exploration_ratio < 1.0
    assert len(headlines) == len(ids)
    
    scores, _ = self.click_predictor.calculate_scores(headlines, user_id)

    headlines_sorted = sorted(headlines, key=scores)
    ids_sorted =sorted(ids, key=scores)
    scores_sorted = sorted(scores)
    headlines_ids_sorted = zip(headlines_sorted, ids_sorted, scores_sorted)

    k_best = int(take_top_k * (1.0 - exploration_ratio))

    similarity_threshold = 0.5

    selected_headlines = []
    while len(selected_headlines) < k_best and len(headlines_ids_sorted) > 0:
      candidate, id, score = headlines_ids_sorted.pop(0)

      #calculate similarity to already existing candidates
      similar_headline_already_selected = False
      for item in selected_headline:
        scores = self.similarity_scorer.score(candidate, item)
        if scores > similarity_threshold:
          similar_headline_already_selected = True
          break

      if similar_headline_already_selected == False:
        selected_headlines.append((candidate, id, score))

    #reverse headlines for low ranked articles
    headlines_ids_sorted = reversed(headlines_ids_sorted)

    #fill the remaining space with exploration headlines
    while len(selected_headlines) < take_top_k and len(headlines_ids_sorted) > 0:
      candidate, id, score = headlines_ids_sorted.pop(0)

      #calculate similarity to already existing candidates
      similar_headline_already_selected = False
      for item in selected_headline:
        scores = self.similarity_scorer.score(candidate, item)
        if scores > similarity_threshold:
          similar_headline_already_selected = True
          break

      if similar_headline_already_selected == False:
        selected_headlines.append((candidate, id, score))

    return selected_headlines
