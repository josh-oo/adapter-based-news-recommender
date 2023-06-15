from .CustomModel import BertForSequenceClassificationAdapters, BertConfigAdapters
from transformers import AutoTokenizer
from typing import List
import torch
from rouge_score import rouge_scorer #pip install rouge-score

class ClickPredictor():
  def __init__(self, huggingface_url : str, commit_hash : str = None, device : str = None):
    """
    load model from remote if not already present on disk
    :param
      huggingface_url (str) : url pointing to the huggingface repository for example "josh-oo/news-classifier"
      commit_has (str) : the corresponding hash if you want to use a certain version for example "748fad327878bbfbba33b55059259bbbb28046ad"
      device (str) : if you want to execute it on a certain device for example "cpu" or "cuda"
    """
    config = BertConfigAdapters.from_pretrained(huggingface_url, revision=commit_hash)
    self.model = BertForSequenceClassificationAdapters.from_pretrained(huggingface_url, revision=commit_hash)
    self.tokenizer = AutoTokenizer.from_pretrained(huggingface_url, revision=commit_hash)
    self.user_mapping = {} #TODO load user mapping
    self.device = device
    if self.device is not None:
      self.model.to(device)

  def calculate_scores(self, headlines : List[str], user_id : str = "CUSTOM"):
    """
    calculate scores for a list of headlines for a given user
    :param
      headlines : (List[str]) : the list of headlines
      user_id (str) : either the user_id corresponding to the MIND dataset (for example) to calculate scores for historic users
                      or "CUSTOM" to calculate scores for the current new user
    :return
      scores : (List[int]) : a score for each headline specifying a probability for a click event (1.0 = 100%)
      word_level_deviations (List[dict]): a list of dicts containing the headlines words and the deviation compared to the unpersonalize net
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
    model_outputs = self.model(**inputs, users=user_index.unsqueeze(dim=0))
    #TODO process model_outputs
    #dummy results:
    scores = [0.9, 0.3, 0.8]
    word_deviations = [
        {'Hello' : 0.1, 'World' : 0.0},
        {'The' : 0.1, 'quick' : 0.0, 'brown' : 0.3, 'fox' : 0.2},
        {'Lorem' : 0.1, 'ipsum' : 0.0, 'dolor' : 0.3, 'sit' : 0.2, 'amet': 0.0},
    ]

    return scores, word_deviations

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
    self.similarity_scorer = ouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

  def rank_headlines(self, headlines : List[str], user_id : str = "CUSTOM", take_top_k : int = 10, exploration_ratio : float = 0.2):
    """
    get the k top ranked (distinct) articles including some exploration articles
    :param
      headlines (List[str]) : the candidate headlines to rank
      user_id (str) : the user
      take_top_k (int) : the number of articles to return
      exploration_rati (float) : the ratio of articles which are used for exploration (in the range of 0.0 and 1.0)
    """
    scores, _ = self.click_predictor.calculate_scores(headlines, user_id)

    headlines_sorted = sorted(headlines, key=scores)

    k_best = int(take_top_k * (1.0 - exploration_ratio))

    similarity_threshold = 0.5

    selected_headlines = []
    while len(selected_headlines) < k_best:
      candidate = headlines_sorted[0]

      #calculate similarity to already existing candidates
      similar_headline_already_selected = False
      for item in selected_headline:
        scores = self.similarity_scorer.score(candidate, item)
        if scores > similarity_threshold:
          similar_headline_already_selected = True
          break

      if similar_headline_already_selected == False:
        selected_headlines.append(candidate)

    #reverse headlines for low ranked articles
    headlines_sorted = reversed(headlines_sorted)

    while len(selected_headlines) < k_exploration:
      candidate = headlines_sorted[0]

      #calculate similarity to already existing candidates
      similar_headline_already_selected = False
      for item in selected_headline:
        scores = self.similarity_scorer.score(candidate, item)
        if scores > similarity_threshold:
          similar_headline_already_selected = True
          break

      if similar_headline_already_selected == False:
        selected_headlines.append(candidate)

    return selected_headlines
