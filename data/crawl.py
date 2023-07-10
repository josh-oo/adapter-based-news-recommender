from transformers import AutoModelForSequenceClassification, TFAutoModelForSequenceClassification
from transformers import AutoTokenizer
import numpy as np
from scipy.special import expit
# encoding = 'latin-1'
def crawl():
    with open('data/ArticleCrawl.txt', 'r', encoding='utf-16') as file:
        crawled_headlines = [line.rstrip() for line in file]
    return crawled_headlines

def classify(crawled_headlines = None):
    MODEL = f"cardiffnlp/tweet-topic-21-multi"
    tokenizer = AutoTokenizer.from_pretrained(MODEL)

    # PT
    model = AutoModelForSequenceClassification.from_pretrained(MODEL)
    class_mapping = model.config.id2label
    classified_headlines = np.empty((300,3), dtype = tuple)

    for i in range(len(crawled_headlines)):
        text = crawled_headlines[i]
        tokens = tokenizer(text, return_tensors='pt')
        output = model(**tokens)
        scores = output[0][0].detach().numpy()
        scores = expit(scores)
        predictions = (scores >= 0.5) * 1
        max_index = np.argmax(predictions)
        label = class_mapping[max_index]
        if label == 'celebrity_&_pop_culture':
            label = 'celebrity & pop culture'
        if label == 'learning_&_educational':
            label = 'learning and education'
        if label == 'news_&_social_concern':
            label = 'news'
        if label =='film_tv_&_video':
            label = 'tv &  movies'
        if label == 'youth_&_student_life':
            label = 'youth & student life'
        if label == 'other_hobbies':
            label = 'other hobbies'
        if label == 'fashion_&_style':
            label = 'fashion & style'
        if label == 'diaries_&_daily_life':
            label = 'dialy lifes'
        if label == 'fitness_&_health':
            label = 'fitness & health'
        if label == 'business_&_entrepreneurs':
            label = 'business'
        if label =='travel_&_adventure':
            label = 'travel'
        if label == 'arts_&_culture':
            label = 'arts & culture'
        if label =='science_&_technology':
            label = 'science & tech'
        if label == 'food_&_dining':
            label = 'food & dining'
        classified_headlines[i] = [i, label, text]

    with open('data/crawled_classified_headlines.npy', 'wb') as f:
        np.save(f, classified_headlines)
    print(classified_headlines)
    
crawled_headlines = crawl()
classify(crawled_headlines)


