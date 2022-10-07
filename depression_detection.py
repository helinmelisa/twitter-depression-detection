import preprocessor as p
import pdb
from collections import Counter
import collections
import seaborn as sns
import tweepy
import re
from nltk.corpus import stopwords, twitter_samples
import numpy as np
import pandas as pd
import nltk
from nltk.stem import PorterStemmer
import string
import utils
#from utils import lookup
from nltk.tokenize import TweetTokenizer
from os import getcwd

#nltk.download('twitter_samples')
#nltk.download('stopwords')

api_key = ''
api_key_secret = ''
access_token = ''
access_token_secret = ''


# authenticate
auth = tweepy.OAuthHandler(api_key, api_key_secret)
auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth)


filePath = f"{getcwd()}/D:/Users/Helin/anaconda3/depression/tmp2/"
nltk.data.path.append(filePath)

# get the sets of positive and negative tweets
all_positive_tweets = twitter_samples.strings('positive_tweets.json')
all_negative_tweets = twitter_samples.strings('negative_tweets.json')

# split the data into two pieces, one for training and one for testing (validation set)
test_pos = all_positive_tweets[4000:]
train_pos = all_positive_tweets[:4000]
test_neg = all_negative_tweets[4000:]
train_neg = all_negative_tweets[:4000]

train_x = train_pos + train_neg
test_x = test_pos + test_neg

# avoid assumptions about the length of all_positive_tweets
train_y = np.append(np.ones(len(train_pos)), np.zeros(len(train_neg)))
test_y = np.append(np.ones(len(test_pos)), np.zeros(len(test_neg)))


def process_tweet(tweet):
    stemmer = PorterStemmer()
    stopwords_english = stopwords.words('english')
    # remove old style retweet text "RT"
    tweet = re.sub(r'^RT[\s]+', '', str(tweet))
    # remove hyperlinks
    tweet = re.sub(r'https?:\/\/.*[\r\n]*', '', str(tweet))
    # remove hashtags
    # only removing the hash # sign from the word
    tweet = re.sub(r'#', '', str(tweet))
    # tokenize tweets
    tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True,
                               reduce_len=True)
    tweet_tokens = tokenizer.tokenize(tweet)
    
    tweets_clean = []
    for word in tweet_tokens:
        if (word not in stopwords_english and  # remove stopwords
                word not in string.punctuation):  # remove punctuation
            # tweets_clean.append(word)
            stem_word = stemmer.stem(word)  # stemming word
            tweets_clean.append(stem_word)

    return tweets_clean


def count_tweets(result, tweets, ys):

    yslist = np.squeeze(ys).tolist()

    for y, tweet in zip(yslist, tweets):
        for word in process_tweet(tweet):
            pair = (word, y)
            if pair in result:
                result[pair] += 1
            else:
                result[pair] = 1
    return result

freqs = count_tweets({}, train_x, train_y)


import math
def train_naive_bayes(freqs, train_x, train_y):
  
    loglikelihood = {}
    logprior = 0

    freq_pos=0
    freq_neg=0
   
    vocab = set([pair[0] for pair in freqs.keys()])
    V = len(vocab)

   
    N_pos = N_neg = 0
    for pair in freqs.keys():
        
        if pair[1] > 0:
            N_pos += freqs.get(pair,0)
        else:
            N_neg += freqs.get(pair,0)

    D = len(train_y)

    D_pos = np.sum(train_y)
    D_neg = D-D_pos
    logprior = math.log(D_pos)-math.log(D_neg)

    for word in vocab:
     
        freq_pos =freqs.get((word,1),0)
        freq_neg =freqs.get((word,0),0)

        p_w_pos = (freq_pos+1)/(N_pos+V)
        p_w_neg = (freq_neg+1)/(N_neg+V)

        loglikelihood[word] = math.log(p_w_pos)-math.log(p_w_neg)

    return logprior, loglikelihood

logprior, loglikelihood = train_naive_bayes(freqs, train_x, train_y)
print(logprior)
print(len(loglikelihood))


def naive_bayes_predict(tweet, logprior, loglikelihood):

    word_l = process_tweet(tweet)
    p = 0
    p += logprior

    for word in word_l:
        if word in loglikelihood:
            p += loglikelihood[word]
    return p


def test_naive_bayes(test_x, test_y, logprior, loglikelihood, naive_bayes_predict=naive_bayes_predict):

    accuracy = 0 
    y_hats = []
    for tweet in test_x:
    
        if naive_bayes_predict(tweet, logprior, loglikelihood) > 0:
            y_hat_i = 1
        else:
           
            y_hat_i = 0
        y_hats.append(y_hat_i)
    error = np.mean([np.abs(y_hats[i] - test_y[i]) for i in range(len(y_hats))])
    accuracy = 1 - error
    return accuracy


print("Naive Bayes accuracy = %0.4f" %
      (test_naive_bayes(test_x, test_y, logprior, loglikelihood)))

def get_ratio(freqs, word):
   
    pos_neg_ratio = {'positive': 0, 'negative': 0, 'ratio': 0.0}
    pos_neg_ratio['positive'] = pd.DataFrame.lookup(freqs, word, 1)
    pos_neg_ratio['negative'] = pd.DataFrame.lookup(freqs, word, 0)
        pos_neg_ratio['ratio'] = (
        pos_neg_ratio['positive'] + 1) / (pos_neg_ratio['negative'] + 1)
    return pos_neg_ratio

from wordcloud import WordCloud
import matplotlib.pyplot as plt

def get_words_by_threshold(freqs, label, threshold, get_ratio=get_ratio):
    word_list = {}

    for key in freqs.keys():
        word, _ = key

        pos_neg_ratio = get_ratio(freqs, word)
        if label == 1 and pos_neg_ratio['ratio'] >= threshold:
    
            word_list[word] = pos_neg_ratio
        elif label == 0 and pos_neg_ratio['ratio'] <= threshold:
            word_list[word] = pos_neg_ratio
        print(word_list)
    
    return word_list

keywords = '#depressed -filter:retweets'
limit = 10000

tweets = tweepy.Cursor(api.search_tweets,q=keywords,lang = 'en',tweet_mode='extended').items(limit)

# tweets = api.user_timeline(screen_name=user, count=limit, tweet_mode='extended')

# create DataFrame
columns = ['User', 'Tweet']
data = []

for tweet in tweets:
    data.append([tweet.user.screen_name, tweet.full_text])

df = pd.DataFrame(data, columns=columns)


df.to_csv (r'D:\Users\Helin\anaconda3\depression\depressiondetection\user_data.csv', index = False, header=True)

p = naive_bayes_predict(data, logprior, loglikelihood)
print(p)

