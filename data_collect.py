import tweepy
import pandas as pd
import csv
import re
import twint
import io
import string
import configparser
from collections import Counter
from configparser import RawConfigParser
# Reading configuration file
# read configs
config = configparser.RawConfigParser()
config.read(r'C:\Users\Helin\PycharmProjects\pythonProject\config.ini')

api_key = config['twitter']['api_key']
api_key_secret = config['twitter']['api_key_secret']

access_token = config['twitter']['access_token']
access_token_secret = config['twitter']['access_token_secret']


# authenticate
auth = tweepy.OAuthHandler(api_key, api_key_secret)
auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth)

# search tweets
keywords = '#depression -filter:retweets'
limit = 300

tweets = tweepy.Cursor(api.search,q=keywords,count=100,lang = 'en',tweet_mode='extended').items(limit)


# tweets = api.user_timeline(screen_name=user, count=limit, tweet_mode='extended')

# create DataFrame
columns = ['ID', 'User', 'Tweet', 'Date','Hashtags']
data = []

for tweet in tweets:
    data.append([tweet.id, tweet.user.screen_name, tweet.full_text, tweet.created_at, tweet.entities.get('hashtags')])


df = pd.DataFrame(data, columns=columns)


import  csv
#print(df)
df.to_csv ('C:\\Users\\Helin\\PycharmProjects\\pythonProject\\tweet_data.csv', index = False, header=True)


keywords = '#depressed -filter:retweets'
limit = 1000

tweets = tweepy.Cursor(api.search_tweets,q=keywords,count=100,lang = 'en',tweet_mode='extended').items(limit)

# tweets = api.user_timeline(screen_name=user, count=limit, tweet_mode='extended')

# create DataFrame
columns = ['ID', 'User', 'Tweet', 'Date','Hashtags']
data1 = []

for tweet in tweets:
    data1.append([tweet.id, tweet.user.screen_name, tweet.full_text, tweet.created_at, tweet.entities.get('hashtags')])

df1 = pd.DataFrame(data1, columns=columns)

df1.to_csv (r'C:\Users\Helin\PycharmProjects\pythonProject\tweet_data2.csv', index = False, header=True)
