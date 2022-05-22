# -*- coding: utf-8 -*-
"""
Created on Fri May 20 17:47:45 2022

@author: Helin
"""
import re
import pandas as pd
import string
import nltk
import time

#import dataset
dataset_columns = ["target", "ids", "date", "flag", "user", "text"]
dataset_encode = "ISO-8859-1"
data = pd.read_csv(r"D:\Users\Helin\anaconda3\depression\depressiondetection\2009trainingdata.csv", encoding = dataset_encode, names = dataset_columns)
data.head()

#data cleaning
data.drop(['ids','date','flag','user'],axis = 1,inplace = True)
print('Null values: ', data['text'].isnull().sum())
data['target'].value_counts()

#remove punctuation

data['clean_text']=data['text'].replace('!', '')
data.head()

#remove hyperlink
data['clean_text'] = data['clean_text'].astype(str).str.replace(r'http\S+', '') 
#remove emoji
data['clean_text'] = data['clean_text'].astype(str).str.replace('[^\w\s#@/:%.,_-]', '', flags=re.UNICODE)
#convert all words to lowercase
data['clean_text'] = data['clean_text'].astype(str).str.lower()
data.head()

#tokenization
start = time.time()

nltk.download('punkt')
def tokenize(text):
    split=re.split("\W+",text) 
    return split
data['clean_text_tokenize']=data['clean_text'].apply(lambda x: tokenize(x.lower()))

end = time.time()
print('Elapsed time: ',end - start)

#stopwords
start = time.time()

nltk.download('stopwords')
stopword = nltk.corpus.stopwords.words('english')
def remove_stopwords(text):
    text=[word for word in text if word not in stopword]
    return text
data['clean_text_tokenize_stopwords'] = data['clean_text_tokenize'].apply(lambda x: remove_stopwords(x))
data.head(10)

end = time.time()
print('Elapsed time: ',end - start)


# store label and text into new dataframe
new_data = pd.DataFrame()
new_data['text'] = data['clean_text']
new_data['label'] = data['target']
new_data['label'] = new_data['label'].replace(4,1)
# 1 for positive, 0 for negative
print(new_data.head())
print('Label: \n', new_data['label'].value_counts())

from sklearn.model_selection import train_test_split
X = new_data['text']
y = new_data['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=42)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


y_train.value_counts()

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

model = make_pipeline(TfidfVectorizer(analyzer = "word", ngram_range=(1,3)), MultinomialNB(alpha = 10))
start = time.time()

model.fit(X_train,y_train)

end = time.time()
print('Elapsed time: ',end - start)


validation = model.predict(X_test)
from sklearn.metrics import accuracy_score
accuracy_score(y_test, validation)


from sklearn.metrics import confusion_matrix
cf_matrix = confusion_matrix(y_test, validation)
cf_matrix


import seaborn as sns
import numpy as np
sns.heatmap(cf_matrix/np.sum(cf_matrix), annot=True, fmt='.001%', cmap='Blues')

from sklearn.metrics import classification_report
print(classification_report(y_test, validation))

train = pd.DataFrame()
train['label'] = y_train
train['text'] = X_train

def depression(s, model=model):
    pred = model.predict([s])
    predprob = model.predict_proba([s])
    if pred[0] == 1:
        return print('Not depressed\nProbability: ', np.max(predprob))
    else:
         return print('Depressed\nProbability: ', np.max(predprob))
     

depression('i love you')
depression('i wanna kill myself')


