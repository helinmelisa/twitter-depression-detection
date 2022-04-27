# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 13:39:44 2022

@author: Helin
"""
#kütüphaneler
import nest_asyncio
nest_asyncio.apply()
import pandas as pd
import twint
import re

from google.colab import drive
drive.mount('/content/gdrive')

#2022 yılına ait tweet ekleme
depress_tags = ["#depressed", "#depression", "#loneliness", "#hopelessness"]

content = {}
for i in range(len(depress_tags)):
    print(depress_tags[i])
    c = twint.Config()
    
    c.Format = "Tweet id: {id} | Tweet: {tweet}"
    c.Search = depress_tags[i]
    c.Limit = 1000
    c.Year = 2022
    c.Store_csv = True
    c.Store_Object = True
    #drive yolu
    c.Output = "/content/gdrive//data/dataset_en_all7.csv"
    c.Hide_output = True
    c.Stats = True
    c.Lowercase  = True
    c.Filter_retweets = True
    twint.run.Search(c)

#overlap olmasın diye 2021 yılına ait tweet ekleme
depress_tags = ["#depressed", "#depression"]

content = {}
for i in range(len(depress_tags)):
    c = twint.Config()
    
    c.Format = "Tweet id: {id} | Tweet: {tweet}"
    c.Search = depress_tags[i]
    c.Limit = 1000
    c.Year = 2021
    c.Store_csv = True
    c.Store_Object = True
    #drive yolu
    c.Output = "/content/gdrive/My Drive/data/dataset_en_al19.csv"
    c.Hide_output = True
    c.Stats = True
    c.Lowercase  = True   
    twint.run.Search(c)

#overlap olmasın diye 2020 yılına ait tweet ekleme    
depress_tags = ["#depressed", "#depression"]

content = {}
for i in range(len(depress_tags)):
    c = twint.Config()
    
    c.Format = "Tweet id: {id} | Tweet: {tweet}"
    c.Search = depress_tags[i]
    c.Limit = 1000
    c.Year = 2020
    c.Store_csv = True
    c.Store_Object = True
    c.Output = "/content/gdrive/My Drive/data/dataset_en_al19.csv"
    c.Hide_output = True
    c.Stats = True
    c.Lowercase  = True   
    twint.run.Search(c)
    
#veri ekleme   
df1 = pd.read_csv("/content/gdrive/My Drive/data/dataset_en_all7.csv")
df2 = pd.read_csv("/content/gdrive/My Drive/data/dataset_en_al19.csv")
df_all = pd.concat([df1, df2])

#her veriseti için boyut kontrolü
len(df1), len(df2), len(df_all)
df1.hashtags.value_counts()
len(df_all.id.value_counts())

#id ve tweet içeriğine göre setleri birleştirme ve duplicate kaldırma
df_all = df_all.drop_duplicates(subset =["id"]) 
df_all.shape
pd.set_option('display.max_colwidth', -1)
df_all.head()
df_all.hashtags.value_counts().head(20)
df_all[df_all["hashtags"] =="['#depression', '#hopelessness', '#invisibleillness', '#robinwilliams', '#socialmedia', '#suicide']"]


#alakasız sütunları filtreleme
selection_to_remove = ["#mentalhealth", "#health", "#happiness", "#mentalillness", "#happy", "#joy", "#wellbeing"]
#pozitif veya tıbbi kayıtları filtreleme
mask1 = df_all.hashtags.apply(lambda x: any(item for item in selection_to_remove if item in x))
df_all[mask1].tweet.tail()
#belli taglere göre sonucu inceleme
df_all[mask1==False].tweet.head(10)

#yukardaki sonuca göre mask1'i uyguluyoruz
df_all = df_all[mask1==False]
len (df_all)

#reklam olabileceği için 3'ten fazla hashtag içerenleri kaldırıyoruz
mask2 = df_all.hashtags.apply(lambda x: x.count("#") < 4)
df_all = df_all[mask2]
len(df_all)
df_all.head()

#genelde retweet olabileceği için yanıtlama tweetlerini kaldırıyoruz
mask3 = df_all.mentions.apply(lambda x: len(x) < 5)
df_all = df_all[mask3]
len(df_all)
#hashtag değerini tekrar kontrol etmemiz gerekiyor
df_all.hashtags.value_counts().head(20)
df_all.tweet.tail(10)

#x den daha az karakterde olanları kaldırıyoruz
mask4a = df_all.tweet.apply(lambda x: len(x) > 25)
df_all = df_all[mask4a]
len(df_all)
mask4b = df_all.tweet.apply(lambda x: x.count(" ") > 5)
df_all = df_all[mask4b]
len(df_all)
df_all.tweet


#reklam içeriği olabiceğinden url içerenleri de kaldırıyoruz
mask5 = df_all.urls.apply(lambda x: len(x) < 5)
#silinen verileri inceleme
df_all[mask5==False].tweet.head(10), df_all[mask5==False].tweet.tail(10)
df_all = df_all[mask5]
len(df_all)

#tüm hashtagler kaldırılmış sadece text içeren sütun oluşturalım
df_all["mod_text"] = df_all["tweet"].apply(lambda x: re.sub(r'#\w+', '', x))
df_all.mod_text.head(15), df_all.mod_text.tail(15)

#hashtag sayısı
df_all.hashtags.value_counts().head(20)
df_all.columns
col_list = ["id", "conversation_id", "date", "username", "mod_text", "hashtags", "tweet"]
df_final1 = df_all[col_list]
df_final1 = df_final1.rename(columns={"mod_text": "tweet_processed", "tweet": "tweet_original"})
df_final1["target"] = 1
df_final1.head()
len(df_final1)
df_final1_1 = df_final1[:400]
df_final1_2 = df_final1[400:800]
df_final1_3 = df_final1[800:]
len(df_final1_1), len(df_final1_2), len(df_final1_3), 

#verilerin son hali
df_final1.to_csv("/content/gdrive/My Drive/data/tweets_final.csv")

df_final1_1.to_csv("/content/gdrive/My Drive/data/tweets_final_1.csv")
df_final1_2.to_csv("/content/gdrive/My Drive/data/tweets_final_2.csv")
df_final1_3.to_csv("/content/gdrive/My Drive/data/tweets_final_3.csv")

df_all.to_csv("/content/gdrive/My Drive/data/tweets_v3.csv")


users = df_all.username

content = {}
for i in users: #users1['Names']:

    
    c = twint.Config()
    c.Search = "#depressed"
    c.Username = "noneprivacy"
    c.Username = i
    c.Format = "Tweet id: {id} | Tweet: {tweet}"
    c.Limit = 100
    c.Store_csv = True
    c.Store_Object = True
    c.Output = "/content/gdrive/My Drive/data/dataset_v3.csv"
    c.Hide_output = True
    c.Stats = True
    c.Lowercase  = True
    twint.run.Search(c)
    
    
    
 print(i)





