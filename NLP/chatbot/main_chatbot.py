import random
import json
import numpy as np
import tensorflow as tf
import pandas as pd
import re
from flask import Flask,render_template, request
import pickle

import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer


lemmatizer = WordNetLemmatizer()
intents = json.loads(open('./intents.json').read())

#to pandas dataframe
df_chat = pd.json_normalize(intents, record_path =['intents'])
df_chat = df_chat[['tag', 'patterns']]
df_chat = df_chat.explode(['patterns'], ignore_index=True)
df_chat = pd.DataFrame(df_chat)

classes = sorted(df_chat['tag'].unique())
tokenizer = word_tokenize
vectorizer_bow = pickle.load(open('260823_CountVecBoW_CBot.pkl', 'rb'))
model = pickle.load(open('260823_mdl_cbot_df.pkl', 'rb'))
response_list = []

def remove_unused(text):
    text = re.sub('[0-9]+', '', text) #untuk menghilangkan angka
    #untuk menghilangkan non-ASCII characters dan unicode
    text = text.encode('ascii', 'ignore').decode('utf-8') 
    text = re.sub(r'[^\x00-\x7f]', r'', text)  
    text = re.sub(r'[^\w]', ' ', text) #untuk menghiilangkan selain alpha numerik
    #untuk menghilangkan double atau lebih spasi
    space = ['    ', '   ', '  ']
    for i in space:
        text = text.replace(i, ' ')
    text = text.lower().strip()
#     text = text.translate(str.maketrans('', '', string.punctuation))
    return text

def lemm(word):
    words = tokenizer(word)
    words = [lemmatizer.lemmatize(text) for text in words]
    words = [" ".join(t for t in words)]
    return words

def bagofword(text):
    words= remove_unused(text)
    words = lemm(words)
    # print('words : {}'.format(words))
    bag = vectorizer_bow.transform(words)
    return bag.toarray()
    
def prediction(text):
    error_treshold = 0.70
    if text == "":
        result_index = 'empty'
    else :
        bow = bagofword(text)
        result = model.predict(bow)[0]
        # print('result : {}'.format(result))
        result_index = np.argmax(result)
        if result[result_index] >= error_treshold:
            result_index = np.argmax(result)
        else:
            result_index = 'unknown'
    
    return result_index

def get_response(res_index, intent_json):
    # print('clss : {}'.format(classes))
    # print('pred clss : {}'.format(res_index))
    if res_index=='unknown':
        result = ['I\'m sorry, I didnt get it', '???']
        result = random.choice(result)
    elif res_index=='empty':
        result = random.choice(["what?, type something!", "try to type something!", "Hello?", "you didnt type anything yet!"])
    else:
        tag = classes[res_index]
        # print('tag : {}'.format(tag))
        list_of_intent = intent_json['intents']
        for i in list_of_intent:
            if i['tag'] == tag:
                result = random.choice(i['responses'])
                break
    return result

def chat():
    print('Bot is Running')
    while True:
        message = input("Anda : ")
        if message.lower() == "stop" or message.lower() == "quit":
            print("Bot Stop Running")
            break
        ints = prediction(message)
        response = get_response(ints, intents)
        # print('Anda : ',message)
        print('Bot : ',response)
        print("##"*8)

chat()