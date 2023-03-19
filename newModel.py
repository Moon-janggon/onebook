import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import os
import urllib.request
from konlpy.tag import Okt
from tqdm import tqdm
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, Dense, LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

loaded_model = load_model('best_model.h5')
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
f = open('./comments.txt', 'w', encoding='UTF-8')
f.write('comment|label|\n')
okt =Okt()
max_len=30
stopwords=['의','가','이','은','들','는','좀','잘','걍','과','도','를','으로','자','에','와','한','하다']

with open('tokenizer.pickle', 'rb') as handle:
  tokenizer = pickle.load(handle)

def write_sentiment(sen_sentence,score):
  if(score>0.4):
    f.write(str(sen_sentence))
    f.write('|G|\n')
  else:
    f.write(str(sen_sentence))
    f.write('|B|\n')


def sentiment_predict(new_sentence):
  born_sentence = new_sentence
  new_sentence = re.sub(r'[^ㄱ-ㅎㅏ-ㅣ가-힣 ]','', str(new_sentence))
  new_sentence = okt.morphs(new_sentence, stem=True) # 토큰화
  new_sentence = [word for word in new_sentence if not word in stopwords] # 불용어 제거

  encoded = tokenizer.texts_to_sequences([new_sentence]) # 정수 인코딩
  pad_new = pad_sequences(encoded, maxlen = max_len) # 패딩
  score = float(loaded_model.predict(pad_new)) # 예측
  if(score > 0.4):
    print("{:.2f}% 확률로 긍정 리뷰입니다.\n".format(score * 100))
    write_sentiment(born_sentence,score)

  else:
    print("{:.2f}% 확률로 부정 리뷰입니다.\n".format((1 - score) * 100))
    write_sentiment(born_sentence, score)

def sentiment_prediction():
  train_data = pd.read_excel('./api.xlsx')

  for sentence in train_data['comment']:
    sentiment_predict(sentence)

  df= pd.DataFrame(train_data)
  ds= pd.read_csv('comments.txt',header = 0, sep = '|' ,encoding='UTF=8')
  label = ds['label']
  print(ds['label'])
  df['sentiment']= label

  df.to_csv('score.csv',sep='\t')