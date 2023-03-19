import numpy as np
import pandas as pd
import re
import json
from konlpy.tag import Okt
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import load_model

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
model= load_model('./content/sample_data/my_models')
okt = Okt()
tokenizer = Tokenizer()
word_vocab = tokenizer.word_index


def sentiment():
    DATA_CONFIGS = 'data_configs.json'
    prepro_configs = json.load(open('./content/sample_data/CLEAN_DATA/' + DATA_CONFIGS, 'r'))
    word_vocab=prepro_configs['vocab']

    tokenizer.fit_on_texts(word_vocab)

    MAX_LENGTH = 8  # 문장최대길이
    train_data = pd.read_excel('./api.xlsx')
    #train_data = pd.read_csv('./api.csv', sep='\t', encoding='utf-8-sig', index=False)
    f = open('./comments.txt', 'w',encoding='UTF-8')
    f.write('comment|label\n')

    for sentence in train_data['comment']:
        # sentence = input("문장 :")
        presentence=sentence

        sentence = re.sub(r'[^ㄱ-ㅎㅏ-ㅣ가-힣\\s ]', '', str(sentence))
        stopwords = ['은', '는', '이', '가', '하', '아', '것', '들', '의', '있', '되', '수', '보', '주', '등',
                     '한']  # 불용어 추가할 것이 있으면 이곳에 추가
        sentence = okt.morphs(sentence, stem=True)  # 토큰화

        sentence = [word for word in sentence if not word in stopwords] # 불용어 제거
        vector = tokenizer.texts_to_sequences(sentence)
        pad_new = pad_sequences(vector, maxlen=MAX_LENGTH)  # 패딩
        #print('변환된 문자 결과값: ',vector)
        model.load_weights('./content/sample_data/DATA_OUT/cnn_classifier_kr/weights.h5')  # 모델 불러오기
        try:
            predictions = model.predict(pad_new)
            predictions = float(predictions.squeeze(-1)[1])
        except:
            continue
            #print(predictions)
        if (predictions > 0.5):
            print("{:.2f}% 확률로 긍정 리뷰입니다.\n".format(predictions * 100))
            f.write(presentence)
            f.write('|G\n')



        else:
            print("{:.2f}% 확률로 부정 리뷰입니다.\n".format((1 - predictions) * 100))
            f.write(presentence)
            f.write('|B\n')
    f.close()

    df= pd.DataFrame(train_data)
    ds= pd.read_csv('comments.txt',header = 0, sep = '|' ,encoding='UTF=8')
    label = ds['label']
    df['sentiment']= label

    df.to_csv('score.csv',sep='\t')
