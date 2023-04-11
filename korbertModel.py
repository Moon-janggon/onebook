import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import gluonnlp as nlp
import numpy as np
from tqdm.notebook import tqdm
import os

from kobert import get_tokenizer
from kobert import get_pytorch_kobert_model

from transformers import AdamW
from transformers.optimization import get_cosine_schedule_with_warmup


# 위에서 설정한 tok, max_len, batch_size, device를 그대로 입력
device=torch.device("cuda:0")
tokenizer = get_tokenizer()
bertmodel, vocab = get_pytorch_kobert_model(cachedir=".cache")
tok = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower=False)
max_len = 64
batch_size = 64
class BERTDataset(Dataset):
  def __init__(self, dataset, sent_idx, label_idx, bert_tokenizer, max_len, pad, pair):
    transform = nlp.data.BERTSentenceTransform(bert_tokenizer, max_seq_length=max_len, pad=pad, pair=pair)

    self.sentences = [transform([i[sent_idx]]) for i in dataset]
    self.labels = [np.int32(i[label_idx]) for i in dataset]

  def __getitem__(self, i):
    return (self.sentences[i] + (self.labels[i],))

  def __len__(self):
    return (len(self.labels))
class BERTClassifier(nn.Module):
  def __init__(self,
               bert,
               hidden_size=768,
               num_classes=2,
               dr_rate=None,
               params=None):
    super(BERTClassifier, self).__init__()
    self.bert = bert
    self.dr_rate = dr_rate

    self.classifier = nn.Linear(hidden_size, num_classes)
    if dr_rate:
      self.dropout = nn.Dropout(p=dr_rate)

  def gen_attention_mask(self, token_ids, valid_length):
    attention_mask = torch.zeros_like(token_ids)
    for i, v in enumerate(valid_length):
      attention_mask[i][:v] = 1
    return attention_mask.float()

  def forward(self, token_ids, valid_length, segment_ids):
    attention_mask = self.gen_attention_mask(token_ids, valid_length)

    _, pooler = self.bert(input_ids=token_ids, token_type_ids=segment_ids.long(),
                          attention_mask=attention_mask.float().to(token_ids.device))
    if self.dr_rate:
      out = self.dropout(pooler)
    else:
      out = pooler
    return self.classifier(out)

dataset_train = nlp.data.TSVDataset(".cache/ratings_train.txt", field_indices=[1, 2], num_discard_samples=1)
dataset_test = nlp.data.TSVDataset(".cache/ratings_test.txt", field_indices=[1, 2], num_discard_samples=1)

## Setting parameters
max_len = 64
batch_size = 64
warmup_ratio = 0.1
num_epochs = 1
max_grad_norm = 1
log_interval = 200
learning_rate = 5e-5

data_train = BERTDataset(dataset_train, 0, 1, tok, max_len, True, False)
data_test = BERTDataset(dataset_test, 0, 1, tok, max_len, True, False)



train_dataloader = torch.utils.data.DataLoader(data_train, batch_size=batch_size, num_workers=0)
test_dataloader = torch.utils.data.DataLoader(data_test, batch_size=batch_size, num_workers=0)

model = BERTClassifier(bertmodel, dr_rate=0.5).to(device)

# Prepare optimizer and schedule (linear warmup and decay)
no_decay = ['bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]

optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)
loss_fn = nn.CrossEntropyLoss()

t_total = len(train_dataloader) * num_epochs
warmup_step = int(t_total * warmup_ratio)

scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_step, num_training_steps=t_total)



def getSentimentValue(comment, tok, max_len, batch_size, device):
  model = BERTClassifier(bertmodel, dr_rate=0.5).to(device)
  model.load_state_dict(torch.load('./model2.pt'))
  model.eval()

  commnetslist = [] # 텍스트 데이터를 담을 리스트
  commentonly=[]
  emo_list = [] # 감성 값을 담을 리스트
  per_list = []
  for c in comment: # 모든 댓글
    commnetslist.append( [c, 5] )
    commentonly.append(c) # [댓글, 임의의 양의 정수값] 설정


  pdData = pd.DataFrame( commnetslist, columns = [['댓글', '감성']] )
  pdData = pdData.values
  test_set = BERTDataset(pdData, 0, 1, tok, max_len, True, False)
  test_input = torch.utils.data.DataLoader(test_set, batch_size=batch_size, num_workers=0)

  for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(test_input):
    token_ids = token_ids.long().to(device)
    segment_ids = segment_ids.long().to(device)
    valid_length= valid_length
    # 이때, out이 예측 결과 리스트
    out = model(token_ids, valid_length, segment_ids)

    # e는 2가지 실수 값으로 구성된 리스트
    # 0번 인덱스가 더 크면 부정, 긍정은 반대
    for e in out:
      if e[0]>e[1]: # 부정
        value = '0'

      else: #긍정
        value = "1"

      emo_list.append(value)


  df =pd.DataFrame({
    'comment' : commentonly,
    'label' : emo_list

  }) # 텍스트 데이터에 1대1 매칭되는 감성값 리스트 반환

  df.transpose()
  return df



  #df.to_csv("./sentiment.txt", sep='\t', encoding='utf-8-sig')


def koberta(data):
# print(lst)
  #train_data = pd.read_csv('./10000.csv',sep = '\t' ,encoding='utf-8-sig')
  train_data = data
  clst= train_data['comment']
  df= getSentimentValue(clst, tok, max_len, batch_size, device)
  if not os.path.exists('ForTrainning.txt'):
      df.to_csv('sentiment_out.txt', sep='\t', mode='w', encoding='utf-8-sig')
      df.to_csv('ForTrainning.txt',sep='\t', mode='w', encoding='utf-8-sig')
  else:
      df.to_csv('sentiment_out.txt', sep='\t', mode='w', encoding='utf-8-sig')
      df.to_csv('ForTrainning.txt', sep='\t', mode='a', encoding='utf-8-sig')
  return df
# df=pd.DataFrame(dataframe)
# df.transpose()
# df.to_csv("./sentiment.txt", sep='\t' , encoding='utf-8-sig')
