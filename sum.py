from transformers import AutoTokenizer, BertForMaskedLM
from tqdm import tqdm

# 모델과 토크나이저 로드
tokenizer = AutoTokenizer.from_pretrained('beomi/kcbert-base')
model = BertForMaskedLM.from_pretrained('beomi/kcbert-base')

# csv 파일 불러오기
import pandas as pd
df = pd.read_csv('./mecab.csv', sep='\t')

# 제목 요약
def summarize(text, max_length=50):
    if len(text) > max_length:
        text = text[:max_length]
    input_ids = tokenizer.encode(text, return_tensors='pt')
    output = model.generate(input_ids, max_length=300, num_beams=5)
    summary = tokenizer.decode(output[0], skip_special_tokens=True)
    return summary

with tqdm(total=len(df)) as pbar:
    summaries = []
    for index, row in df.iterrows():
        summary = summarize(row['comment'])
        summaries.append(summary)
        pbar.update(1)

df['summary'] = summaries

# 요약된 데이터 csv 파일에 저장
df.to_csv('./summarize.csv', index=False)

print('finish')