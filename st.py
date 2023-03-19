import time
import streamlit as st
import pandas as pd
from tqdm import tqdm
from api import get_comments
from test import sentiment
from newModel import sentiment_prediction
import plotly.express as px



st.title('유튜브 댓글 분석/요약 프로그램')
expander_bar = st.expander("About\n")
st.markdown("""
이 앱은 유튜브 댓글을 검색해 다양한 방법으로 분석한다.
""")
expander_bar.markdown("""
* **Python libraries:**  pandas, streamlit, numpy, plotly, nltk, konlpy, textrank
""")

def url_to_id(video_url):
    id = video_url.split('?v=')[1]
    if '&' in id:
        id = id.split('&')[0]
    return id

def get_thumbnail(url):
    id = url_to_id(url)
    img = 'https://img.youtube.com/vi/{}/0.jpg'.format(id)
    return img


sidebar = st.sidebar
sidebar.header('Input Options')
video_url=sidebar.text_input('youtube url')
# multiselect

options = sidebar.multiselect('What sentiment category do you want to see?',
    ['ALL','Positive', 'Negative'])

if sidebar.button('search'):
    # thumbnail
    st.header('Thumbnail')
    st.image(get_thumbnail(video_url))

    # dataframe
    st.header('전체 댓글 요약문')
    with st.spinner('댓글을 수집하고 있습니다...'):
        progress_bar = st.progress(0)  # 프로그레스 바 추가
        df_all_summary, df_top_summary = get_comments(video_url)
        for i in tqdm(range(100)):  # 진행률 업데이트
            time.sleep(0.1)
            progress_bar.progress(i + 1)
    sentiment_prediction()
    st.dataframe(df_all_summary)

    st.header('좋아요 수 상위 10개 댓글의 요약문')
    st.dataframe(df_top_summary)

    st.header('Sentiment')
    df = pd.read_csv('./comments.txt',header = 0, sep = '|' ,encoding='UTF=8')
    df.head()
    sum1 = df['label'].value_counts()['G']
    sum2 = df['label'].value_counts()['B']
    data= { "sum":[sum1,sum2]}
    dfpie= pd.DataFrame(data, index = ['gung','bu'])

    fig1= px.pie(dfpie, values='sum')

    import plotly.graph_objects as go

    labels = ['긍정', '부정', ]
    values = [sum1,sum2]

    fig = go.Figure(data=[go.Pie(labels=labels, values=values, textinfo='label+percent',
                                 insidetextorientation='radial'
                                 )])
    st.plotly_chart(fig, use_container_width=True)

    st.header('Comments')
    df = pd.read_csv('./score.csv', sep='\t', encoding='utf-8-sig')
    # ds = pd.read_csv('./comments.txt', sep='|',encoding='utf-8-sig')
    st.dataframe(df)