from test import sentiment
from crolling import crolling
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')       #서버에서, 화면에 표시하기 위해서 필요
import seaborn as sns
import altair as alt               ##https://altair-viz.github.io/
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

def create_plot(data):
    data = data['sentiment category'].value_counts()
    labels = list(data.index)
    values = list(data)
    fig = px.pie(labels, values=values,
    names= labels, color = labels,
    color_discrete_map = {
        'Positive': '#1f77b4',
        'Negative': '#d62728',
        'Neutral': '#7f7f7f'
    })
    return fig



sidebar = st.sidebar
sidebar.header('Input Options')
video_url=sidebar.text_input('youtube url')
# multiselect

options = sidebar.multiselect('What sentiment category do you want to see?',
    ['ALL','Positive', 'Negative'])

if sidebar.button('search'):
    # thumbnail
    st.header('Thumbnail')
    crolling(video_url)
    sentiment()
    st.image(get_thumbnail(video_url))

    # dataframe
    st.header('Comments')
    df = pd.read_csv('./score.csv', sep = '\t', encoding='utf-8-sig')
    st.dataframe(df)

    st.header('Sentiment')
    df = pd.read_csv('./comments.txt',header = 0, sep = '|' ,encoding='UTF-8')
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

    #pie plot
    #fig = px.pie(values=)
    #st.header('Pie plot')
    #st.plotly_chart(fig, use_container_width=True)

    # multiselect


    # # pie plot
    # fig = create_plot(data)
    # st.header('Pie plot')
    # st.plotly_chart(fig, use_container_width=True)