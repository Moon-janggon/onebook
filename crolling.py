from bs4 import BeautifulSoup
from selenium import webdriver
import pandas as pd
from selenium.webdriver.common.by import By
import time
from openpyxl import Workbook
import re
import sys


def crolling(Url):
    options = webdriver.ChromeOptions()
    options.add_experimental_option("excludeSwitches", ["enable-logging"])
    driver = webdriver.Chrome(options=options)

    # driver = webdriver.Chrome()
    driver.set_window_size(800, 600)

    data_list=[]
    # 유튜브 열기
    # Url = 'https://www.youtube.com/watch?v=b_GYIVQdN-w'
    driver.get(Url)


    # 유튜브 댓글 끝까지 로딩
    last_page_height = driver.execute_script("return document.documentElement.scrollHeight")
    while True:
        driver.execute_script("window.scrollTo(0, document.documentElement.scrollHeight);")
        time.sleep(3.0)
        new_page_height = driver.execute_script("return document.documentElement.scrollHeight")

        if new_page_height == last_page_height:
            break
        last_page_height = new_page_height

        elements = driver.find_elements(By.CSS_SELECTOR, "#more-replies")
        for element in elements :
            driver.execute_script("arguments[0].click();", element)

    # 파싱
    html = driver.page_source
    soup = BeautifulSoup(html, 'html.parser')

    comments_list = soup.findAll('ytd-comment-thread-renderer', {'class': 'style-scope ytd-item-section-renderer'})

    for j in range(len(comments_list)):
        ## 댓글 내용
        comment = comments_list[j].find('yt-formatted-string', {'id': 'content-text'}).text
        comment = comment.replace('\n', '') # 줄 바뀜 없애기
        comment = comment.replace('\t', '') # 탭 줄이기
        # print(comment)

        replies = str(comments_list[j].find("div", id="replies").find_all('yt-formatted-string', {'id': 'content-text'}))
        replies = re.sub('<.+?>', '', replies, 0).strip()
        replies = replies.replace('\n', '')  # 줄 바뀜 없애기
        replies = replies.replace('\t', '')  # 탭 줄이기
        # for replie in replies :
        #     replie_text = replie.text

        youtube_id = comments_list[j].find('a', {'id': 'author-text'}).span.text
        youtube_id = youtube_id.replace('\n', '') # 줄 바뀜 없애기
        youtube_id = youtube_id.replace('\t', '') # 탭 줄이기
        youtube_id = youtube_id.strip()

        ## 댓글 좋아요 개수 (0인 경우 예외 처리)
        try:
            like_num = comments_list[j].find('span',
                                         {'id': 'vote-count-middle',
                                          'class': 'style-scope ytd-comment-action-buttons-renderer',
                                          'aria-label': re.compile('좋아요')}).text
            like_num = like_num.replace('\n', '') # 줄 바뀜 없애기
            like_num = like_num.replace('\t', '') # 탭 줄이기
            like_num = like_num.strip()
        except:
            like_num = 0

        try:
            dislike_num = comments_list[j].find('span',
                                         {'id': 'vote-count-middle',
                                          'class': 'style-scope ytd-comment-action-buttons-renderer',
                                          'aria-label': re.compile('싫어요')}).text
            dislike_num = dislike_num.replace('\n', '') # 줄 바뀜 없애기
            dislike_num = dislike_num.replace('\t', '') # 탭 줄이기
            dislike_num = dislike_num.strip()
        except:
            dislike_num = 0

        data = {'comment': comment, 'replie': replies, 'like_num': like_num, 'dislike_num': dislike_num}
        data_list.append(data)

        print(data)

    result_df = pd.DataFrame(data_list,
                             columns=['comment','replie','like_num','dislike_num'])

    result_df.to_excel('./youtube.xlsx', index=False)
    result_df.to_csv('./score.txt', sep = '\t', encoding='utf-8-sig')
    result_df.to_csv('./score.csv', sep = '\t', encoding='utf-8-sig')



    driver.close()





