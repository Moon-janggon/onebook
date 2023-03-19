import pandas as pd
import re
from googleapiclient.discovery import build
from google.oauth2 import service_account
import urllib.parse
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer
import pandas as pd
from datetime import datetime
from konlpy.tag import Okt
from langdetect import detect
import nltk
nltk.download('punkt')

# API 키 인증
SCOPES = ["https://www.googleapis.com/auth/youtube.force-ssl"]
#SERVICE_ACCOUNT_FILE = "C:/Python/credentials.json"
SERVICE_ACCOUNT_FILE = "C:/Users/moon/PycharmProjects/onebook/content/credentials.json"

creds = None
creds = service_account.Credentials.from_service_account_file(
            SERVICE_ACCOUNT_FILE, scopes=SCOPES)
youtube = build("youtube", "v3", credentials=creds)

def get_comments(url):

    query_string = urllib.parse.urlparse(url).query
    parameters = urllib.parse.parse_qs(query_string)
    video_id = parameters["v"][0]
    data_list = []
    nextPageToken = None

    while True:
        # API request
        response = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=100,
            pageToken=nextPageToken
        ).execute()

        # Extracting comments and relevant information from response
        for item in response["items"]:
            comment = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
            comment = comment.replace('\n', '').replace('\t', '')
            comment = re.sub(r'<[^>]+>', '', comment)  # Remove HTML tags
            comment = comment.replace('&amp;', '&')  # Replace &amp; with &
            like_num = item["snippet"]["topLevelComment"]["snippet"]["likeCount"]
            dislike_num = item["snippet"]["topLevelComment"]["snippet"].get("dislikeCount", 0)

            data = {'comment': comment, 'like_num': like_num, 'dislike_num': dislike_num}
            data_list.append(data)

        try:
            nextPageToken = response["nextPageToken"]
        except:
            break

    result_df = pd.DataFrame(data_list, columns=['comment', 'like_num', 'dislike_num'])
    result_df['like_num'] = result_df['like_num'].astype(int)
    # 좋아요 수 높은 순으로 정렬
    result_df.sort_values('like_num', ascending=False, inplace=True)
    # # 'comment' 열에서 '@'로 시작하는 문자열을 빈 문자열로 바꿉니다.
    # result_df['comment'] = result_df['comment'].apply(lambda x: re.sub(r'@(\w+)', '', x))
    result_df.to_csv('./api.csv', sep='\t', encoding='utf-8-sig', index=False)
    result_df.to_excel('./api.xlsx')

    # NaN 값을 빈 문자열로 변환
    result_df.fillna('', inplace=True)

    # 댓글 텍스트 추출
    comment_text = result_df['comment'].astype(str)

    # 시간 형식에 맞는 댓글 제거
    comment_text = comment_text.apply(lambda x: re.sub(r'\b\d{1,2}:\d{1,2}\b', '', x))

    # 언어 감지하여 요약문 생성
    LANGUAGE = detect(comment_text.iloc[0])
    SENTENCES_COUNT = 5

    if LANGUAGE == 'ko':
        okt = Okt()
        parser = PlaintextParser.from_string(" ".join(comment_text), Tokenizer('korean'))
        summarizer = LexRankSummarizer()
        summarizer.stop_words = okt.nouns(" ".join(comment_text)) + ["있다", "하는", "위해", "이번"]
        summary_sentences = [str(sentence) for sentence in summarizer(parser.document, SENTENCES_COUNT)]
    elif LANGUAGE == 'en':
        parser = PlaintextParser.from_string(" ".join(comment_text), Tokenizer('english'))
        summarizer = LexRankSummarizer()
        summarizer.stop_words = ["the", "a", "an", "is", "are", "and"]
        summary_sentences = [str(sentence) for sentence in summarizer(parser.document, SENTENCES_COUNT)]
    else:
        print(f"Unsupported language detected: {LANGUAGE}")
        summary_sentences = []

    all_summary = '\n'.join(summary_sentences)
    df_all_summary = pd.DataFrame({'All_Summary': summary_sentences})
    df_all_summary.to_csv('./All_summary.csv', sep='\t', encoding='utf-8-sig', index=False)

    # 좋아요 수 상위 10개 댓글의 요약문 생성
    TOP_N = 10  # 상위 10개 댓글
    SENTENCES_COUNT = 3  # 요약문은 3문장으로 생성
    top_n_comments = result_df['comment'][:TOP_N].astype(str)

    parser = PlaintextParser.from_string(" ".join(top_n_comments), Tokenizer(LANGUAGE))
    summarizer = LexRankSummarizer()
    summarizer.stop_words = [" "]
    summary_sentences = [str(sentence) for sentence in summarizer(parser.document, SENTENCES_COUNT)]

    # 요약된 문장 출력
    top_summary = '\n'.join(summary_sentences)
    df_top_summary = pd.DataFrame({'Top_Summary': summary_sentences})
    df_top_summary.to_csv('./Top_summary.csv', sep='\t', encoding='utf-8-sig', index=False)

    return df_all_summary, df_top_summary