
import openai
import json

openai.api_key = "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"

def analyze_sentiment(text):
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=f"Sentiment analysis: {text}",
        max_tokens=1,
        temperature=0,
        n=1,
        stop=None,
        timeout=10
    )
    sentiment = response.choices[0].text.strip().lower()
    return sentiment

def summarize(text):
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=f"Summarize: {text}",
        max_tokens=100,
        temperature=0.5,
        n=1,
        stop=None,
        timeout=10
    )
    summary = response.choices[0].text.strip()
    return summary

with open("./comments.txt",encoding='UTF=8') as f:
    comments = f.readlines()

sentiments = [analyze_sentiment(comment) for comment in comments]
summary = summarize(" ".join(comments))

print("Sentiments:", json.dumps(sentiments))
print("Summary:", summary)