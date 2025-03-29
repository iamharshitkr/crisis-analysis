import tweepy
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
import json
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure necessary nltk resources are available
nltk.download('stopwords')
nltk.download('vader_lexicon')

# API Credentials (Replace with your own keys)
API_KEY = "your_api_key"
API_SECRET = "your_api_secret"
ACCESS_TOKEN = "your_access_token"
ACCESS_SECRET = "your_access_secret"

# Authenticate with Twitter API
auth = tweepy.OAuthHandler(API_KEY, API_SECRET)
auth.set_access_token(ACCESS_TOKEN, ACCESS_SECRET)
api = tweepy.API(auth, wait_on_rate_limit=True)

# Keywords for Filtering
keywords = ["depressed", "suicidal", "anxious", "mental health", "overwhelmed", "self harm", "addiction help", "substance abuse", "panic attack", "hopeless", "therapy", "need support", "lonely", "trauma", "suicide prevention"]

# Extract Tweets
def fetch_tweets(keyword, count=100):
    tweets = []
    for tweet in tweepy.Cursor(api.search_tweets, q=keyword, lang="en", tweet_mode='extended').items(count):
        tweets.append([tweet.id, tweet.created_at, tweet.full_text, tweet.favorite_count, tweet.retweet_count])
    return tweets

# Fetch data and store in DataFrame
tweet_data = []
for word in keywords:
    tweet_data.extend(fetch_tweets(word, count=50))
df = pd.DataFrame(tweet_data, columns=["ID", "Timestamp", "Content", "Likes", "Retweets"])

# Text Cleaning
def clean_text(text):
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove special characters
    text = text.lower()  # Convert to lowercase
    text = ' '.join([word for word in text.split() if word not in stopwords.words('english')])
    return text

df['Cleaned_Content'] = df['Content'].apply(clean_text)

# Sentiment Analysis
def analyze_sentiment(text):
    sia = SentimentIntensityAnalyzer()
    sentiment_score = sia.polarity_scores(text)['compound']
    return "Positive" if sentiment_score > 0.05 else "Negative" if sentiment_score < -0.05 else "Neutral"

df['Sentiment'] = df['Cleaned_Content'].apply(analyze_sentiment)

# Crisis Risk Classification
def classify_risk(text):
    if any(phrase in text for phrase in ["dont want be here", "suicidal", "hopeless", "end life"]):
        return "High-Risk"
    elif any(phrase in text for phrase in ["feel lost", "need help", "struggling", "anxiety attack"]):
        return "Moderate Concern"
    else:
        return "Low Concern"

df['Risk_Level'] = df['Cleaned_Content'].apply(classify_risk)

# Save Data
df.to_csv("classified_crisis_tweets.csv", index=False)
df.to_json("classified_crisis_tweets.json", orient="records", indent=4)

# Visualization
plt.figure(figsize=(8,5))
sns.countplot(x=df['Sentiment'], palette="coolwarm")
plt.title("Sentiment Distribution")
plt.savefig("sentiment_distribution.png")

plt.figure(figsize=(8,5))
sns.countplot(x=df['Risk_Level'], palette="viridis")
plt.title("Crisis Risk Level Distribution")
plt.savefig("risk_level_distribution.png")

print("Task 2 complete: Sentiment and risk classification done. Charts saved!")
