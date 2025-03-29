import tweepy
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
import json

# Ensure necessary nltk resources are available
nltk.download('stopwords')

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
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    text = ' '.join([word for word in text.split() if word not in stopwords.words('english')])
    return text

df['Cleaned_Content'] = df['Content'].apply(clean_text)

# Save Cleaned Data
df.to_csv("cleaned_crisis_tweets.csv", index=False)
df.to_json("cleaned_crisis_tweets.json", orient="records", indent=4)

print("complete: Data extracted, cleaned, and saved!")
