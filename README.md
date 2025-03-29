GSoC Crisis Analysis

Overview

This project is designed to analyze crisis-related discussions on social media, classify posts based on sentiment and risk level, and visualize crisis trends geographically. It performs the following tasks:

Social Media Data Extraction & Preprocessing

Extracts tweets using the Twitter API based on predefined keywords.

Cleans and preprocesses text for NLP analysis.

Stores structured data in CSV and JSON formats.

Sentiment & Crisis Risk Classification

Uses VADER for sentiment analysis.

Classifies posts into three risk levels: High-Risk, Moderate Concern, and Low Concern.

Generates a sentiment and risk level distribution visualization.

Crisis Geolocation & Mapping

Extracts geolocation data from tweets.

Generates an interactive heatmap of crisis-related discussions using Folium.

Identifies and displays the top 5 locations with the highest crisis discussions.

Installation

Prerequisites

Ensure you have Python 3 installed along with the required dependencies.

Install Required Libraries

pip install tweepy pandas nltk textblob matplotlib seaborn folium geopy

NLTK Data Download

import nltk
nltk.download('stopwords')
nltk.download('vader_lexicon')

Usage

1. Configure API Credentials

Replace the following placeholders in gsoc_crisis_analysis.py with your actual Twitter API credentials:

API_KEY = "your_api_key"
API_SECRET = "your_api_secret"
ACCESS_TOKEN = "your_access_token"
ACCESS_SECRET = "your_access_secret"

2. Run the Script

python gsoc_crisis_analysis.py

3. Outputs

classified_crisis_tweets.csv - Processed dataset with extracted tweets.

classified_crisis_tweets.json - JSON-formatted dataset.

crisis_heatmap.html - Interactive heatmap of crisis discussions.

sentiment_distribution.png - Sentiment classification plot.

risk_level_distribution.png - Risk level distribution plot.

Features

Automated Social Media Data Extraction - Fetches tweets using keyword-based search.

Text Cleaning & Preprocessing - Removes noise, stopwords, and special characters.

Sentiment Analysis & Risk Classification - Uses NLP to classify posts.

Geolocation Extraction & Mapping - Visualizes crisis discussion trends on a heatmap.

Future Enhancements

Add support for additional social media platforms (e.g., Reddit API).

Improve geolocation accuracy using advanced NLP location recognition.

Implement real-time monitoring and alert systems for high-risk posts.

License

This project is open-source and available under the MIT License.
