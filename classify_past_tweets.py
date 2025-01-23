import time
import configparser
import tweepy
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# 1) Load Configuration
config = configparser.ConfigParser(interpolation=None)
config.read('config.ini')

api_key = config['twitter']['api_key']
api_key_secret = config['twitter']['api_key_secret']
access_token = config['twitter']['access_token']
access_token_secret = config['twitter']['access_token_secret']
bearer_token = config['twitter']['bearer_token']

# Tweepy Client
client = tweepy.Client(
    bearer_token=bearer_token,
    consumer_key=api_key,
    consumer_secret=api_key_secret,
    access_token=access_token,
    access_token_secret=access_token_secret
)

# 2) Load Sentiment Model
tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")
model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")
classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

# 3) Classify Past Tweets
def classify_past_tweets(username, limit=3):
    # Classifies the last `limit` tweets of a user and measures classification time.
    # Get user ID
    user_data = client.get_user(username=username)
    user_id = user_data.data.id

    # Fetch tweets
    response = client.get_users_tweets(id=user_id, max_results=limit)

    # Classify tweets
    tweets = []
    if response.data:
        for tweet in response.data:
            start_time = time.time()

            # Sentiment classification
            sentiment = classifier(tweet.text)[0]

            end_time = time.time()
            classification_time = end_time - start_time

            tweets.append({
                'Tweet': tweet.text,
                'Sentiment': sentiment['label'],
                'Score': sentiment['score'],
                'Classification_Time_s': classification_time
            })
            print(f"Tweet: {tweet.text[:50]}...")
            print(f"Sentiment: {sentiment['label']} (Score: {sentiment['score']:.3f})")
            print(f"Classification Time: {classification_time:.4f} seconds\n")
    
    return pd.DataFrame(tweets)

# 4) Main Program
if __name__ == "__main__":
    username = "twitter_username" 
    classified_tweets = classify_past_tweets(username, limit=3)
    print("\n--- Classified Past Tweets ---")
    print(classified_tweets)
