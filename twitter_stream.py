import configparser
import tweepy
import pandas as pd
import time
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# 1) Load Configuration
# a) Twitter API credentials
config = configparser.ConfigParser(interpolation=None)
config.read('config.ini')

api_key = config['twitter']['api_key']
api_key_secret = config['twitter']['api_key_secret']
access_token = config['twitter']['access_token']
access_token_secret = config['twitter']['access_token_secret']
bearer_token = config['twitter']['bearer_token']

# b) Load accounts from accounts.ini
accounts_config = configparser.ConfigParser()
accounts_config.read('accounts.ini')
accounts = [user.strip() for user in accounts_config['twitter_accounts']['users'].split(",")]

# 2) Load Sentiment Model
tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")
model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")
classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

# 3) Streaming Client Class
class CryptoStreamListener(tweepy.StreamingClient):
    def __init__(self, bearer_token):
        super().__init__(bearer_token)
        self.tweets_data = []

    def on_tweet(self, tweet):
        """
        Handles incoming tweets, performs sentiment classification, and measures classification time.
        """
        start_time = time.time()

        # Perform sentiment classification
        sentiment = classifier(tweet.text)[0]

        end_time = time.time()
        classification_time = end_time - start_time

        # Print classification result and time
        print(f"New Tweet: {tweet.text[:50]}...")
        print(f"Sentiment: {sentiment['label']} (Score: {sentiment['score']:.3f})")
        print(f"Classification Time: {classification_time:.4f} seconds\n")

        # Save the analyzed tweet
        self.tweets_data.append({
            'Tweet': tweet.text,
            'Sentiment': sentiment['label'],
            'Score': sentiment['score'],
            'Classification_Time_s': classification_time
        })

    def on_errors(self, errors):
        print(f"Error: {errors}")

# 4) Start Streaming
def start_streaming(usernames):
    """
    Starts the streaming client for the specified usernames.
    """
    client = tweepy.Client(bearer_token=bearer_token)
    user_ids = [client.get_user(username=user).data.id for user in usernames]

    # Initialize streaming client
    stream_listener = CryptoStreamListener(bearer_token)

    # Delete existing rules
    rules = stream_listener.get_rules().data
    if rules:
        stream_listener.delete_rules([rule.id for rule in rules])

    # Add new rules
    for user_id in user_ids:
        stream_listener.add_rules(tweepy.StreamRule(f"from:{user_id}"))

    # Start streaming
    print(f"Starting streaming for: {usernames}")
    stream_listener.filter()

# 5) Main Program
if __name__ == "__main__":
    print(f"Streaming Accounts: {accounts}")
    start_streaming(accounts)
