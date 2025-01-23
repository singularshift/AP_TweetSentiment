import time
import configparser
import tweepy
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

# 3) Simulated Streaming Function
def simulate_streaming(username, interval=10):
    """
    Simulates real-time streaming by fetching tweets periodically.
    """
    user_data = client.get_user(username=username)
    user_id = user_data.data.id

    seen_tweets = set()  

    while True:
        try:
            # Fetch the latest tweets
            response = client.get_users_tweets(id=user_id, max_results=5, tweet_fields=["created_at"])
            if response.data:
                for tweet in response.data:
                    if tweet.id not in seen_tweets:
                        seen_tweets.add(tweet.id)

                        # Perform sentiment classification
                        start_time = time.time()
                        sentiment = classifier(tweet.text)[0]
                        end_time = time.time()
                        classification_time = end_time - start_time

                        # Output the results
                        print(f"New Tweet: {tweet.text[:50]}...")
                        print(f"Sentiment: {sentiment['label']} (Score: {sentiment['score']:.3f})")
                        print(f"Classification Time: {classification_time:.4f} seconds\n")
            time.sleep(interval)

        except Exception as e:
            print(f"Error: {e}. Retrying in {interval} seconds...")
            time.sleep(interval)

# 4) Main Program
if __name__ == "__main__":
    simulate_streaming("twittername", interval=10)
