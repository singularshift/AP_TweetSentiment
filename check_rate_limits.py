import time
import configparser
import tweepy

config = configparser.ConfigParser(interpolation=None)
config.read('config.ini')

api_key = config['twitter']['api_key']
api_key_secret = config['twitter']['api_key_secret']
access_token = config['twitter']['access_token']
access_token_secret = config['twitter']['access_token_secret']
bearer_token = config['twitter']['bearer_token']

client = tweepy.Client(
    bearer_token=bearer_token,
    consumer_key=api_key,
    consumer_secret=api_key_secret,
    access_token=access_token,
    access_token_secret=access_token_secret
)

def check_rate_limits():
    try:
        response = client.get_users_tweets(id=1, max_results=5) 
        remaining = int(response.headers["x-rate-limit-remaining"])
        reset_time = int(response.headers["x-rate-limit-reset"])
        print(f"Remaining Requests: {remaining}")
        print(f"Limit will be reset at: {time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(reset_time))}")
    except tweepy.errors.TooManyRequests as e:
        reset_time = int(e.response.headers["x-rate-limit-reset"])
        print(f"Rate limit reached. Waiting time for Reset: {reset_time - int(time.time())} Seconds.")

if __name__ == "__main__":
    check_rate_limits()
