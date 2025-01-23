# AlphaParty_BTC_Sentiment_V1

These files track sentiment using Twitter data. The pipeline analyzes tweets in real time or from past data and uses the fine-tuned **RoBERTa sentiment analysis model** (`cardiffnlp/twitter-roberta-base-sentiment-latest`) to classify tweets as **positive**, **neutral**, or **negative**.

---

## Features

- **Real-Time Sentiment Analysis (Streaming)**: Track sentiment from live tweets from specific accounts (requires Pro Tier for streaming).
- **Simulated Streaming**: Fetch recent tweets periodically to mimic real-time analysis
- **Past Tweet Classification**: Analyze the sentiment of recent tweets from a user
- **Mock Data Testing**: Test the sentiment classification pipeline with mock data

---

- # Setup

- You need to setup **accounts.ini** and **config.ini** with account usernames you want to track, as well as your X API credentials
- The username you want to track in simulation needs to be added at the bottom of the script as well
