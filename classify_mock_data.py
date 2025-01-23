import time
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# 1) Load Sentiment Model From HuggingFace (https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest)
tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")
model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")
classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

# 2) Mock Data Classification
def classify_mock_data():
    """
    Classifies mock tweets and measures classification time for each.
    """
    mock_tweets = [
        "Iran is attacking Israel",
        "China wants to ban Bitcoin",
        "Stablecoins remain stable despite the volatility."
    ]

    for tweet in mock_tweets:
        start_time = time.time()

        # Sentiment classification
        sentiment = classifier(tweet)[0]

        end_time = time.time()
        classification_time = end_time - start_time

        # Classification result and time
        print(f"Mock Tweet: {tweet}")
        print(f"Sentiment: {sentiment['label']} (Score: {sentiment['score']:.3f})")
        print(f"Classification Time: {classification_time:.4f} seconds\n")

# 3) Main Program
if __name__ == "__main__":
    classify_mock_data()
