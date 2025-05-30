# sentiment/finbert_sentiment.py
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline

def load_finbert_pipeline():
    model = AutoModelForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone")
    tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")
    nlp = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
    return nlp

def analyze_sentiment(texts):
    finbert = load_finbert_pipeline()
    return finbert(texts)
