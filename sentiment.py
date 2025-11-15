from nltk.sentiment.vader import SentimentIntensityAnalyzer

sia = SentimentIntensityAnalyzer()

def analyze_sentiment(text):
    score = sia.polarity_scores(text)
    if score["compound"] > 0.05:
        label = "Positive"
    elif score["compound"] < -0.05:
        label = "Negative"
    else:
        label = "Neutral"
    return {"label": label, "score": round(score["compound"], 3)}
