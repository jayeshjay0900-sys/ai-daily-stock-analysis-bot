from __future__ import annotations

from collections import Counter
from typing import Iterable

from transformers import pipeline


_MODEL = None


def get_sentiment_pipeline():
    global _MODEL
    if _MODEL is None:
        _MODEL = pipeline(
            "sentiment-analysis",
            model="ProsusAI/finbert",
        )
    return _MODEL


def analyze_news_sentiment(articles: Iterable[dict]) -> list[dict]:
    articles = list(articles)
    if not articles:
        return []

    model = get_sentiment_pipeline()
    texts = [a.get("headline", "")[:1000] for a in articles]
    results = model(texts, truncation=True)

    enriched = []
    for article, result in zip(articles, results):
        item = dict(article)
        item["sentiment"] = str(result["label"]).upper()
        item["confidence"] = float(result["score"])
        enriched.append(item)

    return enriched


def sentiment_summary(articles: list[dict]) -> dict:
    counts = Counter(a.get("sentiment", "UNKNOWN") for a in articles)
    total = len(articles)

    def pct(label):
        return (counts.get(label, 0) / total * 100) if total else 0.0

    return {
        "positive": pct("POSITIVE"),
        "negative": pct("NEGATIVE"),
        "neutral": pct("NEUTRAL"),
        "total": total,
    }
