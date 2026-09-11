import os
import json
import feedparser
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Add more verified working RSS feed URLs here as you find them.
DEFAULT_FEEDS = {
    "Economic Times": "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms",
}

def fetch_relevant_headlines(keywords, feed_urls=None, max_per_feed=25):
    feed_urls = feed_urls or DEFAULT_FEEDS
    collected = []
    for source_name, url in feed_urls.items():
        feed = feedparser.parse(url)
        for entry in feed.entries[:max_per_feed]:
            title = entry.title
            matched_keywords = [k for k in keywords if k.lower() in title.lower()]
            if matched_keywords:
                collected.append({
                    "headline": title,
                    "source": source_name,
                    "link": getattr(entry, "link", None),
                    "matched_keywords": matched_keywords,
                })
    return collected

def get_news_sentiment(tickers, keywords, feed_urls=None):
    headlines_meta = fetch_relevant_headlines(keywords, feed_urls)

    if not headlines_meta:
        return None

    headlines_text = "\n".join(h["headline"] for h in headlines_meta)

    prompt = f"""Score the sentiment of each of the following headlines on a scale from -1 (very negative) to +1 (very positive) for the stock they mention.

Headlines:
{headlines_text}

Return ONLY a JSON array, no other text, in this exact format:
[{{"headline": "...", "sentiment": 0.0}}, ...]
"""

    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        max_tokens=1000,
        messages=[{"role": "user", "content": prompt}]
    )

    raw_content = response.choices[0].message.content
    cleaned = raw_content.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        cleaned = cleaned.replace("json", "", 1).strip()

    sentiment_data = json.loads(cleaned)
    sentiment_by_headline = {item["headline"]: item["sentiment"] for item in sentiment_data}

    merged = []
    for h in headlines_meta:
        score = sentiment_by_headline.get(h["headline"])
        if score is not None:
            merged.append({**h, "sentiment": score})

    if not merged:
        return None

    avg_sentiment = sum(m["sentiment"] for m in merged) / len(merged)

    per_keyword = {}
    for k in keywords:
        scores = [m["sentiment"] for m in merged if k in m["matched_keywords"]]
        if scores:
            per_keyword[k] = sum(scores) / len(scores)

    positive_count = sum(1 for m in merged if m["sentiment"] > 0.1)
    negative_count = sum(1 for m in merged if m["sentiment"] < -0.1)
    neutral_count = len(merged) - positive_count - negative_count

    return {
        "headlines": merged,
        "average_sentiment": avg_sentiment,
        "per_keyword_sentiment": per_keyword,
        "positive_count": positive_count,
        "negative_count": negative_count,
        "neutral_count": neutral_count,
        "sources_used": list((feed_urls or DEFAULT_FEEDS).keys()),
    }

def compute_adaptive_news_threshold(session_history, fallback_threshold=0.3, z_threshold=1.0):
    past_scores = [
        r["news_result"]["average_sentiment"]
        for r in session_history
        if r.get("news_result")
    ]

    if len(past_scores) < 3:
        return {"threshold": fallback_threshold, "mode": "fallback (not enough history yet)"}

    import numpy as np
    mean_score = np.mean(past_scores)
    std_score = np.std(past_scores)

    if std_score == 0:
        return {"threshold": fallback_threshold, "mode": "fallback (no variation in history)"}

    adaptive_threshold = mean_score + z_threshold * std_score
    return {"threshold": abs(adaptive_threshold), "mode": "adaptive (based on session history)"}