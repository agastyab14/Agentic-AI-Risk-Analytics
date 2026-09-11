def compute_severity(macro_result, news_result, news_threshold):
    macro_score = abs(macro_result["z_score"]) / macro_result["threshold"] if macro_result else 0

    if news_result and news_threshold["threshold"] > 0:
        news_score = abs(news_result["average_sentiment"]) / news_threshold["threshold"]
    else:
        news_score = 0

    combined_score = max(macro_score, news_score)

    if combined_score >= 2.0:
        tier = "High"
    elif combined_score >= 1.0:
        tier = "Medium"
    else:
        tier = "Low"

    return {
        "macro_score": macro_score,
        "news_score": news_score,
        "combined_score": combined_score,
        "tier": tier
    }   