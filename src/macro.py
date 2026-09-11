import os
from dotenv import load_dotenv
from fredapi import Fred

load_dotenv()
fred = Fred(api_key=os.getenv("FRED_API_KEY"))

def check_macro_surprise(series_id="INDCPIALLMINMEI", z_threshold=1.5):
    series = fred.get_series(series_id).dropna()
    pct_changes = series.pct_change().dropna()

    latest_pct_change = pct_changes.iloc[-1]
    historical_changes = pct_changes.iloc[:-1]  # exclude latest from its own baseline

    mean_change = historical_changes.mean()
    std_change = historical_changes.std()
    z_score = (latest_pct_change - mean_change) / std_change

    return {
        "series_id": series_id,
        "latest_value": series.iloc[-1],
        "prior_value": series.iloc[-2],
        "pct_change": latest_pct_change,
        "z_score": z_score,
        "threshold": z_threshold,
        "surprise": bool(abs(z_score) > z_threshold)
    }