import numpy as np
import yfinance as yf
from scipy.optimize import minimize

UNIVERSE = [
    "CDSL.NS", "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS",
    "ICICIBANK.NS", "SBIN.NS", "ITC.NS", "LT.NS", "HINDUNILVR.NS",
    "BAJFINANCE.NS", "MARUTI.NS", "SUNPHARMA.NS", "TATAMOTORS.NS", "WIPRO.NS"
]

def get_returns(tickers, period="1y"):
    data = yf.download(tickers, period=period)["Close"]

    # Drop any ticker missing more than 5% of trading days — likely a bad symbol or partial listing
    missing_pct = data.isna().mean()
    bad_tickers = missing_pct[missing_pct > 0.05].index.tolist()
    if bad_tickers:
        print(f"Dropping tickers with too much missing data: {bad_tickers}")
        data = data.drop(columns=bad_tickers)

    # Forward-fill small, isolated gaps (e.g. a single missed trading day), then drop any leftover NaN rows
    data = data.ffill().dropna()

    return data.pct_change().dropna()

def compute_min_variance_weights(returns_df):
    cov_matrix = returns_df.cov() * 10000
    n = len(cov_matrix)

    def portfolio_variance(weights):
        return weights.T @ cov_matrix.values @ weights

    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
    bounds = tuple((0, 1) for _ in range(n))
    init_guess = np.array([1 / n] * n)

    result = minimize(
        portfolio_variance, init_guess, method="SLSQP",
        bounds=bounds, constraints=constraints,
        options={'maxiter': 1000, 'ftol': 1e-12}
    )

    print("Optimizer success:", result.success)
    print("Optimizer message:", result.message)

    weights_dict = dict(zip(cov_matrix.columns, result.x))
    return weights_dict