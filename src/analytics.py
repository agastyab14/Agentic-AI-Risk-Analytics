import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats

def compute_advanced_metrics(tickers, weights, period="1y", risk_free_rate=0.06, benchmark="^NSEI"):
    data = yf.download(tickers, period=period)["Close"]

    missing_pct = data.isna().mean()
    bad_tickers = missing_pct[missing_pct > 0.05].index.tolist()
    if bad_tickers:
        data = data.drop(columns=bad_tickers)
    data = data.ffill().dropna()

    returns = data.pct_change().dropna()
    surviving_tickers = list(returns.columns)
    aligned_weights = weights[:len(surviving_tickers)]

    portfolio_returns = (returns * aligned_weights).sum(axis=1)

    # --- Historical VaR (what you already have) ---
    hist_var_95 = -portfolio_returns.quantile(0.05)

    # --- Parametric (Variance-Covariance) VaR — assumes normal distribution ---
    mean_r = portfolio_returns.mean()
    std_r = portfolio_returns.std()
    z_95 = stats.norm.ppf(0.05)  # ~ -1.645
    parametric_var_95 = -(mean_r + z_95 * std_r)

    # --- CVaR / Expected Shortfall — average loss BEYOND the VaR cutoff ---
    var_cutoff = portfolio_returns.quantile(0.05)
    cvar_95 = -portfolio_returns[portfolio_returns <= var_cutoff].mean()

    # --- Annualized volatility ---
    annualized_vol = std_r * np.sqrt(252)

    # --- Skewness & Kurtosis — shape of the return distribution ---
    skewness = portfolio_returns.skew()
    kurtosis = portfolio_returns.kurt()  # excess kurtosis (0 = normal)

    # --- Sortino Ratio — like Sharpe, but only penalizes downside volatility ---
    downside_returns = portfolio_returns[portfolio_returns < 0]
    downside_std = downside_returns.std()
    excess_return = mean_r - (risk_free_rate / 252)
    sortino = (excess_return / downside_std) * np.sqrt(252) if downside_std != 0 else None

    # --- Beta vs Nifty 50 ---
    try:
        bench_data = yf.download(benchmark, period=period)["Close"]
        bench_returns = bench_data.pct_change().dropna()
        aligned = pd.concat([portfolio_returns, bench_returns], axis=1, join="inner")
        aligned.columns = ["portfolio", "benchmark"]
        covariance = aligned["portfolio"].cov(aligned["benchmark"])
        benchmark_variance = aligned["benchmark"].var()
        beta = covariance / benchmark_variance
    except Exception as e:
        beta = None
        aligned = None

    return {
        "surviving_tickers": surviving_tickers,
        "dropped_tickers": bad_tickers,
        "portfolio_returns": portfolio_returns,
        "hist_var_95": hist_var_95,
        "parametric_var_95": parametric_var_95,
        "cvar_95": cvar_95,
        "annualized_vol": annualized_vol,
        "skewness": skewness,
        "kurtosis": kurtosis,
        "sortino": sortino,
        "beta": beta,
        "beta_scatter_data": aligned,
    }

result = compute_advanced_metrics(
    tickers=["CDSL.NS", "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS"],
    weights=[0.25, 0.25, 0.25, 0.25]
)

for k, v in result.items():
    if k not in ["portfolio_returns", "beta_scatter_data"]:
        print(k, ":", v)