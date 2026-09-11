import yfinance as yf
import numpy as np

def compute_risk_snapshot(tickers, weights, period="1y", risk_free_rate=0.06):
    data = yf.download(tickers, period=period)["Close"]

    missing_pct = data.isna().mean()
    bad_tickers = missing_pct[missing_pct > 0.05].index.tolist()
    if bad_tickers:
        print(f"Dropping tickers with too much missing data: {bad_tickers}")
        data = data.drop(columns=bad_tickers)

    data = data.ffill().dropna()
    returns = data.pct_change().dropna()

    portfolio_returns = (returns * weights[:len(returns.columns)]).sum(axis=1)

    var_95 = -portfolio_returns.quantile(0.05)

    excess_daily_return = portfolio_returns.mean() - (risk_free_rate / 252)
    sharpe = (excess_daily_return / portfolio_returns.std()) * np.sqrt(252)

    cumulative = (1 + portfolio_returns).cumprod()
    running_max = cumulative.cummax()
    drawdown_series = (cumulative - running_max) / running_max
    max_drawdown = drawdown_series.min()

    correlation_matrix = returns.corr()

    return {
        "var_95": var_95,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "correlation_matrix": correlation_matrix.to_dict(),
        "dates": [d.strftime("%Y-%m-%d") for d in portfolio_returns.index],
        "portfolio_returns": portfolio_returns.tolist(),
        "cumulative": cumulative.tolist(),
        "drawdown_series": drawdown_series.tolist(),
    }