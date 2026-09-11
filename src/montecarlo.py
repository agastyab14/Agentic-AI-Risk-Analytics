import numpy as np
import yfinance as yf

def _clean_returns(tickers, period="1y"):
    data = yf.download(tickers, period=period)["Close"]
    missing_pct = data.isna().mean()
    bad_tickers = missing_pct[missing_pct > 0.05].index.tolist()
    if bad_tickers:
        data = data.drop(columns=bad_tickers)
    data = data.ffill().dropna()
    return data.pct_change().dropna(), list(data.columns)

def run_monte_carlo(tickers, weights, n_simulations=5000, n_days=252, initial_value=1_000_000, method="bootstrap"):
    returns_df, surviving_tickers = _clean_returns(tickers)
    aligned_weights = weights[:len(surviving_tickers)]
    portfolio_returns = (returns_df * aligned_weights).sum(axis=1).values

    rng = np.random.default_rng()

    if method == "bootstrap":
        # Resamples ACTUAL historical daily returns with replacement.
        # This preserves the real fat tails and skew in your data, unlike assuming a normal distribution.
        sampled_returns = rng.choice(portfolio_returns, size=(n_simulations, n_days), replace=True)
    else:
        # Parametric: assumes daily returns are normally distributed around the historical mean/std.
        mu = portfolio_returns.mean()
        sigma = portfolio_returns.std()
        sampled_returns = rng.normal(mu, sigma, size=(n_simulations, n_days))

    growth_factors = 1 + sampled_returns
    paths = initial_value * np.cumprod(growth_factors, axis=1)

    terminal_values = paths[:, -1]
    terminal_returns = (terminal_values - initial_value) / initial_value

    var_95_mc = -np.percentile(terminal_returns, 5)
    cutoff = np.percentile(terminal_returns, 5)
    cvar_95_mc = -terminal_returns[terminal_returns <= cutoff].mean()
    prob_loss = float((terminal_values < initial_value).mean())

    # Only keep a subset of paths for plotting — sending 5000 lines to a chart would be unusable and slow.
    sample_idx = rng.choice(n_simulations, size=min(150, n_simulations), replace=False)

    return {
        "method": method,
        "n_simulations": n_simulations,
        "n_days": n_days,
        "initial_value": initial_value,
        "surviving_tickers": surviving_tickers,
        "sample_paths": paths[sample_idx].tolist(),
        "terminal_values": terminal_values.tolist(),
        "var_95_mc": var_95_mc,
        "cvar_95_mc": cvar_95_mc,
        "prob_loss": prob_loss,
        "mean_terminal_value": float(terminal_values.mean()),
        "median_terminal_value": float(np.median(terminal_values)),
    }