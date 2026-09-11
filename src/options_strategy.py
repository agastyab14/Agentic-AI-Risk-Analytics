import numpy as np
import yfinance as yf
from scipy.stats import norm

def _get_spot_and_vol(tickers, period="1y"):
    data = yf.download(tickers, period=period)["Close"]
    missing_pct = data.isna().mean()
    bad_tickers = missing_pct[missing_pct > 0.05].index.tolist()
    if bad_tickers:
        data = data.drop(columns=bad_tickers)
    data = data.ffill().dropna()

    returns = data.pct_change().dropna()
    annualized_vol = returns.std() * np.sqrt(252)
    spot_prices = data.iloc[-1]
    return spot_prices.to_dict(), annualized_vol.to_dict(), list(data.columns)

def black_scholes_call(S, K, T, r, sigma):
    if sigma <= 0 or T <= 0:
        return 0.0, 0.0, 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    prob_itm = norm.cdf(d2)  # risk-neutral probability of finishing ITM — used as an assignment-risk proxy
    delta = norm.cdf(d1)
    return float(call_price), float(prob_itm), float(delta)

def compute_covered_call_overlay(tickers, weights, portfolio_value, otm_pct=0.05, days_to_expiry=30, risk_free_rate=0.06):
    spot_prices, annualized_vols, surviving_tickers = _get_spot_and_vol(tickers)
    aligned_weights = weights[:len(surviving_tickers)]
    T = days_to_expiry / 365

    breakdown = []
    total_premium_income = 0.0

    for ticker, weight in zip(surviving_tickers, aligned_weights):
        S = spot_prices[ticker]
        sigma = annualized_vols[ticker]
        K = S * (1 + otm_pct)

        call_price, prob_itm, delta = black_scholes_call(S, K, T, risk_free_rate, sigma)

        position_value = weight * portfolio_value
        shares_held = position_value / S
        premium_income = call_price * shares_held

        breakdown.append({
            "ticker": ticker, "spot_price": S, "strike_price": K, "annualized_vol": sigma,
            "call_price_per_share": call_price, "shares_held": shares_held,
            "premium_income": premium_income, "probability_of_assignment": prob_itm,
            "delta": delta, "position_value": position_value,
        })
        total_premium_income += premium_income

    total_yield_pct = total_premium_income / portfolio_value
    annualized_yield_pct = total_yield_pct * (365 / days_to_expiry)

    return {
        "breakdown": breakdown, "total_premium_income": total_premium_income,
        "total_yield_pct": total_yield_pct, "annualized_yield_pct": annualized_yield_pct,
        "otm_pct": otm_pct, "days_to_expiry": days_to_expiry, "portfolio_value": portfolio_value,
    }

def compute_payoff_curve(S, K, premium, price_range_pct=0.3, n_points=100):
    prices = np.linspace(S * (1 - price_range_pct), S * (1 + price_range_pct), n_points)
    stock_only_payoff = prices - S
    covered_call_payoff = np.where(prices <= K, prices - S + premium, (K - S) + premium)
    return prices.tolist(), stock_only_payoff.tolist(), covered_call_payoff.tolist()