def generate_report(macro_result, news_result, severity, risk_result, memo, client):
    correlation_summary = risk_result.get("correlation_matrix", {})

    prompt = f"""You are a senior portfolio risk analyst writing an internal report for a portfolio manager. Write a clear, well-structured report in Markdown with the following exact sections:

## What Happened
Explain in plain language what triggered this report — cite the specific macro and/or news signals.

## Why It Matters
Explain the risk implications using the actual numbers below. Reference VaR, Sharpe ratio, max drawdown, and correlation specifically, and explain what each number means for someone who understands finance but wants it explained clearly.

## Recommended Strategy
Give ONE clear, specific recommended action (e.g., reduce exposure to a specific holding by X%, add a hedge, diversify into an uncorrelated asset, or hold). Be concrete, not vague.

## Why This Recommendation
Justify the recommendation using portfolio theory concepts — diversification, correlation, risk-adjusted return (Sharpe), or drawdown risk. Explain the reasoning, not just the conclusion.

## What to Watch Next
Give 2-3 specific things to monitor going forward (e.g., a specific data release, a specific news development, a specific metric threshold).

---

DATA:

Severity: {severity['tier']} (combined score: {severity['combined_score']:.2f}, macro contribution: {severity['macro_score']:.2f}x, news contribution: {severity['news_score']:.2f}x)

Macro signal: {macro_result}

News signal: {news_result}

Risk metrics:
- VaR (95%): {risk_result['var_95']*100:.2f}%
- Sharpe Ratio: {risk_result['sharpe']:.2f}
- Max Drawdown: {risk_result['max_drawdown']*100:.2f}%
- Correlation matrix: {correlation_summary}

Short memo already generated: {memo}

Write the full report now, following the exact section structure above.
"""

    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        max_tokens=1200,
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content