# Sentinel — Multi-Agent Portfolio Risk Early-Warning System

Sentinel is an agentic pipeline that monitors macroeconomic surprises and news sentiment for a portfolio, decides how much further analysis is warranted, and — when conditions justify it — generates a full quantitative risk report, a Monte Carlo simulation, ML-based risk forecasting, and an options-based income strategy.

Built for [Hackathon Name] — Agentic AI Hackathon.

## Why This Is Agentic, Not Just a Script

- **Goal-driven execution**: the system's goal is portfolio risk early-warning, not a single fixed task
- **Dynamic action selection**: a severity-scoring layer decides whether to do nothing, run a lightweight check, or trigger a full risk report — the pipeline does different amounts of work depending on what it finds
- **Multi-step execution**: macro check → news check → severity assessment → conditional risk recompute → memo → full report, orchestrated as a LangGraph state machine
- **Adaptation**: the news-sentiment threshold adjusts based on the system's own recent run history rather than staying fixed
- **Robustness**: tickers with insufficient price data are automatically detected and excluded, with a visible warning, instead of crashing or silently corrupting results

## Architecture

```text
                         ┌───────────────┐
                         │    FRED API   │
                         └───────┬───────┘
                                 │
                                 ▼
                         ┌───────────────┐
                         │  Macro Agent  │
                         └───────┬───────┘
                                 │
                                 │
┌───────────────┐                ▼
│   RSS Feeds   │────────► ┌───────────────┐
└───────────────┘          │   News Agent  │
                           └───────┬───────┘
                                   │
                                   ▼
                           ┌───────────────────┐
                           │ Severity          │
                           │ Assessment        │
                           └─────────┬─────────┘
                                     │
                              ┌──────┴──────┐
                              │ Conditional │
                              │   Trigger   │
                              └──────┬──────┘
                                     │
                            ┌────────▼────────┐
                            │    Risk Agent   │
                            │ VaR / Sharpe /  │
                            │ Drawdown / Corr │
                            └────────┬────────┘
                                     │
                                     ▼
                            ┌─────────────────┐
                            │    Memo Agent   │
                            └────────┬────────┘
                                     │
                                     ▼
                            ┌─────────────────┐
                            │   Report Agent  │
                            └─────────────────┘

             ┌─────────────────────────────────────┐
             │       Adaptive Threshold            │
             │ Learns from recent session history  │
             └─────────────────────────────────────┘

                         yfinance
                            │
                            └──────────► Risk Agent
```

Additional standalone modules, explorable independently of the main pipeline:
- **Deep Risk Analytics**: Historical/Parametric VaR, CVaR, Sortino Ratio, Beta, skew/kurtosis
- **Monte Carlo Simulation**: bootstrap-resampled portfolio value paths, simulated VaR/CVaR
- **ML Risk Forecasting**: Decision Tree and Logistic Regression predicting large-loss-day events
- **Covered Call Income Strategy**: Black-Scholes-based options income overlay with payoff diagrams

## Tech Stack

- **Orchestration**: LangGraph
- **LLM**: Groq API (`openai/gpt-oss-120b`)
- **Data**: `yfinance` (equities), FRED API (macro), RSS via `feedparser` (news)
- **ML**: scikit-learn (Decision Tree, Logistic Regression)
- **Frontend**: Streamlit + Plotly
- **Optimization**: SciPy (minimum-variance portfolio weights)

## Setup

1. Clone this repo:
```bash
   git clone https://github.com/YOUR_USERNAME/sentinel-portfolio-risk.git
   cd sentinel-portfolio-risk


2. Create and activate a virtual environment:
```bash
   python -m venv venv
   venv\Scripts\activate      # Windows
   source venv/bin/activate   # Mac/Linux
```

3. Install dependencies:
```bash
   pip install -r requirements.txt
```

4. Create a `.env` file in the project root with your own API keys:
FRED_API_KEY=your_fred_key
GROQ_API_KEY=your_groq_key
   - Get a free FRED key at https://fred.stlouisfed.org/docs/api/api_key.html
   - Get a free Groq key at https://console.groq.com

5. Run the app:
```bash
   streamlit run app.py
```


## Project Structure

```text
sentinel/
│
├── app.py                         # Streamlit UI + LangGraph pipeline wiring
│
├── src/
│   ├── __init__.py
│   ├── risk.py                    # Core VaR, Sharpe, Drawdown, Correlation engine
│   ├── macro.py                   # FRED-based macro surprise detection (z-score)
│   ├── news.py                    # RSS + LLM sentiment scoring + adaptive threshold
│   ├── severity.py                # Severity tiering (Low / Medium / High)
│   ├── report.py                  # Structured LLM-generated risk report
│   ├── portfolio.py               # Ticker universe + minimum-variance optimizer
│   ├── analytics.py               # CVaR, Sortino, Beta, skewness, kurtosis
│   ├── montecarlo.py              # Bootstrap Monte Carlo simulation
│   ├── forecast_model.py          # Decision Tree / Logistic Regression forecasting
│   └── options_strategy.py        # Black-Scholes covered call income overlay
│
├── notebooks/
│   ├── NSE_Analyser.ipynb
│   ├── langgraph_test.ipynb
│   ├── macrodashboard.ipynb
│   ├── newsagent.ipynb
│   ├── sentinel_graph.ipynb
│   └── test_imports.ipynb
│
├── requirements.txt               # Python dependencies
├── .env.example                   # Environment variable template
├── .gitignore                     # Git exclusions
└── README.md                      # Project documentation
```

## Honest Engineering Notes (Design Decisions & Limitations)

- **Scheduling**: the original design called for n8n for autonomous scheduling. Docker Desktop on the development machine could not enable required virtualization support, and Node.js was unavailable as a fallback. A lightweight Python `schedule`-based loop achieves the same unattended, periodic-run behavior and is easily swappable for n8n or a cron job in a production environment.
- **Options pricing**: the covered call strategy uses each stock's *historical realized volatility* as a stand-in for *implied volatility*, since live options-market data wasn't available. Real options desks price off implied volatility, which can diverge from historical volatility — this is a reasonable approximation for a hackathon build, not a production-grade options pricing system.
- **ML forecasting**: this predicts "is tomorrow a large-loss day for this portfolio" (a data-availability-driven proxy), not literal equipment failure or credit default — the same modeling technique (a chronologically-split binary event classifier) applies, mapped to the data actually available.
- **Adaptive threshold**: the news-sentiment threshold adapts based on session history, which means it needs at least 3 runs before adaptation kicks in, and can become very sensitive with very little historical variance. A production version would use a longer rolling window with a sensible floor value.

## Team / Submission

Agastya Banerjee
