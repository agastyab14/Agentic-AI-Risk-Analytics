import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from typing import TypedDict, Optional, Dict, Any
from langgraph.graph import StateGraph, END

from src.risk import compute_risk_snapshot
from src.macro import check_macro_surprise
from src.news import get_news_sentiment, compute_adaptive_news_threshold, client
from src.portfolio import UNIVERSE, get_returns, compute_min_variance_weights
from src.severity import compute_severity
from src.report import generate_report
from src.analytics import compute_advanced_metrics
from src.montecarlo import run_monte_carlo
from src.forecast_model import train_risk_event_models
from src.options_strategy import compute_covered_call_overlay, compute_payoff_curve

st.set_page_config(page_title="Sentinel", layout="wide")
st.title("🛡️ Sentinel — Portfolio Risk Early-Warning System")

TIER_COLORS = {"Low": "🟢", "Medium": "🟡", "High": "🔴"}

# --- LangGraph state ---

class PipelineState(TypedDict):
    tickers: list
    weights: list
    keywords: list
    force_full_run: bool
    news_threshold: Optional[Dict[str, Any]]
    macro_result: Optional[Dict[str, Any]]
    news_result: Optional[Dict[str, Any]]
    severity: Optional[Dict[str, Any]]
    risk_result: Optional[Dict[str, Any]]
    memo: Optional[str]
    report: Optional[str]

# --- Node functions ---

def macro_check(state: PipelineState) -> PipelineState:
    return {"macro_result": check_macro_surprise()}

def news_check(state: PipelineState) -> PipelineState:
    return {"news_result": get_news_sentiment(tickers=state["tickers"], keywords=state["keywords"])}

def assess_severity(state: PipelineState) -> PipelineState:
    severity = compute_severity(state["macro_result"], state["news_result"], state["news_threshold"])
    return {"severity": severity}

def risk_check(state: PipelineState) -> PipelineState:
    return {"risk_result": compute_risk_snapshot(tickers=state["tickers"], weights=state["weights"])}

def route_after_severity(state: PipelineState) -> str:
    tier = state["severity"]["tier"]
    if state.get("force_full_run") or tier in ("Medium", "High"):
        return "risk_check"
    return END

def synthesize_memo(state: PipelineState) -> PipelineState:
    risk_summary = {k: v for k, v in state["risk_result"].items() if k in ["var_95", "sharpe", "max_drawdown"]}
    tier = state["severity"]["tier"]

    urgency_instruction = {
        "Low": "This is a mild, precautionary check. Keep the tone calm and low-urgency.",
        "Medium": "This is a moderate concern. Recommend a specific, proportionate risk-reducing action.",
        "High": "This is a significant, urgent concern. Recommend immediate, specific risk mitigation."
    }[tier]

    prompt = f"""You are a portfolio risk analyst. Write a short risk memo (3-4 sentences) based on the following data.

Severity tier: {tier} (combined score: {state["severity"]["combined_score"]:.2f})
{urgency_instruction}

Macro surprise: {state["macro_result"]}
News sentiment: {state["news_result"]}
Risk metrics: {risk_summary}

Explain what changed, the portfolio's current exposure, and a suggested SPECIFIC action (e.g. reduce exposure by X%, hedge with Y, or hold), matching the urgency of the {tier} severity tier.
"""
    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        max_tokens=300,
        messages=[{"role": "user", "content": prompt}]
    )
    return {"memo": response.choices[0].message.content}

def generate_full_report(state: PipelineState) -> PipelineState:
    report_text = generate_report(
        macro_result=state["macro_result"],
        news_result=state["news_result"],
        severity=state["severity"],
        risk_result=state["risk_result"],
        memo=state["memo"],
        client=client
    )
    return {"report": report_text}

# --- Build the graph ---

graph = StateGraph(PipelineState)
graph.add_node("macro_check", macro_check)
graph.add_node("news_check", news_check)
graph.add_node("assess_severity", assess_severity)
graph.add_node("risk_check", risk_check)
graph.add_node("synthesize_memo", synthesize_memo)
graph.add_node("generate_full_report", generate_full_report)
graph.set_entry_point("macro_check")
graph.add_edge("macro_check", "news_check")
graph.add_edge("news_check", "assess_severity")
graph.add_conditional_edges("assess_severity", route_after_severity)
graph.add_edge("risk_check", "synthesize_memo")
graph.add_edge("synthesize_memo", "generate_full_report")
graph.add_edge("generate_full_report", END)
app_graph = graph.compile()

# --- Session state ---
if "history" not in st.session_state:
    st.session_state.history = []

# --- Sidebar ---
with st.sidebar:
    st.header("Portfolio Setup")

    selected_tickers = st.multiselect(
        "Select tickers",
        UNIVERSE,
        default=["CDSL.NS", "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS"]
    )

    weight_mode = st.radio("Weight Assignment", ["Auto (Minimum Variance)", "Custom"])

    custom_weights_input = None
    if weight_mode == "Custom":
        custom_weights_input = st.text_input(
            "Weights (comma-separated, must match ticker count and sum to 1)",
            ""
        )

    keywords_input = st.text_input("News keywords (comma-separated)", "CDSL, Reliance, RIL, TCS, HDFC Bank, HDFC")
    force_full_run = st.checkbox("Force full analysis (demo mode)", value=False)
    run_clicked = st.button("Run Check Now", type="primary")

    st.divider()
    st.header("Deep Analytics")
    analyze_clicked = st.button("Run Deep Risk Analytics")

    st.divider()
    st.header("Monte Carlo Simulation")
    mc_method = st.selectbox("Method", ["bootstrap", "parametric"])
    mc_n_sims = st.number_input("Number of simulations", min_value=500, max_value=20000, value=5000, step=500)
    mc_n_days = st.number_input("Time horizon (trading days)", min_value=30, max_value=756, value=252, step=30)
    mc_initial = st.number_input("Initial portfolio value (₹)", min_value=10000, value=1000000, step=10000)
    mc_clicked = st.button("Run Monte Carlo Simulation")

    st.divider()
    st.header("ML Risk Forecast")
    ml_percentile = st.slider("Define 'large loss day' as bottom X percentile", 1, 25, 10)
    ml_clicked = st.button("Train Risk Event Models")

    st.divider()
    st.header("Covered Call Income Strategy")
    cc_otm_pct = st.slider("Strike % above current price (OTM)", 1, 20, 5) / 100
    cc_days = st.number_input("Days to expiry", min_value=7, max_value=90, value=30, step=1)
    cc_portfolio_value = st.number_input("Notional Portfolio Value (₹)", min_value=10000, value=1000000, step=10000)
    cc_clicked = st.button("Run Covered Call Analysis")

def _resolve_tickers_weights():
    tickers = selected_tickers
    if weight_mode == "Custom" and custom_weights_input.strip():
        weights = [float(w.strip()) for w in custom_weights_input.split(",")]
    else:
        returns_df = get_returns(tickers)
        weights_dict = compute_min_variance_weights(returns_df)
        tickers = list(returns_df.columns)
        weights = [weights_dict[t] for t in tickers]
    return tickers, weights

# --- Run main pipeline ---
if run_clicked:
    if not selected_tickers:
        st.error("Select at least one ticker.")
        st.stop()

    tickers = selected_tickers
    keywords = [k.strip() for k in keywords_input.split(",")]

    if weight_mode == "Custom" and custom_weights_input.strip():
        weights = [float(w.strip()) for w in custom_weights_input.split(",")]
        if len(weights) != len(tickers):
            st.error(f"You selected {len(tickers)} tickers but gave {len(weights)} weights.")
            st.stop()
        if abs(sum(weights) - 1.0) > 0.01:
            st.error(f"Weights sum to {sum(weights):.2f}, must sum to 1.")
            st.stop()
        weights_source = "custom"
    else:
        with st.spinner("No custom weights given — computing minimum-variance portfolio..."):
            returns_df = get_returns(tickers)
            surviving_tickers = list(returns_df.columns)
            dropped = [t for t in tickers if t not in surviving_tickers]
            if dropped:
                st.warning(f"Dropped due to insufficient data: {', '.join(dropped)}")
            weights_dict = compute_min_variance_weights(returns_df)
            tickers = surviving_tickers
            weights = [weights_dict[t] for t in tickers]
        weights_source = "minimum-variance (auto)"

    st.sidebar.subheader("Weights used")
    for t, w in zip(tickers, weights):
        st.sidebar.write(f"{t}: {w:.3f}")
    st.sidebar.caption(f"Source: {weights_source}")

    news_threshold = compute_adaptive_news_threshold(st.session_state.history)

    with st.spinner("Running Sentinel pipeline..."):
        result = app_graph.invoke({
            "tickers": tickers,
            "weights": weights,
            "keywords": keywords,
            "force_full_run": force_full_run,
            "news_threshold": news_threshold,
            "macro_result": None,
            "news_result": None,
            "severity": None,
            "risk_result": None,
            "memo": None,
            "report": None
        })

    result["_timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    result["_tickers"] = tickers
    result["_weights"] = weights
    st.session_state.history.append(result)
    st.session_state.latest_result = result

# --- Run deep analytics ---
if analyze_clicked:
    if not selected_tickers:
        st.error("Select at least one ticker.")
        st.stop()
    with st.spinner("Running deep risk analytics..."):
        tickers, weights = _resolve_tickers_weights()
        analytics = compute_advanced_metrics(tickers=tickers, weights=weights)
    st.session_state.latest_analytics = analytics

# --- Run Monte Carlo ---
if mc_clicked:
    if not selected_tickers:
        st.error("Select at least one ticker.")
        st.stop()
    with st.spinner(f"Running {mc_n_sims} Monte Carlo simulations..."):
        tickers, weights = _resolve_tickers_weights()
        mc_result = run_monte_carlo(
            tickers=tickers, weights=weights,
            n_simulations=mc_n_sims, n_days=mc_n_days,
            initial_value=mc_initial, method=mc_method
        )
    st.session_state.latest_mc = mc_result

# --- Run ML forecast ---
if ml_clicked:
    if not selected_tickers:
        st.error("Select at least one ticker.")
        st.stop()
    with st.spinner("Training decision tree and logistic regression models..."):
        tickers, weights = _resolve_tickers_weights()
        ml_result = train_risk_event_models(tickers=tickers, weights=weights, loss_percentile=ml_percentile)
    st.session_state.latest_ml = ml_result

# --- Run Covered Call Analysis ---
if cc_clicked:
    if not selected_tickers:
        st.error("Select at least one ticker.")
        st.stop()
    with st.spinner("Pricing covered calls (Black-Scholes)..."):
        tickers, weights = _resolve_tickers_weights()
        cc_result = compute_covered_call_overlay(
            tickers=tickers, weights=weights, portfolio_value=cc_portfolio_value,
            otm_pct=cc_otm_pct, days_to_expiry=cc_days
        )
    st.session_state.latest_cc = cc_result

# --- Tabs ---
(tab_dashboard, tab_news, tab_analytics, tab_montecarlo, tab_ml, tab_income,
 tab_report, tab_methodology, tab_trace, tab_history) = st.tabs(
    ["📊 Dashboard", "📰 News Sentiment", "🧮 Risk Analytics", "🎲 Monte Carlo",
     "🤖 ML Forecast", "💰 Income Strategy", "📄 Report", "📚 Methodology", "🔍 Agent Trace", "🕒 History"]
)

if "latest_result" in st.session_state:
    result = st.session_state.latest_result
    severity = result.get("severity")

    with tab_dashboard:
        if severity:
            tier = severity["tier"]
            st.subheader(f"{TIER_COLORS[tier]} Severity: {tier}  (score: {severity['combined_score']:.2f})")

        risk = result.get("risk_result")
        if risk:
            col1, col2, col3 = st.columns(3)
            col1.metric("VaR (95%)", f"{risk['var_95']*100:.2f}%")
            col2.metric("Sharpe Ratio", f"{risk['sharpe']:.2f}")
            col3.metric("Max Drawdown", f"{risk['max_drawdown']*100:.2f}%")

            st.subheader("Correlation Heatmap")
            corr_df = pd.DataFrame(risk["correlation_matrix"])
            fig_corr = px.imshow(corr_df, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1, aspect="auto")
            st.plotly_chart(fig_corr, use_container_width=True)

            st.subheader("Portfolio Value & Drawdown (1Y)")
            fig_dd = go.Figure()
            fig_dd.add_trace(go.Scatter(x=risk["dates"], y=risk["cumulative"], name="Portfolio Value", line=dict(color="cyan")))
            fig_dd.add_trace(go.Scatter(x=risk["dates"], y=risk["drawdown_series"], name="Drawdown", yaxis="y2", line=dict(color="red")))
            fig_dd.update_layout(
                yaxis=dict(title="Cumulative Value"),
                yaxis2=dict(title="Drawdown", overlaying="y", side="right", tickformat=".0%"),
                legend=dict(orientation="h")
            )
            st.plotly_chart(fig_dd, use_container_width=True)

            st.subheader("Daily Returns Distribution (VaR check)")
            fig_hist = px.histogram(x=risk["portfolio_returns"], nbins=40, title="Daily Portfolio Returns")
            fig_hist.add_vline(x=-risk["var_95"], line_dash="dash", line_color="red", annotation_text="VaR 95%")
            st.plotly_chart(fig_hist, use_container_width=True)
        else:
            st.info("Risk check was skipped this run — severity was Low, nothing warranted a full recompute.")

        if result.get("memo"):
            st.subheader("📝 Risk Memo")
            st.write(result["memo"])

    with tab_news:
        news = result.get("news_result")
        if news:
            st.caption(f"Sources: {', '.join(news['sources_used'])}")

            colA, colB, colC = st.columns(3)
            colA.metric("Average Sentiment", f"{news['average_sentiment']:.2f}")
            colB.metric("Headlines Analyzed", len(news["headlines"]))
            colC.metric("Positive / Negative / Neutral", f"{news['positive_count']} / {news['negative_count']} / {news['neutral_count']}")

            col1, col2 = st.columns([1, 1])
            with col1:
                fig_donut = px.pie(
                    names=["Positive", "Negative", "Neutral"],
                    values=[news["positive_count"], news["negative_count"], news["neutral_count"]],
                    hole=0.5,
                    color_discrete_sequence=["#2ECC71", "#E74C3C", "#95A5A6"],
                    title="Sentiment Breakdown"
                )
                st.plotly_chart(fig_donut, use_container_width=True)

            with col2:
                if news["per_keyword_sentiment"]:
                    kw_df = pd.DataFrame(
                        list(news["per_keyword_sentiment"].items()), columns=["Ticker Keyword", "Avg Sentiment"]
                    ).sort_values("Avg Sentiment")
                    fig_kw = px.bar(
                        kw_df, x="Avg Sentiment", y="Ticker Keyword", orientation="h",
                        color="Avg Sentiment", color_continuous_scale="RdYlGn", range_color=[-1, 1],
                        title="Sentiment by Ticker"
                    )
                    st.plotly_chart(fig_kw, use_container_width=True)

            df_news = pd.DataFrame(news["headlines"]).sort_values("sentiment")
            fig_bar = px.bar(
                df_news, x="sentiment", y="headline", orientation="h",
                color="sentiment", color_continuous_scale="RdYlGn", range_color=[-1, 1],
                title="All Headlines by Sentiment"
            )
            fig_bar.update_layout(yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig_bar, use_container_width=True)

            st.markdown("#### Headline Detail")
            try:
                st.dataframe(
                    df_news[["headline", "sentiment", "source", "matched_keywords", "link"]],
                    column_config={"link": st.column_config.LinkColumn("Read")},
                    use_container_width=True
                )
            except Exception:
                st.dataframe(df_news[["headline", "sentiment", "source", "matched_keywords", "link"]], use_container_width=True)
        else:
            st.info("No relevant headlines found this run.")

    with tab_report:
        if result.get("report"):
            st.subheader("Risk Assessment Overview")
            colA, colB = st.columns([1, 1])
            with colA:
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number", value=severity["combined_score"],
                    title={"text": f"Severity: {severity['tier']}"},
                    gauge={
                        "axis": {"range": [0, 3]},
                        "bar": {"color": {"Low": "green", "Medium": "orange", "High": "red"}[severity["tier"]]},
                        "steps": [
                            {"range": [0, 1], "color": "rgba(0,200,0,0.2)"},
                            {"range": [1, 2], "color": "rgba(255,165,0,0.2)"},
                            {"range": [2, 3], "color": "rgba(255,0,0,0.2)"},
                        ],
                    }
                ))
                fig_gauge.update_layout(height=300)
                st.plotly_chart(fig_gauge, use_container_width=True)
            with colB:
                fig_contrib = go.Figure(go.Bar(
                    x=["Macro Signal", "News Signal"],
                    y=[severity["macro_score"], severity["news_score"]],
                    marker_color=["#4C78A8", "#F58518"]
                ))
                fig_contrib.update_layout(title="What Triggered This (x = multiples of threshold)", yaxis_title="Score (1.0 = at threshold)", height=300)
                st.plotly_chart(fig_contrib, use_container_width=True)

            st.subheader("Portfolio Composition")
            if result.get("_tickers") and result.get("_weights"):
                fig_pie = px.pie(names=result["_tickers"], values=result["_weights"], title="Current Portfolio Weights")
                st.plotly_chart(fig_pie, use_container_width=True)

            st.divider()
            st.subheader("Full Report")
            st.markdown(result["report"])
        else:
            st.info("No report generated this run — severity was Low. Enable 'Force full analysis' in the sidebar to see a sample report.")

    with tab_trace:
        st.subheader("Pipeline Execution Trace")
        macro = result.get("macro_result")
        st.markdown("**1. Macro Check** — ran")
        if macro:
            st.write(f"Latest change: {macro['pct_change']*100:.2f}% | z-score: {macro['z_score']:.2f} (threshold: ±{macro['threshold']})")
            st.write("🚨 Flagged as surprise" if macro["surprise"] else "✅ Within normal range")

        news = result.get("news_result")
        st.markdown("**2. News Check** — ran")
        if news:
            st.write(f"Average sentiment: {news['average_sentiment']:.2f} across {len(news['headlines'])} headlines")
            thresh = result.get("news_threshold", {})
            st.write(f"Threshold used: {thresh.get('threshold', 0.3):.2f} — {thresh.get('mode', 'fallback')}")
        else:
            st.write("No relevant headlines found")

        st.markdown("**3. Severity Assessment**")
        if severity:
            st.write(f"Macro score: {severity['macro_score']:.2f}x threshold | News score: {severity['news_score']:.2f}x threshold")
            st.write(f"{TIER_COLORS[severity['tier']]} Combined severity: **{severity['tier']}** ({severity['combined_score']:.2f})")

        escalated = result.get("risk_result") is not None
        st.markdown("**4. Decision (conditional edge)**")
        if escalated:
            reason = "forced by demo mode" if (result.get("force_full_run") and severity["tier"] == "Low") else f"severity tier is {severity['tier']}"
            st.success(f"Escalated to full risk check — reason: {reason}")
        else:
            st.warning("Skipped risk check — severity tier is Low (cost-saving branch)")

        st.markdown("**5. Risk Check**")
        st.write("Ran — recomputed VaR / Sharpe / Drawdown / Correlation" if escalated else "Skipped")
        st.markdown("**6. Memo Synthesis**")
        st.write(f"Ran — LLM generated a {severity['tier']}-urgency risk memo" if result.get("memo") else "Skipped (no memo needed)")
        st.markdown("**7. Full Report Generation**")
        st.write("Ran — full structured report generated" if result.get("report") else "Skipped (no report needed)")

    with tab_history:
        st.subheader("Run History (this session)")
        if st.session_state.history:
            rows = []
            for r in st.session_state.history:
                r_sev = r.get("severity", {})
                rows.append({
                    "Time": r.get("_timestamp"),
                    "Severity": f"{TIER_COLORS.get(r_sev.get('tier'), '')} {r_sev.get('tier', 'N/A')}",
                    "Combined Score": round(r_sev.get("combined_score", 0), 2),
                    "Escalated": "Yes" if r.get("risk_result") else "No",
                    "Report Generated": "Yes" if r.get("report") else "No",
                })
            df_hist = pd.DataFrame(rows)
            st.dataframe(df_hist, use_container_width=True)
            fig_line = px.line(df_hist, x="Time", y="Combined Score", markers=True, title="Severity Score Across Runs")
            fig_line.add_hline(y=1.0, line_dash="dash", line_color="orange", annotation_text="Medium threshold")
            fig_line.add_hline(y=2.0, line_dash="dash", line_color="red", annotation_text="High threshold")
            st.plotly_chart(fig_line, use_container_width=True)
        else:
            st.info("No runs yet.")
else:
    with tab_dashboard:
        st.info("Configure your portfolio in the sidebar and click 'Run Check Now' to begin.")

# --- Risk Analytics tab ---
with tab_analytics:
    st.subheader("Deep Risk Analytics")
    if "latest_analytics" in st.session_state:
        a = st.session_state.latest_analytics
        if a["dropped_tickers"]:
            st.warning(f"Excluded due to insufficient data: {', '.join(a['dropped_tickers'])}")

        st.markdown("#### Point Estimates")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Historical VaR (95%)", f"{a['hist_var_95']*100:.2f}%")
        col2.metric("Parametric VaR (95%)", f"{a['parametric_var_95']*100:.2f}%")
        col3.metric("CVaR / Expected Shortfall", f"{a['cvar_95']*100:.2f}%")
        col4.metric("Annualized Volatility", f"{a['annualized_vol']*100:.2f}%")
        col5, col6, col7 = st.columns(3)
        col5.metric("Sortino Ratio", f"{a['sortino']:.2f}" if a['sortino'] else "N/A")
        col6.metric("Beta (vs Nifty 50)", f"{a['beta']:.2f}" if a['beta'] else "N/A")
        col7.metric("Skew / Kurtosis", f"{a['skewness']:.2f} / {a['kurtosis']:.2f}")

        st.divider()
        st.markdown("#### VaR Methodology Comparison")
        fig_var_compare = go.Figure(go.Bar(
            x=["Historical VaR", "Parametric VaR", "CVaR (Expected Shortfall)"],
            y=[a["hist_var_95"], a["parametric_var_95"], a["cvar_95"]],
            marker_color=["#4C78A8", "#72B7B2", "#E45756"]
        ))
        fig_var_compare.update_layout(yaxis_title="Loss (%)", yaxis_tickformat=".1%")
        st.plotly_chart(fig_var_compare, use_container_width=True)

        st.markdown("#### Return Distribution (with VaR & CVaR marked)")
        fig_dist = px.histogram(a["portfolio_returns"], nbins=50, title="Daily Portfolio Returns Distribution")
        fig_dist.add_vline(x=-a["hist_var_95"], line_dash="dash", line_color="orange", annotation_text="VaR 95%")
        fig_dist.add_vline(x=-a["cvar_95"], line_dash="dash", line_color="red", annotation_text="CVaR 95%")
        fig_dist.update_layout(showlegend=False)
        st.plotly_chart(fig_dist, use_container_width=True)

        if a["beta_scatter_data"] is not None:
            st.markdown("#### Portfolio vs Nifty 50 (Beta)")
            fig_beta = px.scatter(
                a["beta_scatter_data"], x="benchmark", y="portfolio", trendline="ols",
                labels={"benchmark": "Nifty 50 Daily Return", "portfolio": "Portfolio Daily Return"}
            )
            st.plotly_chart(fig_beta, use_container_width=True)
            st.caption(f"Beta = {a['beta']:.2f} — the slope of this line. Above 1 means the portfolio amplifies market moves.")
    else:
        st.info("Click 'Run Deep Risk Analytics' in the sidebar.")

# --- Monte Carlo tab ---
with tab_montecarlo:
    st.subheader("Monte Carlo Simulation")
    if "latest_mc" in st.session_state:
        mc = st.session_state.latest_mc
        st.caption(f"Method: {mc['method']} | {mc['n_simulations']} simulations over {mc['n_days']} trading days")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Simulated VaR (95%)", f"{mc['var_95_mc']*100:.2f}%")
        col2.metric("Simulated CVaR (95%)", f"{mc['cvar_95_mc']*100:.2f}%")
        col3.metric("Probability of Loss", f"{mc['prob_loss']*100:.1f}%")
        col4.metric("Median Terminal Value", f"₹{mc['median_terminal_value']:,.0f}")

        st.markdown("#### Simulated Portfolio Paths")
        fig_fan = go.Figure()
        for path in mc["sample_paths"]:
            fig_fan.add_trace(go.Scatter(y=path, mode="lines", line=dict(width=0.5, color="rgba(100,150,255,0.15)"), showlegend=False))
        fig_fan.add_hline(y=mc["initial_value"], line_dash="dash", line_color="white", annotation_text="Initial Value")
        fig_fan.update_layout(title="Sample of Simulated Paths", yaxis_title="Portfolio Value (₹)", xaxis_title="Trading Day")
        st.plotly_chart(fig_fan, use_container_width=True)

        st.markdown("#### Distribution of Terminal Portfolio Values")
        fig_term = px.histogram(mc["terminal_values"], nbins=60, title=f"Portfolio Value After {mc['n_days']} Trading Days")
        fig_term.add_vline(x=mc["initial_value"], line_dash="dash", line_color="white", annotation_text="Initial Value")
        fig_term.update_layout(xaxis_title="Portfolio Value (₹)", showlegend=False)
        st.plotly_chart(fig_term, use_container_width=True)
    else:
        st.info("Configure and click 'Run Monte Carlo Simulation' in the sidebar.")

# --- ML Forecast tab ---
with tab_ml:
    st.subheader("ML-Based Risk Event Forecasting")
    if "latest_ml" in st.session_state:
        ml = st.session_state.latest_ml
        st.caption(f"Event definition: a daily return below the {ml['loss_percentile']}th percentile ({ml['threshold_return']*100:.2f}%) — used here as a stand-in for the 'equipment failure / default event' style forecasting task.")

        col1, col2, col3 = st.columns(3)
        col1.metric("Decision Tree Accuracy", f"{ml['tree']['accuracy']*100:.1f}%")
        col2.metric("Logistic Regression Accuracy", f"{ml['logistic']['accuracy']*100:.1f}%")
        col3.metric("Base Event Rate (test set)", f"{ml['event_rate_test']*100:.1f}%")
        st.caption(f"Trained on {ml['n_train']} days, tested on {ml['n_test']} days (chronological split, no lookahead).")

        colA, colB = st.columns(2)
        with colA:
            st.markdown("#### Decision Tree — Feature Importance")
            fi_df = pd.DataFrame(list(ml["tree"]["feature_importances"].items()), columns=["Feature", "Importance"]).sort_values("Importance")
            fig_fi = px.bar(fi_df, x="Importance", y="Feature", orientation="h")
            st.plotly_chart(fig_fi, use_container_width=True)
            st.metric("Latest predicted probability (Tree)", f"{ml['tree']['latest_probability']*100:.1f}%")

        with colB:
            st.markdown("#### Logistic Regression — Coefficients")
            coef_df = pd.DataFrame(list(ml["logistic"]["coefficients"].items()), columns=["Feature", "Coefficient"]).sort_values("Coefficient")
            fig_coef = px.bar(coef_df, x="Coefficient", y="Feature", orientation="h", color="Coefficient", color_continuous_scale="RdBu_r")
            st.plotly_chart(fig_coef, use_container_width=True)
            st.metric("Latest predicted probability (Logistic)", f"{ml['logistic']['latest_probability']*100:.1f}%")

        st.markdown("#### Confusion Matrix (Decision Tree)")
        cm = ml["tree"]["confusion_matrix"]
        fig_cm = px.imshow(cm, text_auto=True, labels=dict(x="Predicted", y="Actual"),
                            x=["No Event", "Event"], y=["No Event", "Event"], color_continuous_scale="Blues")
        st.plotly_chart(fig_cm, use_container_width=True)

        st.markdown("#### Predicted Probability Over Test Period")
        fig_prob = go.Figure()
        fig_prob.add_trace(go.Scatter(x=ml["test_dates"], y=ml["test_tree_probs"], mode="lines", name="Predicted Probability (Tree)"))
        event_dates = [d for d, a_ in zip(ml["test_dates"], ml["test_actual"]) if a_ == 1]
        if event_dates:
            fig_prob.add_trace(go.Scatter(
                x=event_dates, y=[0.5]*len(event_dates), mode="markers",
                marker=dict(color="red", size=8, symbol="x"), name="Actual Event Occurred"
            ))
        fig_prob.update_layout(yaxis_title="Predicted Probability of Large-Loss Day")
        st.plotly_chart(fig_prob, use_container_width=True)
    else:
        st.info("Configure and click 'Train Risk Event Models' in the sidebar.")

# --- Income Strategy tab ---
with tab_income:
    st.subheader("Covered Call Income Overlay")
    if "latest_cc" in st.session_state:
        cc = st.session_state.latest_cc
        st.caption(f"Assumes selling {cc['otm_pct']*100:.0f}% OTM calls, {cc['days_to_expiry']} days to expiry, on a ₹{cc['portfolio_value']:,.0f} notional portfolio. Priced via Black-Scholes using historical realized volatility.")

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Premium Income", f"₹{cc['total_premium_income']:,.0f}")
        col2.metric("Yield This Period", f"{cc['total_yield_pct']*100:.2f}%")
        col3.metric("Annualized Yield Estimate", f"{cc['annualized_yield_pct']*100:.2f}%")

        df_cc = pd.DataFrame(cc["breakdown"])
        st.markdown("#### Per-Holding Breakdown")
        st.dataframe(
            df_cc[["ticker", "spot_price", "strike_price", "annualized_vol", "call_price_per_share",
                   "premium_income", "probability_of_assignment", "delta"]],
            use_container_width=True
        )

        st.markdown("#### Premium Income by Holding")
        fig_income = px.bar(df_cc, x="ticker", y="premium_income", color="probability_of_assignment",
                             color_continuous_scale="RdYlGn_r", title="Premium Income vs Assignment Risk")
        st.plotly_chart(fig_income, use_container_width=True)

        st.markdown("#### Payoff Diagram")
        selected_ticker_for_payoff = st.selectbox("Select holding to view payoff", df_cc["ticker"].tolist())
        row = df_cc[df_cc["ticker"] == selected_ticker_for_payoff].iloc[0]
        prices, stock_payoff, cc_payoff = compute_payoff_curve(
            S=row["spot_price"], K=row["strike_price"], premium=row["call_price_per_share"]
        )
        fig_payoff = go.Figure()
        fig_payoff.add_trace(go.Scatter(x=prices, y=stock_payoff, name="Stock Only", line=dict(color="gray", dash="dash")))
        fig_payoff.add_trace(go.Scatter(x=prices, y=cc_payoff, name="Covered Call", line=dict(color="green")))
        fig_payoff.add_vline(x=row["spot_price"], line_dash="dot", annotation_text="Current Price")
        fig_payoff.add_vline(x=row["strike_price"], line_dash="dot", line_color="red", annotation_text="Strike")
        fig_payoff.update_layout(title=f"Payoff at Expiry — {selected_ticker_for_payoff}", xaxis_title="Price at Expiry", yaxis_title="P&L per Share (₹)")
        st.plotly_chart(fig_payoff, use_container_width=True)

        st.markdown("""
**Trade-off, stated honestly:** selling a covered call generates immediate premium income, but caps upside at the strike — if the stock rallies past it, you're obligated to sell there, missing further gains you'd have had just holding the stock. This suits flat-to-mildly-bullish conditions, not a strong rally.
""")
    else:
        st.info("Configure and click 'Run Covered Call Analysis' in the sidebar.")

# --- Methodology tab ---
with tab_methodology:
    st.subheader("How Sentinel Works")

    st.markdown("""
Sentinel is built as a directed graph (via LangGraph) of specialized agents, each responsible for one part of the risk-monitoring workflow. Data flows through the graph in one direction, and a conditional branch decides how much work is worth doing based on what's actually found.
""")

    with st.expander("1. Data Sources"):
        st.markdown("""
- **Equity prices**: `yfinance`, pulled live for each ticker's daily closing price
- **Macroeconomic data**: FRED API, used for India CPI as a proxy for macro surprises
- **News**: RSS feeds (Economic Times), filtered by ticker-related keywords
- **LLM inference**: Groq API (`openai/gpt-oss-120b`) for sentiment scoring, memo writing, and report generation
""")

    with st.expander("2. Macro Agent — Z-Score Surprise Detection"):
        st.latex(r"z = \frac{\Delta_{latest} - \mu_{historical}}{\sigma_{historical}}")
        st.markdown("""
Rather than a fixed percentage threshold, the macro agent measures how many standard deviations the latest change is from its own historical average change. The baseline always excludes the most recent observation, so the check isn't circular.
""")

    with st.expander("3. News Agent — LLM Sentiment Scoring"):
        st.markdown("""
Headlines are pulled from RSS, filtered by ticker keyword, then sent to an LLM with a strict instruction to return sentiment scores from -1 to +1 as JSON only. The average sentiment across matched headlines becomes the signal.
""")

    with st.expander("4. Adaptive Threshold — Learning What's Normal"):
        st.latex(r"\text{threshold} = |\mu_{past} + z \cdot \sigma_{past}|")
        st.markdown("""
Sentinel tracks sentiment from its own past runs within a session and sets the threshold based on that history. With fewer than 3 past runs, it falls back to a fixed default (0.3).
""")

    with st.expander("5. Severity Assessment — Dynamic Action Selection"):
        st.latex(r"\text{score} = \frac{|\text{signal}|}{\text{threshold}}")
        st.markdown("""
Each signal becomes a multiple of its own threshold. The combined severity score is the maximum of the macro and news scores, bucketed into Low / Medium / High — this determines how much downstream work the pipeline actually does.
""")

    with st.expander("6. Risk Metrics — Formulas Used"):
        st.latex(r"VaR_{95} = -\text{Quantile}_{5\%}(\text{portfolio returns})")
        st.latex(r"CVaR_{95} = -\mathbb{E}[\text{returns} \mid \text{returns} \leq \text{VaR cutoff}]")
        st.latex(r"\text{Sharpe} = \frac{\bar{r} - r_f}{\sigma_r} \times \sqrt{252}")
        st.latex(r"\text{Sortino} = \frac{\bar{r} - r_f}{\sigma_{downside}} \times \sqrt{252}")
        st.latex(r"\beta = \frac{\text{Cov}(r_{portfolio}, r_{benchmark})}{\text{Var}(r_{benchmark})}")
        st.markdown("Auto-assigned weights come from the global minimum-variance portfolio:")
        st.latex(r"\min_w \; w^T \Sigma w \quad \text{s.t.} \sum w_i = 1, \; w_i \geq 0")

    with st.expander("7. Monte Carlo Simulation"):
        st.markdown("""
The bootstrap method resamples the portfolio's actual historical daily returns (with replacement) to build thousands of possible future paths, preserving real fat tails and skew rather than assuming a normal distribution.
""")

    with st.expander("8. ML Risk Event Forecasting"):
        st.markdown("""
A Decision Tree and Logistic Regression predict whether the next trading day falls below a chosen historical loss percentile, using only features available strictly before that day. The train/test split is chronological, avoiding lookahead bias.
""")

    with st.expander("9. Covered Call Income Strategy"):
        st.latex(r"C = S \cdot N(d_1) - K e^{-rT} N(d_2)")
        st.markdown("""
Prices a call option via Black-Scholes using each holding's historical realized volatility as a proxy for implied volatility, then estimates the income from selling that call against existing shares. Assignment probability uses N(d2), the risk-neutral probability of finishing in-the-money.
""")

    with st.expander("10. Failure Handling (Robustness)"):
        st.markdown("""
Tickers with more than 5% missing price data are automatically detected and excluded (with a visible warning), rather than letting one bad symbol silently corrupt calculations or crash the app.
""")