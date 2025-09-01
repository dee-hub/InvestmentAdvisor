# streamlit_cryptoadvisor_app.py
# -------------------------------------------------------------
# Simple "CryptoAdvisor" research app (alpha)
# Sidebar lets you pick an asset and a section. Only "Research" works.
# "Backtrack" and "Test strategy" are intentionally deadlinks (coming soon).
#
# Quickstart:
#   pip install streamlit yfinance plotly pandas numpy
#   streamlit run streamlit_cryptoadvisor_app.py
# -------------------------------------------------------------

from __future__ import annotations
import math
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from helpers.indicators import *
from helpers.functionalities import *
# ----------------------------
# Page setup
# ----------------------------
st.set_page_config(
    page_title="CryptoAdvisor (alpha)",
    page_icon="🪙",
    layout="wide",
)

@st.cache_data(show_spinner=False, ttl=10 * 60)
def fetch_news(symbol: str) -> list[dict]:
    """Fetch latest news via yfinance's Yahoo Finance API. Falls back to empty list."""
    try:
        t = yf.Ticker(symbol)
        news = t.news or []  # list of dicts with keys: title, link, publisher, providerPublishTime, etc.
        # sort newest first just in case
        news = sorted(news, key=lambda x: x.get('providerPublishTime', 0), reverse=True)
        return news
    except Exception:
        return []


# ----------------------------
# Sidebar
# ----------------------------
with st.sidebar:
    st.markdown("## 🪙 CryptoAdvisor")
    st.caption("alpha • research-only")

    section = st.radio("Section", ["Research", "Backtrack", "Test strategy"], index=0, help="Only Research is available right now.")

    st.markdown("---")
    asset_type = st.selectbox("Asset type", ["Crypto", "Stock"], index=1)

    popular_crypto = ["BTC", "ETH", "SOL", "XRP", "DOGE", "ADA", "BNB", "AVAX", "TON", "DOT"]
    popular_stocks = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AMD", "NFLX", "BRK-B"]

    if asset_type == "Crypto":
        st.markdown("Crpto asset research is underdevelopment")
        st.stop()
        #base_choice = st.selectbox("Popular cryptos", popular_crypto, index=0)
        #default_symbol = f"{base_choice}-USD"
        #sym_help = "Enter a custom crypto pair as a Yahoo Finance symbol, e.g., BTC-USD, ETH-USD."
    else:
        default_symbol = st.selectbox("Popular stocks", popular_stocks)
        sym_help = "Enter any stock ticker, e.g., AAPL, MSFT. Use Yahoo Finance symbol format."

    st.markdown("---")
    st.caption("Not financial advice. Data may be delayed.")

# ----------------------------
# Main content routing
# ----------------------------
if section != "Research":
    st.title("🚧 Coming soon")
    st.info(
        "The 'Backtrack' and 'Test strategy' sections are placeholders for now. "
        "Only the 'Research' tab is active in this alpha."
    )
    st.stop()

# ----------------------------
# Research Section
# ----------------------------
st.title("🔎 Research")
st.caption("Quick KPIs, price action, and headlines for your selected asset.")

col_head_left, col_head_right = st.columns([0.7, 0.3])
with col_head_left:
    st.subheader(f"{default_symbol}")
with col_head_right:
    st.write("")

conn = load_db(db_path="data/market/market.duckdb")

if not conn:
    st.error("No data found for that symbol/period. Try a different ticker, a simpler interval (e.g., 1d), or a broader history window.")
    st.stop()

# Compute KPIs
# KPI cards
fig, explain, result = plot_rolling_returns(con=conn, symbol=default_symbol)
m1, m2, m3, m4 = st.columns(4)
with m1:
    cagr = get_cagr_from_duckdb(con=conn, symbol=default_symbol)
    cagr_vals = [val.get("rolling_cagr") for val in result]
    end_date = result[-2].get("end_date").date().year
    st.metric(label="CAGR", value=cagr.get("cagr"),
              delta=f"{(cagr_vals[-1] - cagr_vals[-2]) / cagr_vals[-1]:.2f}% from {end_date}",
              chart_data=cagr_vals, chart_type="line", help=cagr.get("description"), 
              border=True, width="stretch", height=200)
with m2:
    fig, summary, kpi_sum = plot_max_drawdown(con=conn, symbol=default_symbol)
    st.metric(label="Max Drawdown", help=summary, value=f"{kpi_sum[0]:.2%}", width="stretch", border=True, height=200, chart_data=kpi_sum[1]["drawdown_pct"], chart_type="area")
with m3:
    vol_lists = calculate_annual_volatility_by_year(con=conn, symbol=default_symbol)
    note= "Annualized volatility is a measure of how much an asset’s price fluctuates in a given year. It captures the standard deviation of returns, scaled to reflect a full year's worth of movement"
    chg_in_ann_vol = f"{(cagr_vals[-1] - cagr_vals[-2]) / cagr_vals[-1]:.2f}% from {end_date}"
    ann_vols_dt = [ann_vols.get("annualized_volatility_pct") for ann_vols in vol_lists]
    st.metric(label="Volatility (ann.)", value=vol_lists[-1].get("annualized_volatility_pct"), 
              delta=f"{(ann_vols_dt[-1] - ann_vols_dt[-2]) / ann_vols_dt[-1]:.2f}% from {end_date}", help=note, width="stretch", border=True, height=200, 
              chart_data=ann_vols_dt, chart_type="area")
with m4:
    metrics, note = compute_sharpe_sortino_calmar(con=conn, symbol=default_symbol)
    sharpe = metrics["sharpe"]                  
    sortino = metrics["sortino"]                
    calmar = metrics["calmar"]                  
    delta_text = f"Sortino {sortino:.2f} · Calmar {calmar:.2f}"
    st.metric(label=f"Sharpe (rf={2}%)", value=f"{sharpe:.2f}", delta=delta_text, delta_color="off", help=note, width="stretch", border=True, height=200)


# === PERFORMANCE SECTION ===
with st.expander("📈 Performance", expanded=True):
    st.subheader("📊 Rolling Returns (CAGR)")
    st.markdown("""
    Rolling returns show how the asset performed over overlapping **3-year periods**.
    This helps you see how consistently the asset delivered returns — not just the average over time.
    """)
    rolling_fig, rolling_note, _ = plot_rolling_returns(conn, default_symbol, window_years=3)
    st.plotly_chart(rolling_fig, use_container_width=True)
    st.write(rolling_note)

    st.markdown("---")

    st.subheader("📈 Total Return vs Benchmark")
    st.markdown("""
    This compares the growth of a **$10,000 investment** in this asset versus a common benchmark like the S&P 500 (SPY). 
    It helps you understand whether the asset outperformed or lagged the broader market.
    """)
    comp_fig, comp_note = plot_total_return_vs_benchmark(conn, default_symbol, benchmark="SPY")
    st.plotly_chart(comp_fig, use_container_width=True)
    st.markdown(comp_note)
    st.markdown("---")
    best_worst_month = get_best_worst_month_and_hit_rate(conn, default_symbol)
    st.caption(
    f"""
    From **2010-01-31** to **2025-07-31**, **{default_symbol.upper()}** had positive monthly returns in **{best_worst_month['hit_rate']}%** of months. The best-performing month was **{best_worst_month['best_month']}**, returning **{best_worst_month['best_month_return']}%**, while the worst was **{best_worst_month['worst_month']}**, with a return of **{best_worst_month['worst_month_return']}%**.  
    \n On a yearly basis, the best year was **{best_worst_month['best_year']}** with a gain of **{best_worst_month['best_year_return']}%**, and the worst year was **{best_worst_month['worst_year']}**, posting a loss of **{best_worst_month['worst_year_return']}%**.
    """
)
    


with st.expander("⚠️ Risk Metrics", expanded=False):
    st.subheader("📉 Max Drawdown & Duration")
    st.markdown("""
    This shows the **largest drop** from a previous high (drawdown) and how long it took to recover.
    It tells you how painful a worst-case investment period could have been, which is crucial for risk tolerance.
    """)
    dd_fig, dd_note, _ = plot_max_drawdown(conn, default_symbol)
    st.plotly_chart(dd_fig, use_container_width=True)
    st.markdown(dd_note)

    st.markdown("---")

    st.subheader("📏 Risk Ratios: Sharpe, Sortino, Calmar")
    st.markdown("""
    These ratios tell you how efficiently the asset has delivered returns relative to its risk:
    - **Sharpe**: return vs. total volatility.
    - **Sortino**: return vs. downside-only volatility (focuses on losses).
    - **Calmar**: return vs. worst-case loss (drawdown).
    
    Higher is better — values above **1.0** generally signal good risk-adjusted performance.
    """)
    metrics, note = compute_sharpe_sortino_calmar(conn, default_symbol)
    sharpe_val, sharpe_delta = format_metric(metrics["sharpe"], "sharpe")
    sortino_val, sortino_delta = format_metric(metrics["sortino"], "sortino")
    calmar_val, calmar_delta = format_metric(metrics["calmar"], "calmar")

    m1, m2, m3 = st.columns(3)
    m1.metric("Sharpe", sharpe_val, border=True, delta=sharpe_delta, delta_color="normal")
    m2.metric("Sortino", sortino_val, border=True, delta=sortino_delta, delta_color="normal")
    m3.metric("Calmar", calmar_val, border=True, delta=calmar_delta, delta_color="normal")
    st.caption(note)

    st.markdown("---")

    st.subheader("📊 Monthly Return Distribution")
    st.markdown("""
    This chart shows how monthly returns have been distributed over time — are they mostly positive? How extreme are the losses?  
    Look for **skew** (bias toward gains or losses) and **kurtosis** (how often extreme outcomes happen).
    """)
    dist_fig, dist_note = plot_return_distribution(conn, default_symbol)
    st.plotly_chart(dist_fig, use_container_width=True)
    st.caption(dist_note)

# Footer disclaimer
st.caption("This app is for informational and educational purposes only and does not constitute financial advice.")
