from __future__ import annotations

# ----------------------------
# Helpers
# ----------------------------
import streamlit as st
import math
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import duckdb
from datetime import datetime
import plotly.graph_objects as go
from typing import Dict, Tuple, Dict
from math import sqrt
#--------------------------------------------------------------------------------------#
#----------------------------Performance Indicators-----------------------------------#
#-------------------------------------------------------------------------------------#

def load_db(db_path):
    conn = duckdb.connect(db_path)
    return conn

def get_cagr_from_duckdb(con: any, symbol: str) -> dict:
    """
    Calculate CAGR for a symbol using DuckDB OHLCV data.
    """

    # Pull first and last valid close price
    query = f"""
        SELECT 
            MIN(ts) AS start_date,
            MAX(ts) AS end_date,
            FIRST(close ORDER BY ts ASC) AS start_price,
            FIRST(close ORDER BY ts DESC) AS end_price
        FROM ohlcv
        WHERE symbol = '{symbol.upper()}' AND close IS NOT NULL;
    """
    result = con.execute(query).fetchone()
    

    start_date, end_date, start_price, end_price = result

    if not all([start_date, end_date, start_price, end_price]):
        return {
            "error": "Not enough data to compute CAGR."
        }

    # Compute number of years
    start_dt = datetime.strptime(str(start_date), "%Y-%m-%d %H:%M:%S")
    end_dt = datetime.strptime(str(end_date), "%Y-%m-%d %H:%M:%S")
    num_years = (end_dt - start_dt).days / 365.25

    # Compute CAGR
    cagr = (end_price / start_price) ** (1 / num_years) - 1
        
    return {
        "symbol": symbol.upper(),
        "start_date": str(start_date)[:10],
        "end_date": str(end_date)[:10],
        "start_price": round(start_price, 4),
        "end_price": round(end_price, 4),
        "num_years": round(num_years, 2),
        "cagr": round(cagr * 100, 2),  # in %
        "description": f"CAGR (Compound Annual Growth Rate) shows the smoothed annual return assuming reinvestment over time. \n\n This figure is computed over the past {num_years:.1f} years rolling window, so the current CAGR is for {start_date.date()} to {end_date.date()}. The chart is a 3 year rolling average."
    }

def plot_rolling_returns(con: any, symbol: str, window_years: int = 3):
    """
    Plots annualized rolling returns (CAGR) using yearly data from DuckDB,
    excluding the current year to avoid incomplete return periods.

    Parameters:
        db_path (str): Path to DuckDB database.
        symbol (str): Asset symbol (e.g., "AAPL").
        window_years (int): Rolling window in years (e.g., 3, 5).

    Returns:
        go.Figure: Plotly figure showing the rolling CAGR.
    """
    # Connect to DB and fetch daily close prices
    
    df = con.execute(f"""
        SELECT ts, close
        FROM ohlcv
        WHERE symbol = '{symbol.upper()}'
        ORDER BY ts
    """).fetchdf()

    # Clean and resample to year-end closing prices
    df['ts'] = pd.to_datetime(df['ts'])
    df.set_index('ts', inplace=True)
    yearly = df['close'].resample('Y').last().dropna()

    # Remove current year if it's incomplete
    current_year = datetime.utcnow().year
    if yearly.index[-1].year == current_year:
        yearly = yearly.iloc[:-1]

    # Calculate rolling CAGR
    results = []
    for i in range(window_years, len(yearly)):
        start = yearly.index[i - window_years]
        end = yearly.index[i]
        start_price = yearly.iloc[i - window_years]
        end_price = yearly.iloc[i]
        cagr = (end_price / start_price) ** (1 / window_years) - 1
        results.append({
            "start_date": start,
            "end_date": end,
            "rolling_cagr": round(cagr * 100, 2),
            "hover_text": f"{start.date()} → {end.date()}<br>CAGR: {round(cagr * 100, 2)}%"
        })

    rolling_df = pd.DataFrame(results)

    # Plot using Plotly
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=rolling_df["end_date"],
        y=rolling_df["rolling_cagr"],
        mode='lines',
        name=f"{window_years}-Year Rolling CAGR",
        line=dict(width=2),
        text=rolling_df["hover_text"],
        hoverinfo="text"
    ))

    fig.update_layout(
        title=f"{symbol.upper()} - {window_years}-Year Rolling CAGR (Yearly)",
        xaxis_title="End Date",
        yaxis_title="Rolling CAGR (%)",
        template="plotly_white",
        hovermode="x unified"
    )

    example = rolling_df.iloc[-1]
    cagr_value = example["rolling_cagr"]
    start_dt = example["start_date"].year
    end_dt = example["end_date"].year
    final_value = 10000 * (1 + cagr_value / 100) ** window_years

    explanation = (
        f"**Rolling CAGR ({window_years}-Year Windows)**\n\n"
        f"This chart shows how the compound annual growth rate (CAGR) of {symbol.upper()} "
        f"has evolved over time for every {window_years}-year investment window.\n\n"
        f"For example, if you had invested **10,000USD** at the start of **{start_dt}** and held it "
        f"until the end of **{end_dt}**, your investment would have grown at an average rate of "
        f"**{cagr_value}% per year**, ending up with approximately **${final_value:,.2f}**.\n\n"
        f"Rolling returns help assess how consistent performance has been over long periods, "
        f"highlighting periods of strong or weak momentum."
    )

    return fig, explanation, results

def plot_total_return_vs_benchmark(
    con: any,
    symbol: str,
    benchmark: str = "SPY",
    start_date: str = "2010-01-01"
    ) -> tuple[go.Figure, str]:
    """
    Plots the total return of an asset vs a benchmark as a normalized $10,000 investment chart.

    Parameters:
    db_path (str): Path to DuckDB database.
    symbol (str): Asset symbol (e.g., "AAPL").
    benchmark (str): Benchmark symbol (e.g., "SPY").
    start_date (str): Start date for comparison.

    Returns:
    Tuple[go.Figure, str]: Plotly chart and explanatory footnote.
    """
    


    # Query both asset and benchmark
    query = f"""
        SELECT ts, symbol, close
        FROM ohlcv
        WHERE symbol IN ('{symbol.upper()}', '{benchmark.upper()}')
        AND ts >= '{start_date}'
        ORDER BY ts
    """
    df = con.execute(query).fetchdf()
    df['ts'] = pd.to_datetime(df['ts'])

    # Pivot into wide format
    df_wide = df.pivot(index='ts', columns='symbol', values='close').dropna()
    # Normalize to $10,000
    initial = df_wide.iloc[0]
    norm_df = df_wide / initial * 10000

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=norm_df.index, y=norm_df[symbol.upper()], mode='lines', name=symbol.upper()
        ))
    fig.add_trace(go.Scatter(
        x=norm_df.index, y=norm_df[benchmark.upper()], mode='lines', name=benchmark.upper()
        ))

    fig.update_layout(
        title=f"Total Return: {symbol.upper()} vs {benchmark.upper()} (Growth of $10,000)",
        xaxis_title="Date",
        yaxis_title="Value of $10,000 Investment ($)",
        template="plotly_white",
        hovermode="x unified"
        )

    # Footnote text with example
    symbol_return = norm_df[symbol.upper()].iloc[-1] / 10000 - 1
    benchmark_return = norm_df[benchmark.upper()].iloc[-1] / 10000 - 1

    footnote = (
        f"If you had invested **10,000USD** in {symbol.upper()} starting from **{start_date}**, "
        f"it would have grown to **{norm_df[symbol.upper()].iloc[-1]:,.2f}**, a total return of **{symbol_return*100:.2f}%**.\n"
        f"In comparison, the benchmark **{benchmark.upper()}** would have grown to **{norm_df[benchmark.upper()].iloc[-1]:,.2f}USD**, "
        f"a total return of {benchmark_return*100:.2f}%."
        )

    return fig, footnote

def get_best_worst_month_and_hit_rate(con: any, symbol: str) -> Dict:
    """
    Computes the best/worst month, best/worst year, and hit rate for the asset.

    Parameters:
        db_path (str): Path to DuckDB database.
        symbol (str): Asset symbol (e.g., "AAPL").

    Returns:
        Dict: Dictionary of stats.
    """
    
    df = con.execute(f"""
        SELECT ts, close
        FROM ohlcv
        WHERE symbol = '{symbol.upper()}'
        ORDER BY ts
    """).fetchdf()

    df['ts'] = pd.to_datetime(df['ts'])
    df.set_index('ts', inplace=True)
    monthly_returns = df['close'].resample('M').ffill().pct_change().dropna()
    yearly_returns = df['close'].resample('Y').ffill().pct_change().dropna()

    best_month = monthly_returns.idxmax()
    best_month_val = monthly_returns.max() * 100
    worst_month = monthly_returns.idxmin()
    worst_month_val = monthly_returns.min() * 100

    hit_rate = (monthly_returns > 0).sum() / len(monthly_returns) * 100

    best_year = yearly_returns.idxmax().year
    best_year_val = yearly_returns.max() * 100
    worst_year = yearly_returns.idxmin().year
    worst_year_val = yearly_returns.min() * 100

    return {
        "best_month": best_month.strftime("%Y-%m"),
        "best_month_return": round(best_month_val, 2),
        "worst_month": worst_month.strftime("%Y-%m"),
        "worst_month_return": round(worst_month_val, 2),
        "hit_rate": round(hit_rate, 2),
        "best_year": best_year,
        "best_year_return": round(best_year_val, 2),
        "worst_year": worst_year,
        "worst_year_return": round(worst_year_val, 2),
    }

#--------------------------------------------------------------------------------------#
#----------------------------Risk-----------------------------------#
#-------------------------------------------------------------------------------------#

def calculate_annual_volatility_by_year(con: any, symbol: str) -> List[Dict]:
    """
    Calculates annualized volatility (monthly return std × sqrt(12)) for each year
    and returns a list of dictionaries with start date, end date, and volatility.

    Parameters:
        con (duckdb.DuckDBPyConnection): Active DuckDB connection.
        symbol (str): Asset symbol (e.g., "AAPL").

    Returns:
        List[Dict]: List of annual volatility stats.
    """
    # Fetch data
    df = con.execute(f"""
        SELECT ts, close
        FROM ohlcv
        WHERE symbol = '{symbol.upper()}'
        ORDER BY ts
    """).fetchdf()

    if df.empty:
        return []

    df['ts'] = pd.to_datetime(df['ts'])
    df.set_index('ts', inplace=True)

    # Resample to monthly closes
    monthly = df['close'].resample('M').last()
    monthly_returns = monthly.pct_change().dropna()

    # Group by year
    results = []
    for year, group in monthly_returns.groupby(monthly_returns.index.year):
        if len(group) >= 3:  # Skip years with too little data
            start = group.index.min().strftime("%Y-%m-%d")
            end = group.index.max().strftime("%Y-%m-%d")
            vol = group.std(ddof=0) * np.sqrt(12)
            results.append({
                "year": year,
                "start": start,
                "end": end,
                "annualized_volatility_pct": round(vol * 100, 2)
            })

    return results

def calculate_downside_deviation(con: any, symbol: str, target_return: float = 0.0) -> float:
    """
    Calculates annualized downside deviation of monthly returns.

    Parameters:
        db_path (str): Path to DuckDB database.
        symbol (str): Asset symbol (e.g., "AAPL").
        target_return (float): Minimum acceptable return (default is 0 for absolute downside).

    Returns:
        float: Annualized downside deviation as a percentage.
    """
    # Connect and load prices
    
    df = con.execute(f"""
        SELECT ts, close
        FROM ohlcv
        WHERE symbol = '{symbol.upper()}'
        ORDER BY ts
    """).fetchdf()

    df['ts'] = pd.to_datetime(df['ts'])
    df.set_index('ts', inplace=True)

    # Monthly returns
    monthly = df['close'].resample('M').last()
    returns = monthly.pct_change().dropna()

    # Filter returns below target
    downside_returns = returns[returns < target_return]
    squared_diff = (downside_returns - target_return) ** 2

    # Annualize
    downside_deviation = np.sqrt(squared_diff.mean()) * np.sqrt(12)
    return round(downside_deviation * 100, 2)  # As percentage

def plot_max_drawdown(con: any, symbol: str):
    """
    Plot drawdown curve (monthly) and highlight the worst drawdown,
    showing drawdown period in years/months (no price in hover).

    Returns:
        (go.Figure, str): Plotly figure and summary text.
    """
    # Load daily prices -> monthly closes
    
    df = con.execute(f"""
        SELECT ts, close
        FROM ohlcv
        WHERE symbol = '{symbol.upper()}'
        ORDER BY ts
    """).fetchdf()

    if df.empty:
        raise ValueError(f"No data found for {symbol}")

    df["ts"] = pd.to_datetime(df["ts"])
    df.set_index("ts", inplace=True)
    df = df.resample("M").last().dropna()

    # Drawdown math
    df["running_max"] = df["close"].cummax()
    df["drawdown_pct"] = (df["close"] / df["running_max"]) - 1.0  # negative during drawdowns

    trough_date = df["drawdown_pct"].idxmin()
    max_drawdown = float(df.loc[trough_date, "drawdown_pct"])      # e.g., -0.33

    # Peak is the last time the running max was set before trough
    peak_date = df.loc[:trough_date, "close"].idxmax()

    # Recovery: first time we get back to (or above) the peak close after trough
    peak_level = float(df.loc[peak_date, "close"])
    rec_slice = df.loc[trough_date:]
    recovered_mask = rec_slice["close"] >= peak_level
    if recovered_mask.any():
        recovery_date = recovered_mask[recovered_mask].index[0]
        recovered = True
    else:
        recovery_date = df.index[-1]
        recovered = False

    # Duration in months/years (calendar, not trading days)
    def months_between(a: pd.Timestamp, b: pd.Timestamp) -> int:
        return (b.year - a.year) * 12 + (b.month - a.month)

    total_months = max(0, months_between(peak_date, recovery_date))
    years = total_months // 12
    months = total_months % 12
    period_str = (f"{years}y {months}m" if years else f"{months}m") if recovered else f"ongoing ({years}y {months}m)"

    # Marker positions & hover text (no prices)
    peak_hover = f"Peak<br>Date: {peak_date.date()}"
    trough_hover = f"Trough<br>Date: {trough_date.date()}<br>Max Drawdown: {max_drawdown:.2%}"
    if recovered:
        recovery_hover = f"Recovery<br>Date: {recovery_date.date()}<br>Drawdown Period: {period_str}"
    else:
        recovery_hover = f"No Full Recovery<br>Through: {recovery_date.date()}<br>Drawdown Period: {period_str}"

    marker_x = [peak_date, trough_date, recovery_date]
    marker_y = [
        0.0,                                  # drawdown at peak is 0%
        df.loc[trough_date, "drawdown_pct"]*100,
        0.0 if recovered else df.loc[recovery_date, "drawdown_pct"]*100
    ]
    marker_hover = [peak_hover, trough_hover, recovery_hover]
    marker_labels = ["Peak", "Trough", "Recovery" if recovered else "Current"]

    # Build figure
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df["drawdown_pct"] * 100,
        mode="lines",
        fill="tozeroy",
        name="Drawdown (%)"
    ))
    fig.add_trace(go.Scatter(
        x=marker_x,
        y=marker_y,
        mode="markers+text",
        name="Drawdown Points",
        text=marker_labels,
        textposition="top center",
        marker=dict(size=10),
        hovertext=marker_hover,
        hoverinfo="text"
    ))
    fig.update_layout(
        title=f"{symbol.upper()} — Max Drawdown & Duration (Monthly)",
        xaxis_title="Date",
        yaxis_title="Drawdown (%)",
        template="plotly_white",
        hovermode="x unified"
    )

    # Summary text (uses years/months wording)
    kpi_summary = [max_drawdown, df]
    human_period = f"{years} years {months} months" if years else f"{months} months"
    if recovered:
        summary = (
            f"Between **{peak_date.date()}** and **{recovery_date.date()}**, "
            f"{symbol.upper()} experienced a maximum drawdown of **{max_drawdown:.2%}**. "
            f"The drawdown lasted **{human_period}** until recovery."
        )
    else:
        summary = (
            f"From the peak on **{peak_date.date()}** to **{recovery_date.date()}**, "
            f"{symbol.upper()}'s maximum drawdown is **{max_drawdown:.2%}** and remains "
            f"**unrecovered**, with an ongoing drawdown period of **{human_period}**."
        )

    return fig, summary, kpi_summary

def compute_sharpe_sortino_calmar(
    con: any,
    symbol: str,
    start_date: str = "2010-01-01",
    rf_annual: float = 0.02  # simple risk-free proxy (2%/yr by default)
) -> Tuple[Dict[str, float], str]:
    """
    Compute Sharpe, Sortino, and Calmar ratios from monthly closes (price-only).

    - Sharpe  = (CAGR - rf) / annualized_volatility
    - Sortino = (CAGR - rf) / annualized_downside_deviation
    - Calmar  =  CAGR / |max_drawdown|

    Returns:
        metrics (dict): {
            'cagr_pct', 'vol_ann_pct', 'downside_dev_ann_pct', 'max_dd_pct',
            'sharpe', 'sortino', 'calmar'
        }
        note (str): human-readable footnote to render under KPIs
    """
    # 1) Load daily -> monthly closes
    
    df = con.execute(f"""
        SELECT ts, close
        FROM ohlcv
        WHERE symbol = '{symbol.upper()}'
          AND ts >= '{start_date}'
        ORDER BY ts
    """).fetchdf()
    

    if df.empty:
        raise ValueError(f"No data found for {symbol} from {start_date}")

    df["ts"] = pd.to_datetime(df["ts"])
    df.set_index("ts", inplace=True)

    monthly_close = df["close"].resample("M").last().dropna()

    # Drop ongoing month (avoid partial-month bias)
    today = pd.Timestamp.today()
    if len(monthly_close) and monthly_close.index[-1].to_period("M") == today.to_period("M"):
        # If today is NOT the actual month-end, drop the last point
        if today.day != monthly_close.index[-1].day:
            monthly_close = monthly_close.iloc[:-1]

    if len(monthly_close) < 24:  # need enough data for stable stats
        raise ValueError("Not enough monthly observations to compute risk metrics (need >= 24 months).")

    # 2) Returns and timing
    monthly_ret = monthly_close.pct_change().dropna()

    start_ts = monthly_close.index[0]
    end_ts = monthly_close.index[-1]
    years = (end_ts - start_ts).days / 365.25
    start_price = float(monthly_close.iloc[0])
    end_price = float(monthly_close.iloc[-1])

    # 3) Core stats
    cagr = (end_price / start_price) ** (1 / years) - 1                     # geometric
    vol_ann = monthly_ret.std(ddof=0) * sqrt(12)                            # annualized stdev (population)
    downside_vec = np.minimum(monthly_ret.values - 0.0, 0.0)                # target = 0% (downside only)
    downside_dev_ann = np.sqrt((downside_vec ** 2).mean()) * sqrt(12)       # annualized downside deviation

    # Max drawdown (monthly)
    running_max = monthly_close.cummax()
    dd_series = (monthly_close / running_max) - 1.0
    max_dd = float(dd_series.min())  # negative number; use absolute magnitude for Calmar

    # 4) Ratios per your spec
    excess = cagr - rf_annual
    sharpe = (excess / vol_ann) if vol_ann > 0 else np.nan
    sortino = (excess / downside_dev_ann) if downside_dev_ann > 0 else np.nan
    calmar = (cagr / abs(max_dd)) if abs(max_dd) > 0 else np.nan

    metrics = {
        "cagr_pct": round(cagr * 100, 2),
        "vol_ann_pct": round(vol_ann * 100, 2),
        "downside_dev_ann_pct": round(downside_dev_ann * 100, 2),
        "max_dd_pct": round(abs(max_dd) * 100, 2),
        "sharpe": round(float(sharpe), 2) if np.isfinite(sharpe) else None,
        "sortino": round(float(sortino), 2) if np.isfinite(sortino) else None,
        "calmar": round(float(calmar), 2) if np.isfinite(calmar) else None,
    }

    note = (
        f"Computed from monthly closes (price-only) from {start_ts.date()} to {end_ts.date()}. "
        f"\n\nSharpe = (CAGR − rf) / σ; Sortino = (CAGR − rf) / σ_down; Calmar = CAGR / |MaxDD|. "
        f"rf assumed {rf_annual*100:.1f}%/yr. "
        f"\n\nCAGR {metrics['cagr_pct']}%, Vol {metrics['vol_ann_pct']}%, "
        f"Downside Dev {metrics['downside_dev_ann_pct']}%, Max DD {metrics['max_dd_pct']}%."
    )

    return metrics, note

def plot_return_distribution(
    con: any,
    symbol: str,
    start_date: str = "2010-01-01",
    bins: int = 40,
    overlay_normal: bool = True,
    show_percentiles: bool = True,
) -> Tuple[go.Figure, str]:
    """
    Plot a histogram of MONTHLY returns and report skew & (excess) kurtosis.
    Returns (fig, note).

    - Uses daily closes from DuckDB, resampled to month-end.
    - Drops current (incomplete) month to avoid bias.
    - Histogram shown in % units. If overlay_normal=True, histogram
      is shown as a probability density and a normal PDF is overlaid.
    """
    # 1) Load daily closes
    
    df = con.execute(f"""
        SELECT ts, close
        FROM ohlcv
        WHERE symbol = '{symbol.upper()}'
          AND ts >= '{start_date}'
        ORDER BY ts
    """).fetchdf()
    
    if df.empty:
        raise ValueError(f"No data for {symbol} from {start_date}")

    # 2) Monthly closes; drop incomplete current month
    df["ts"] = pd.to_datetime(df["ts"])
    df.set_index("ts", inplace=True)
    monthly_close = df["close"].resample("M").last().dropna()
    today = pd.Timestamp.today()
    if len(monthly_close) and monthly_close.index[-1].to_period("M") == today.to_period("M"):
        monthly_close = monthly_close.iloc[:-1]

    # 3) Monthly arithmetic returns
    r = monthly_close.pct_change().dropna()
    if len(r) < 12:
        raise ValueError("Not enough monthly observations to build a distribution (need >= 12 months).")

    # Summary stats
    mean_m = r.mean()
    std_m = r.std(ddof=0)
    skew = r.skew()
    ex_kurt = r.kurt()  # pandas uses Fisher definition: excess kurtosis (normal ~ 0)
    neg_share = (r < 0).mean() * 100

    # Values in percent for plotting
    r_pct = r.values * 100
    mean_pct = float(mean_m * 100)
    std_pct = float(std_m * 100)

    # 4) Build figure
    fig = go.Figure()

    # Histogram mode depends on overlay_normal
    hist_kwargs = dict(
        x=r_pct,
        nbinsx=bins,
        name="Monthly Returns",
        opacity=0.9,
    )
    if overlay_normal:
        hist_kwargs["histnorm"] = "probability density"

    fig.add_trace(go.Histogram(**hist_kwargs))

    # Reference lines
    fig.add_vline(x=0, line_width=1, line_dash="dash", annotation_text="0%")
    fig.add_vline(x=mean_pct, line_width=1, line_dash="dot", annotation_text=f"Mean {mean_pct:.2f}%")

    # Optional: 5th–95th percentile band
    note_extra = ""
    if show_percentiles:
        p5, p95 = np.percentile(r_pct, [5, 95])
        fig.add_vrect(x0=p5, x1=p95, fillcolor="LightGray", opacity=0.2, line_width=0,
                      annotation_text="5th–95th pct", annotation_position="top left")
        note_extra = f" 5th–95th percentile range: {p5:.2f}% to {p95:.2f}%."

    # Optional: overlay normal curve with same mean/std as data
    if overlay_normal and std_pct > 0:
        xs = np.linspace(min(r_pct.min(), mean_pct - 4*std_pct),
                         max(r_pct.max(), mean_pct + 4*std_pct), 400)
        pdf = (1 / (std_pct * np.sqrt(2*np.pi))) * np.exp(-0.5 * ((xs - mean_pct) / std_pct) ** 2)
        fig.add_trace(go.Scatter(
            x=xs, y=pdf, mode="lines", name="Normal PDF (mean/std-matched)"
        ))

    fig.update_layout(
        title=f"{symbol.upper()} — Monthly Return Distribution",
        xaxis_title="Monthly return (%)",
        yaxis_title="Density" if overlay_normal else "Frequency",
        template="plotly_white",
        bargap=0.02,
        hovermode="x",
        legend=dict(
            orientation="h",     # Horizontal legend
            yanchor="top",
            y=-0.2,              # Push it below the chart
            xanchor="center",
            x=0.5
        )
    )

    # 5) Footnote
    note = (
        f"Computed from monthly closes (price-only) from **{monthly_close.index[0].date()}** "
        f"to **{monthly_close.index[-1].date()}**. "
        f"\n\n**Skew** = {skew:.2f} "
        f"({'right/upside' if skew>0 else 'left/downside' if skew<0 else 'symmetric'}), "
        f"**Excess kurtosis** = {ex_kurt:.2f} "
        f"({'fat tails' if ex_kurt>0 else 'thin tails' if ex_kurt<0 else 'normal-like'}). "
        f"\n\nMean monthly return **{mean_pct:.2f}**%, stdev **{std_pct:.2f}**%. "
        f"\n\n**{neg_share:.1f}**% of months were negative." + note_extra
    )

    return fig, note