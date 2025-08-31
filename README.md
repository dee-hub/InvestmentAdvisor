# Investment Analysis Framework

For long-horizon investment research (not trading), you want a small, clear, evidence-based set of metrics that answer key questions:

* What has this asset done?
* How bumpy was the ride?
* How does it behave alongside other assets?
* Is it currently expensive?

This framework breaks the analysis into tiers based on what can be computed from **end-of-day (EOD) prices** (from Stooq) vs. what requires **fundamentals** or additional data.

---

## Tier 1 — Price-Based Metrics (Available Now)

### Performance

* **CAGR** (since 2010, 10-year, 5-year): Geometric annual return.
* **Rolling returns**: 1/3/5/10-year windows and rolling CAGR chart.
* **Total return vs. benchmark**: Based on normalized growth (price-only).
* **Best/Worst month and year**, **hit rate** (% of positive months).

### Risk

* **Volatility**: Annualized standard deviation of monthly returns.
* **Downside deviation**: Only negative months, annualized.
* **Max drawdown** and **drawdown duration**: How deep and how long.
* **Sharpe ratio**: `(CAGR − risk-free rate) / volatility`
* **Sortino ratio**: `(CAGR − risk-free rate) / downside deviation`
* **Calmar ratio**: `CAGR / max drawdown`
* **Return distribution**: Histogram with skew and excess kurtosis.

### Trend & Momentum (v2)

* **12-1 momentum**: 12-month return, skipping the most recent month.
* **200-day and 50-day SMA trend**:

  * % time above the 200-day SMA.
  * Last golden/death cross signal.
* **Distance from all-time high** and **52-week high/low range**.

### Diversification (v3)

* **Correlation and beta** vs. benchmark (e.g., SPY or VTI).
* **R²** and **tracking error** (for ETF-like behavior).
* **Information ratio**: Excess return relative to tracking error.

### Liquidity (Sanity Checks)

* **Average daily dollar volume**: Price × Volume.
* **Volume regime flag**: Filters for low-liquidity periods.

## Tier 2 — Fundamentals (Coming soon)