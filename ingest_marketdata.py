#!/usr/bin/env python3
"""
Ingest **stocks** from **Stooq** (via pandas-datareader) into a local DuckDB for
long‑horizon research. Crypto/Binance paths removed per request.

- Daily bars only (EOD). OK if data is ~1 day delayed.
- Idempotent & incremental: each run inserts rows **after the latest ts** already in DB.
- Defaults to start at **2010‑01‑01** unless DB already has newer data.

Install:
  pip install duckdb pandas pyarrow pandas-datareader

Examples:
  # Single symbol
  python ingest_to_duckdb_stooq.py \
    --db ~/data/market/market.duckdb \
    --symbols AAPL \
    --start 2010-01-01

  # Multiple symbols (comma-separated)
  python ingest_to_duckdb_stooq.py \
    --db ~/data/market/market.duckdb \
    --symbols AAPL,MSFT,NVDA
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional

import duckdb
import pandas as pd

try:
    import pandas_datareader.data as pdr
except Exception as e:  # pragma: no cover
    raise SystemExit("pandas-datareader is required. Install with: pip install pandas-datareader") from e

# ----------------------------
# Config
# ----------------------------
DEFAULT_START = "2010-01-01"
SUPPORTED_INTERVALS = {"1d"}  # Stooq reader here is EOD only


@dataclass
class IngestConfig:
    db_path: str
    symbols: List[str]
    interval: str = "1d"
    start: str = DEFAULT_START
    end: Optional[str] = None  # defaults to today


# ----------------------------
# Helpers
# ----------------------------

def _parse_date(s: Optional[str], default_to_eod: bool = True) -> datetime:
    if not s:
        d = pd.Timestamp.today().normalize()
        return (d + pd.Timedelta(hours=23, minutes=59, seconds=59)).to_pydatetime()
    dt = pd.to_datetime(s, errors="raise")
    if default_to_eod and isinstance(dt, pd.Timestamp) and dt.tzinfo is None and dt.hour == 0 and dt.minute == 0:
        dt = dt + pd.Timedelta(hours=23, minutes=59, seconds=59)
    return dt.to_pydatetime() if isinstance(dt, pd.Timestamp) else dt


def _fetch_stooq(symbol: str, start_dt: datetime, end_dt: datetime) -> pd.DataFrame:
    """Fetch daily bars from Stooq (ascending by date)."""
    df = pdr.DataReader(symbol, "stooq", start=start_dt.date(), end=end_dt.date())
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.sort_index()  # ascending
    out = pd.DataFrame({
        "ts": pd.to_datetime(df.index).tz_localize(None),
        "open": pd.to_numeric(df.get("Open"), errors="coerce"),
        "high": pd.to_numeric(df.get("High"), errors="coerce"),
        "low": pd.to_numeric(df.get("Low"), errors="coerce"),
        "close": pd.to_numeric(df.get("Close"), errors="coerce"),
        "volume": pd.to_numeric(df.get("Volume"), errors="coerce"),
        "adj_close": pd.to_numeric(df.get("Adj Close"), errors="coerce") if "Adj Close" in df.columns else pd.NA,
    })
    out = out.dropna(subset=["ts","open","high","low","close"]).reset_index(drop=True)
    return out


# ----------------------------
# DuckDB schema & utils
# ----------------------------
DDL = """
CREATE TABLE IF NOT EXISTS ohlcv (
    symbol TEXT,
    asset_type TEXT,
    source TEXT,
    ts TIMESTAMP,
    open DOUBLE,
    high DOUBLE,
    low DOUBLE,
    close DOUBLE,
    volume DOUBLE,
    adj_close DOUBLE
);
"""

# Unique index enforces 1 row per (symbol, interval, ts)
CREATE_UNIQ_IDX = """
CREATE UNIQUE INDEX IF NOT EXISTS idx_ohlcv_unique
ON ohlcv(symbol, ts);
"""


def ensure_schema(con: duckdb.DuckDBPyConnection) -> None:
    con.execute(DDL)
    con.execute(CREATE_UNIQ_IDX)


def existing_max_ts(con: duckdb.DuckDBPyConnection, symbol: str) -> Optional[datetime]:
    res = con.execute("SELECT max(ts) FROM ohlcv WHERE symbol = ?;", [symbol]).fetchone()
    return res[0] if res and res[0] is not None else None


def _insert_df(con: duckdb.DuckDBPyConnection, df: pd.DataFrame, symbol: str) -> int:
    if df is None or df.empty:
        return 0
    df = df.copy()
    df.insert(0, "symbol", symbol.upper())
    df.insert(1, "asset_type", "stock")
    df.insert(2, "source", "stooq_eod")
    # De-dupe within batch
    df = df.drop_duplicates(subset=["symbol","ts"], keep="last")
    con.register("_staging", df)
    # Use NOT EXISTS for wide DuckDB compatibility (avoids LEFT ANTI syntax)
    inserted = con.execute(
        """
        INSERT INTO ohlcv
        SELECT s.symbol, s.asset_type, s.source, s.ts, s.open, s.high, 
               s.low, s.close, s.volume, s.adj_close
        FROM _staging s
        WHERE NOT EXISTS (
            SELECT 1 FROM ohlcv t
            WHERE t.symbol = s.symbol AND t.ts = s.ts
        );
        """
    ).rowcount
    return inserted


# ----------------------------
# Core ingestion
# ----------------------------

def ingest_stooq_to_duckdb(cfg: IngestConfig) -> int:
    con = duckdb.connect(cfg.db_path)
    ensure_schema(con)

    user_start = _parse_date(cfg.start)
    user_end = _parse_date(cfg.end)

    total_inserted = 0
    for raw_sym in cfg.symbols:
        symbol = raw_sym.strip().upper()
        max_ts = existing_max_ts(con, symbol)
        start_dt = max_ts if max_ts and max_ts > user_start else user_start
        df = _fetch_stooq(symbol, start_dt, user_end)
        total_inserted += _insert_df(con, df, symbol)
        print(f"{symbol}: inserted {total_inserted} total rows so far.")

    print(f"Done. Inserted {total_inserted} rows into {cfg.db_path}.")
    return total_inserted


# ----------------------------
# CLI
# ----------------------------

def _parse_symbols_arg(s: str) -> List[str]:
    return [t.strip() for t in s.split(",") if t.strip()]


def main():
    p = argparse.ArgumentParser(description="Ingest stocks from Stooq into DuckDB (incremental, daily, from 2010).")
    p.add_argument("--db", required=True, help="Path to DuckDB file (will be created if missing).")
    p.add_argument("--symbols", required=True, help="Comma-separated tickers, e.g., AAPL,MSFT,NVDA")
    p.add_argument("--interval", default="1d", choices=sorted(SUPPORTED_INTERVALS), help="Bar interval (Stooq path is daily only).")
    p.add_argument("--start", default=DEFAULT_START, help="Start date (default 2010-01-01). If DB has newer data, that wins.")
    p.add_argument("--end", default=None, help="End date (default today).")
    args = p.parse_args()

    cfg = IngestConfig(
        db_path=args.db,
        symbols=_parse_symbols_arg(args.symbols),
        interval=args.interval,
        start=args.start,
        end=args.end,
    )
    ingest_stooq_to_duckdb(cfg)


if __name__ == "__main__":
    main()
