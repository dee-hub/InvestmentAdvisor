#!/usr/bin/env python3
"""
Ingest OHLCV into a local DuckDB for long-horizon research.
Now supports **direct pulls from Stooq** (via pandas-datareader) for stocks,
plus the existing CSV paths for crypto/stocks.

- Optimized for **daily** bars (ok if data is ~1 day delayed)
- Idempotent & incremental: on each run, picks up **after the last timestamp** in the DB
- Default start is **2010-01-01** unless DB already has newer data

Quickstart:
  pip install duckdb pandas pyarrow pandas-datareader

  # Pull AAPL from Stooq into DuckDB (daily)
  python ingest_to_duckdb_stooq.py \
      --db ~/data/market/market.duckdb \
      --asset-type stock \
      --symbol AAPL \
      --interval 1d \
      --pull-from stooq

  # Crypto from CSVs (Binance bulk examples remain supported)
  python ingest_to_duckdb_stooq.py \
      --data-dir ~/data/market \
      --db ~/data/market/market.duckdb \
      --asset-type crypto \
      --symbol BTCUSDT \
      --interval 1d \
      --pull-from csv
"""
from __future__ import annotations

import argparse
import glob
import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Optional

import duckdb
import pandas as pd

# Try to import pandas-datareader for Stooq
try:
    import pandas_datareader.data as pdr
    _HAS_PDR = True
except Exception:
    _HAS_PDR = False

# ----------------------------
# Config
# ----------------------------
DEFAULT_START = "2010-01-01"
SUPPORTED_INTERVALS = {"1d", "1h", "30m", "15m"}  # Stooq path enforces 1d


@dataclass
class IngestConfig:
    data_dir: Optional[str]
    db_path: str
    asset_type: str  # "crypto" | "stock"
    symbol: str      # e.g., "BTCUSDT" for crypto, "AAPL" for stock
    interval: str    # e.g., "1d"
    pull_from: str   # "stooq" | "csv"
    start: str = DEFAULT_START
    end: Optional[str] = None  # default: today


# ----------------------------
# Utility
# ----------------------------

def _parse_date(s: Optional[str], default_to_eod: bool = True) -> datetime:
    if not s:
        # today at 23:59:59
        d = pd.Timestamp.today().normalize()
        return (d + pd.Timedelta(hours=23, minutes=59, seconds=59)).to_pydatetime()
    dt = pd.to_datetime(s, errors="raise")
    if default_to_eod and isinstance(dt, pd.Timestamp) and dt.tzinfo is None and dt.hour == 0 and dt.minute == 0:
        dt = dt + pd.Timedelta(hours=23, minutes=59, seconds=59)
    return dt.to_pydatetime() if isinstance(dt, pd.Timestamp) else dt


def _binance_csv_files(root: str, symbol: str, interval: str) -> List[str]:
    pattern1 = os.path.join(root, "**", f"{symbol}-{interval}-*.csv")
    pattern2 = os.path.join(root, "**", f"{symbol}-{interval.upper()}-*.csv")
    files = sorted(set(glob.glob(pattern1, recursive=True)) | set(glob.glob(pattern2, recursive=True)))
    return files


def _read_binance_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, header=None)
    if df.shape[1] < 6:
        return pd.DataFrame()
    df = df.iloc[:, :12]
    df.columns = [
        "open_time","open","high","low","close","volume",
        "close_time","quote_vol","trades","taker_buy_base","taker_buy_quote","ignore"
    ][:df.shape[1]]
    ts = pd.to_numeric(df["open_time"], errors="coerce")
    if ts.dropna().max() and ts.dropna().max() > 10_000_000_000_000:
        idx = pd.to_datetime(ts, unit="us")
    else:
        idx = pd.to_datetime(ts, unit="ms")
    out = pd.DataFrame({
        "ts": pd.to_datetime(idx).tz_localize(None),
        "open": pd.to_numeric(df["open"], errors="coerce"),
        "high": pd.to_numeric(df["high"], errors="coerce"),
        "low": pd.to_numeric(df["low"], errors="coerce"),
        "close": pd.to_numeric(df["close"], errors="coerce"),
        "volume": pd.to_numeric(df["volume"], errors="coerce"),
        "adj_close": pd.NA,
    })
    return out.dropna(subset=["ts","open","high","low","close"]).reset_index(drop=True)


def _read_stock_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    dt_col = None
    for c in df.columns:
        cl = c.lower().replace(" ", "")
        if cl in {"date","datetime","time"}:
            dt_col = c
            break
    if dt_col is None:
        dt_col = df.columns[0]
    rename_map = {}
    for c in df.columns:
        cl = c.lower().replace(" ", "")
        if cl == "open": rename_map[c] = "open"
        elif cl == "high": rename_map[c] = "high"
        elif cl == "low": rename_map[c] = "low"
        elif cl == "close": rename_map[c] = "close"
        elif cl in {"adjclose","adjustedclose","adj_close"}: rename_map[c] = "adj_close"
        elif cl == "volume": rename_map[c] = "volume"
    df = df.rename(columns=rename_map)
    ts = pd.to_datetime(df[dt_col], errors="coerce").dt.tz_localize(None)
    out = pd.DataFrame({
        "ts": ts,
        "open": pd.to_numeric(df.get("open"), errors="coerce"),
        "high": pd.to_numeric(df.get("high"), errors="coerce"),
        "low": pd.to_numeric(df.get("low"), errors="coerce"),
        "close": pd.to_numeric(df.get("close"), errors="coerce"),
        "volume": pd.to_numeric(df.get("volume"), errors="coerce"),
        "adj_close": pd.to_numeric(df.get("adj_close"), errors="coerce"),
    })
    return out.dropna(subset=["ts","open","high","low","close"]).reset_index(drop=True)


# ----------------------------
# Stooq fetcher (stocks)
# ----------------------------

def _fetch_stooq(symbol: str, start_dt: datetime, end_dt: datetime) -> pd.DataFrame:
    if not _HAS_PDR:
        raise RuntimeError("pandas-datareader is required for Stooq. pip install pandas-datareader")
    # Stooq returns most-recent-first; we sort ascending.
    df = pdr.DataReader(symbol, "stooq", start=start_dt.date(), end=end_dt.date())
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.sort_index()  # ascending by date
    # Normalize to canonical columns
    # Stooq provides: Open, High, Low, Close, Volume (Adj Close not guaranteed)
    out = pd.DataFrame({
        "ts": pd.to_datetime(df.index).tz_localize(None),
        "open": pd.to_numeric(df.get("Open"), errors="coerce"),
        "high": pd.to_numeric(df.get("High"), errors="coerce"),
        "low": pd.to_numeric(df.get("Low"), errors="coerce"),
        "close": pd.to_numeric(df.get("Close"), errors="coerce"),
        "volume": pd.to_numeric(df.get("Volume"), errors="coerce"),
        "adj_close": pd.to_numeric(df.get("Adj Close"), errors="coerce") if "Adj Close" in df.columns else pd.NA,
    })
    return out.dropna(subset=["ts","open","high","low","close"]).reset_index(drop=True)


# ----------------------------
# Core ingestion
# ----------------------------
DDL = """
CREATE TABLE IF NOT EXISTS ohlcv (
    symbol TEXT,
    asset_type TEXT,
    source TEXT,
    interval TEXT,
    ts TIMESTAMP,
    open DOUBLE,
    high DOUBLE,
    low DOUBLE,
    close DOUBLE,
    volume DOUBLE,
    adj_close DOUBLE,
    PRIMARY KEY (symbol, interval, ts)
);
"""

CREATE_UNIQ_IDX = """
CREATE UNIQUE INDEX IF NOT EXISTS idx_ohlcv_unique
ON ohlcv(symbol, interval, ts);
"""


def ensure_schema(con: duckdb.DuckDBPyConnection) -> None:
    con.execute(DDL)
    con.execute(CREATE_UNIQ_IDX)


def existing_max_ts(con: duckdb.DuckDBPyConnection, symbol: str, interval: str) -> Optional[datetime]:
    q = "SELECT max(ts) FROM ohlcv WHERE symbol = ? AND interval = ?;"
    res = con.execute(q, [symbol, interval]).fetchone()
    return res[0] if res and res[0] is not None else None


def _insert_df(con: duckdb.DuckDBPyConnection, df: pd.DataFrame, symbol: str, asset_type: str, interval: str, source: str) -> int:
    if df is None or df.empty:
        return 0
    df = df.copy()
    df.insert(0, "symbol", symbol.upper())
    df.insert(1, "asset_type", asset_type.lower())
    df.insert(2, "source", source)
    df.insert(3, "interval", interval)
    df = df.drop_duplicates(subset=["symbol","interval","ts"], keep="last")
    con.register("_staging", df)
    inserted = con.execute(
        """
        INSERT INTO ohlcv
        SELECT s.symbol, s.asset_type, s.source, s.interval,
               s.ts, s.open, s.high, s.low, s.close, s.volume, s.adj_close
        FROM _staging s
        LEFT ANTI JOIN ohlcv t
        ON s.symbol = t.symbol AND s.interval = t.interval AND s.ts = t.ts;
        """
    ).rowcount
    return inserted


def ingest_to_duckdb(cfg: IngestConfig) -> int:
    """Ingest from Stooq or CSVs into DuckDB, incrementally from last ts (or cfg.start)."""
    if cfg.pull_from == "stooq" and cfg.asset_type.lower() != "stock":
        raise ValueError("Stooq path is for asset_type=stock only.")
    if cfg.pull_from == "stooq" and cfg.interval != "1d":
        raise ValueError("Stooq path currently supports daily (1d) bars only.")
    if cfg.interval not in SUPPORTED_INTERVALS:
        raise ValueError(f"Unsupported interval '{cfg.interval}'. Supported: {sorted(SUPPORTED_INTERVALS)}")

    os.makedirs(os.path.dirname(os.path.expanduser(cfg.db_path)), exist_ok=True)
    con = duckdb.connect(os.path.expanduser(cfg.db_path))
    ensure_schema(con)

    user_start = _parse_date(cfg.start)
    user_end = _parse_date(cfg.end)

    max_ts = existing_max_ts(con, cfg.symbol, cfg.interval)
    start_dt = max_ts if max_ts and max_ts > user_start else user_start

    total_inserted = 0

    if cfg.pull_from == "stooq":
        # Fetch EOD from Stooq
        df = _fetch_stooq(cfg.symbol, start_dt, user_end)
        total_inserted += _insert_df(con, df, cfg.symbol, cfg.asset_type, cfg.interval, source="stooq_eod")
    else:
        # CSV path (crypto or stock)
        files: List[str] = []
        reader = None
        source = None
        if cfg.asset_type.lower() == "crypto":
            if not cfg.data_dir:
                raise ValueError("--data-dir is required for crypto CSV ingestion")
            files = _binance_csv_files(os.path.expanduser(cfg.data_dir), cfg.symbol, cfg.interval)
            reader = _read_binance_csv
            source = "binance_bulk"
        else:
            if not cfg.data_dir:
                raise ValueError("--data-dir is required for stock CSV ingestion (or use --pull-from stooq)")
            pattern = os.path.join(os.path.expanduser(cfg.data_dir), "**", f"*{cfg.symbol.upper()}*.csv")
            files = sorted(glob.glob(pattern, recursive=True))
            reader = _read_stock_csv
            source = "stock_csv"
        if not files:
            print("No CSV files found matching your criteria.")
            return 0
        frames = []
        for fp in files:
            try:
                df = reader(fp)
                if df.empty:
                    continue
                frames.append(df)
            except Exception as e:
                print(f"WARN: failed to read {fp}: {e}")
        if not frames:
            print("No usable rows in CSV files.")
            return 0
        df_all = pd.concat(frames).dropna(subset=["ts"]).sort_values("ts")
        df_all = df_all[(df_all["ts"] > start_dt) & (df_all["ts"] <= user_end)]
        total_inserted += _insert_df(con, df_all, cfg.symbol, cfg.asset_type, cfg.interval, source=source)

    print(f"Inserted {total_inserted} rows into {cfg.db_path}.")
    return total_inserted


# ----------------------------
# CLI
# ----------------------------

def main():
    p = argparse.ArgumentParser(description="Ingest OHLCV into DuckDB (Stooq or CSV), incremental from 2010.")
    p.add_argument("--data-dir", default=None, help="Root folder containing CSV files (if using --pull-from csv).")
    p.add_argument("--db", required=True, dest="db_path", help="Path to DuckDB file (will be created if missing).")
    p.add_argument("--asset-type", required=True, choices=["crypto","stock"], help="Asset type.")
    p.add_argument("--symbol", required=True, help="Stock ticker (e.g., AAPL) or crypto symbol (e.g., BTCUSDT).")
    p.add_argument("--interval", default="1d", choices=sorted(SUPPORTED_INTERVALS), help="Bar interval.")
    p.add_argument("--pull-from", default="csv", choices=["stooq","csv"], help="Data source. Use stooq for stocks.")
    p.add_argument("--start", default=DEFAULT_START, help="Start date (default 2010-01-01). If DB has newer data, that wins.")
    p.add_argument("--end", default=None, help="End date (default today).")
    args = p.parse_args()

    cfg = IngestConfig(
        data_dir=args.data_dir,
        db_path=args.db_path,
        asset_type=args.asset_type,
        symbol=args.symbol,
        interval=args.interval,
        pull_from=args.pull_from,
        start=args.start,
        end=args.end,
    )
    ingest_to_duckdb(cfg)


if __name__ == "__main__":
    main()
