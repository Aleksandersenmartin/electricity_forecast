"""
Fetch commodity prices for electricity price forecasting.

Primary source: CommodityPriceAPI (rates/historical endpoint, day-by-day).
Fallback source: yfinance (free, no API key, good for historical backfill).

The Lite plan for CommodityPriceAPI does not include a timeseries endpoint,
so historical backfill uses yfinance by default, and the API is used for
recent/latest data.

API docs: See docs/commodity_price_api.md for full reference.
Working endpoints (Lite plan):
    /v2/rates/latest          — current rates
    /v2/rates/historical      — OHLC for a single date (param: date)
    /v2/rates/fluctuation     — price change between two dates
    /v2/symbols               — list all symbols
    /v2/usage                 — account usage
"""

import os
import time
import logging
from pathlib import Path

import requests
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

# --- Configuration ---

BASE_URL = "https://api.commoditypriceapi.com/v2"
API_KEY = os.getenv("COMMODITY_API_KEY")

# Symbols relevant for electricity price forecasting
# See docs/commodity_price_api.md section 10 for why these matter
PRIMARY_SYMBOLS = ["TTF-GAS", "BRENTOIL-SPOT", "NG-FUT", "COAL"]

# yfinance ticker mapping for fallback
YFINANCE_TICKERS = {
    "TTF-GAS": "TTF=F",        # TTF Gas futures (EUR/MWh)
    "BRENTOIL-SPOT": "BZ=F",   # Brent crude futures (USD/barrel)
    "NG-FUT": "NG=F",          # US natural gas futures (USD/MMBtu)
    "COAL": "MTF=F",           # Newcastle coal futures (USD/ton)
}

# Project root for data paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


# --- CommodityPriceAPI functions ---


def _check_response(response: requests.Response) -> dict:
    """Check CommodityPriceAPI response and return parsed JSON.

    Args:
        response: The requests Response object.

    Returns:
        Parsed JSON dict on success.

    Raises:
        requests.HTTPError: On HTTP errors.
        ValueError: If API reports success=false.
    """
    if response.status_code == 429:
        logger.warning("Rate limited by CommodityPriceAPI — wait before retrying")
        response.raise_for_status()

    response.raise_for_status()

    data = response.json()
    if not data.get("success", False):
        msg = data.get("message", "Unknown API error")
        raise ValueError(f"CommodityPriceAPI error: {msg}")

    return data


def fetch_latest(symbols: list[str] = PRIMARY_SYMBOLS) -> pd.DataFrame:
    """Fetch latest commodity rates from CommodityPriceAPI.

    Args:
        symbols: List of commodity symbol strings.

    Returns:
        Single-row DataFrame with one column per symbol (latest price).
    """
    if not API_KEY:
        raise RuntimeError("COMMODITY_API_KEY not set in .env")

    response = requests.get(
        f"{BASE_URL}/rates/latest",
        params={"apiKey": API_KEY, "symbols": ",".join(symbols)},
    )
    data = _check_response(response)

    rates = data.get("rates", {})
    timestamp = pd.Timestamp.fromtimestamp(data["timestamp"], tz="UTC")

    row = {"timestamp": timestamp}
    for symbol, value in rates.items():
        row[f"{symbol}_close"] = float(value)

    df = pd.DataFrame([row]).set_index("timestamp")
    return df


def fetch_historical_date(
    date: str,
    symbols: list[str] = PRIMARY_SYMBOLS,
) -> dict:
    """Fetch OHLC rates for a single date from CommodityPriceAPI.

    Args:
        date: Date as "YYYY-MM-DD".
        symbols: List of commodity symbol strings.

    Returns:
        Dict mapping symbol to OHLC dict, e.g.:
        {"TTF-GAS": {"open": 58.26, "high": 58.27, "low": 58.27, "close": 58.27}}
    """
    if not API_KEY:
        raise RuntimeError("COMMODITY_API_KEY not set in .env")

    response = requests.get(
        f"{BASE_URL}/rates/historical",
        params={
            "apiKey": API_KEY,
            "symbols": ",".join(symbols),
            "date": date,
        },
    )
    data = _check_response(response)
    return data.get("rates", {})


# --- yfinance backfill functions ---


def fetch_yfinance(
    start_date: str,
    end_date: str,
    symbols: list[str] = PRIMARY_SYMBOLS,
) -> pd.DataFrame:
    """Fetch historical commodity data from yfinance (free, no API key).

    Uses futures contracts as proxies. Returns daily close prices.

    Args:
        start_date: Start date as "YYYY-MM-DD".
        end_date: End date as "YYYY-MM-DD".
        symbols: List of commodity symbol strings (mapped to yfinance tickers).

    Returns:
        DataFrame with DatetimeIndex and columns like "TTF-GAS_close",
        "BRENTOIL-SPOT_close", etc.
    """
    import yfinance as yf

    all_data = {}

    for symbol in symbols:
        ticker = YFINANCE_TICKERS.get(symbol)
        if not ticker:
            logger.warning("No yfinance ticker mapping for %s — skipping", symbol)
            continue

        logger.info("Fetching %s (%s) from yfinance", symbol, ticker)
        try:
            df = yf.download(
                ticker,
                start=start_date,
                end=end_date,
                progress=False,
                auto_adjust=True,
            )
            if df.empty:
                logger.warning("No yfinance data for %s (%s)", symbol, ticker)
                continue

            # yfinance returns MultiIndex columns: (Price, Ticker)
            # Flatten to single-level by taking the Price level
            if isinstance(df.columns, pd.MultiIndex):
                df = df.droplevel("Ticker", axis=1)

            # Extract OHLC columns, rename with symbol prefix
            for col in ["Open", "High", "Low", "Close"]:
                if col in df.columns:
                    all_data[f"{symbol}_{col.lower()}"] = df[col]

        except Exception as e:
            logger.error("Failed to fetch %s from yfinance: %s", symbol, e)

    if not all_data:
        return pd.DataFrame()

    result = pd.DataFrame(all_data)
    result.index.name = "date"
    result = result.sort_index()

    return result


# --- High-level fetch with caching ---


def fetch_commodities(
    start_date: str,
    end_date: str,
    symbols: list[str] = PRIMARY_SYMBOLS,
    source: str = "yfinance",
    cache: bool = True,
) -> pd.DataFrame:
    """Fetch historical commodity prices with caching.

    For large historical ranges, yfinance is recommended (free, fast, no rate limits).
    CommodityPriceAPI is better for latest/recent data.

    Args:
        start_date: Start date as "YYYY-MM-DD".
        end_date: End date as "YYYY-MM-DD".
        symbols: List of commodity symbol strings.
        source: "yfinance" (default, recommended for backfill) or "api".
        cache: If True, check data/raw/ before fetching and save after.

    Returns:
        DataFrame with DatetimeIndex and OHLC columns per symbol.
    """
    cache_dir = PROJECT_ROOT / "data" / "raw"
    cache_path = cache_dir / f"commodity_{source}_{start_date}_{end_date}.parquet"

    if cache and cache_path.exists():
        logger.info("Loading cached commodity data from %s", cache_path)
        return pd.read_parquet(cache_path)

    if source == "yfinance":
        df = fetch_yfinance(start_date, end_date, symbols)
    elif source == "api":
        df = _fetch_api_range(start_date, end_date, symbols)
    else:
        raise ValueError(f"Unknown source: {source}. Use 'yfinance' or 'api'.")

    if cache and not df.empty:
        cache_dir.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache_path)
        logger.info("Cached commodity data to %s", cache_path)

    return df


def _fetch_api_range(
    start_date: str,
    end_date: str,
    symbols: list[str],
) -> pd.DataFrame:
    """Fetch a date range from CommodityPriceAPI day-by-day.

    Slow due to one API call per day. Use for short ranges or recent data.
    For large backfills, use yfinance instead.

    Args:
        start_date: Start date as "YYYY-MM-DD".
        end_date: End date as "YYYY-MM-DD".
        symbols: List of commodity symbol strings.

    Returns:
        DataFrame with DatetimeIndex and OHLC columns per symbol.
    """
    dates = pd.date_range(start=start_date, end=end_date, freq="B")  # Business days
    rows = []

    for date in dates:
        date_str = date.strftime("%Y-%m-%d")
        logger.info("Fetching commodity data for %s", date_str)

        try:
            rates = fetch_historical_date(date_str, symbols)
            row = {"date": date}
            for symbol, ohlc in rates.items():
                if isinstance(ohlc, dict):
                    for field in ["open", "high", "low", "close"]:
                        row[f"{symbol}_{field}"] = ohlc.get(field)
                else:
                    row[f"{symbol}_close"] = float(ohlc)
            rows.append(row)
        except Exception as e:
            logger.warning("Failed for %s: %s", date_str, e)

        time.sleep(1)  # Respect rate limits

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows).set_index("date").sort_index()
    return df


def fetch_commodities_daily_filled(
    start_date: str,
    end_date: str,
    symbols: list[str] = PRIMARY_SYMBOLS,
    source: str = "yfinance",
    cache: bool = True,
) -> pd.DataFrame:
    """Fetch commodity prices and forward-fill to every calendar day.

    Args:
        start_date: Start date as "YYYY-MM-DD".
        end_date: End date as "YYYY-MM-DD".
        symbols: List of commodity symbol strings.
        source: "yfinance" or "api".
        cache: If True, use caching for the raw fetch.

    Returns:
        DataFrame with a complete daily DatetimeIndex, forward-filled.
    """
    df = fetch_commodities(start_date, end_date, symbols, source=source, cache=cache)

    if df.empty:
        return df

    full_range = pd.date_range(start=start_date, end=end_date, freq="D")
    df = df.reindex(full_range)
    df.index.name = "date"
    df = df.ffill().bfill()

    return df


# --- Entry point for testing ---

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    print(f"COMMODITY_API_KEY configured: {bool(API_KEY)}")
    print(f"Primary symbols: {PRIMARY_SYMBOLS}")

    # Test 1: Latest rates from CommodityPriceAPI
    print("\n--- CommodityPriceAPI: Latest rates ---")
    try:
        df_latest = fetch_latest()
        print(df_latest)
    except Exception as e:
        print(f"Error: {e}")

    # Test 2: yfinance backfill (2020–current)
    print("\n--- yfinance: Historical 2020-01-01 to 2026-02-22 ---")
    df = fetch_commodities("2020-01-01", "2026-02-22", source="yfinance")

    if not df.empty:
        print(f"Shape: {df.shape}")
        print(f"Date range: {df.index.min()} to {df.index.max()}")
        print(f"Columns: {list(df.columns)}")
        print(f"\nFirst 5 rows:")
        print(df.head())
        print(f"\nLast 5 rows:")
        print(df.tail())
        print(f"\nBasic stats:")
        print(df.describe())
        print(f"\nMissing values:")
        print(df.isna().sum())
    else:
        print("No data returned!")
