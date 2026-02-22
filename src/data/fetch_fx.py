"""
Fetch EUR/NOK exchange rates from Norges Bank.

Retrieves daily EUR/NOK rates from the Norges Bank Data Warehouse (SDMX REST API).
No authentication required — fully open.

Norges Bank only publishes rates on business days. Weekends and Norwegian public
holidays have no data. Use forward-fill to fill gaps when merging with hourly data.

API docs: https://www.norges-bank.no/en/topics/statistics/open-data/guide-data-warehouse/
Query builder: https://app.norges-bank.no/query/#/en/
"""

import logging
from pathlib import Path

import requests
import pandas as pd

logger = logging.getLogger(__name__)

# --- Configuration ---

NORGES_BANK_BASE_URL = "https://data.norges-bank.no/api/data"

# EXR = Exchange Rates, B = Business frequency, EUR = base, NOK = quote, SP = spot
EUR_NOK_FLOW = "EXR/B.EUR.NOK.SP"

# Project root for data paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def _check_response(response: requests.Response) -> None:
    """Check Norges Bank API response and raise descriptive errors.

    Args:
        response: The requests Response object.

    Raises:
        requests.HTTPError: With descriptive message.
    """
    if response.status_code == 200:
        return

    error_messages = {
        400: "Bad request — check date format (YYYY-MM-DD)",
        404: "No data found for the requested period",
        500: "Norges Bank API internal server error",
    }

    msg = error_messages.get(
        response.status_code,
        f"Unexpected status code: {response.status_code}",
    )

    logger.error("Norges Bank API error %d: %s", response.status_code, msg)
    response.raise_for_status()


def _parse_sdmx_json(data: dict) -> pd.DataFrame:
    """Parse SDMX-JSON response from Norges Bank into a DataFrame.

    The SDMX-JSON structure maps observation indices to time periods:
    - observations: {"0": ["11.7138"], "1": ["11.721"], ...}
    - TIME_PERIOD values: [{"id": "2025-02-03"}, {"id": "2025-02-04"}, ...]

    Args:
        data: Parsed JSON response from Norges Bank API.

    Returns:
        DataFrame with DatetimeIndex (date) and column "eur_nok".
    """
    datasets = data.get("data", {}).get("dataSets", [])
    if not datasets:
        logger.warning("No datasets in response")
        return pd.DataFrame()

    # Get time period dimension (maps indices to dates)
    obs_dimensions = data["data"]["structure"]["dimensions"]["observation"]
    time_dim = next(d for d in obs_dimensions if d["id"] == "TIME_PERIOD")
    time_values = time_dim["values"]

    # Get observations from the single series
    series = datasets[0].get("series", {})
    if not series:
        logger.warning("No series in dataset")
        return pd.DataFrame()

    # There's only one series key (e.g., "0:0:0:0")
    series_key = next(iter(series))
    observations = series[series_key].get("observations", {})

    # Build rows: index → date, value → rate
    rows = []
    for idx_str, values in observations.items():
        idx = int(idx_str)
        date = time_values[idx]["id"]
        rate = float(values[0])
        rows.append({"date": date, "eur_nok": rate})

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()

    return df


def fetch_eur_nok(
    start_date: str,
    end_date: str,
    cache: bool = True,
) -> pd.DataFrame:
    """Fetch daily EUR/NOK exchange rates from Norges Bank.

    Returns business-day data only. Use forward-fill when merging with
    hourly data to handle weekends and holidays.

    Args:
        start_date: Start date as "YYYY-MM-DD".
        end_date: End date as "YYYY-MM-DD".
        cache: If True, check data/raw/ before fetching and save after.

    Returns:
        DataFrame with DatetimeIndex (date) and column "eur_nok".
        Index contains only business days (no weekends/holidays).
    """
    cache_dir = PROJECT_ROOT / "data" / "raw"
    cache_path = cache_dir / f"fx_eur_nok_{start_date}_{end_date}.parquet"

    if cache and cache_path.exists():
        logger.info("Loading cached FX data from %s", cache_path)
        return pd.read_parquet(cache_path)

    url = f"{NORGES_BANK_BASE_URL}/{EUR_NOK_FLOW}"
    params = {
        "startPeriod": start_date,
        "endPeriod": end_date,
        "format": "sdmx-json",
    }

    logger.info("Fetching EUR/NOK rates from %s to %s", start_date, end_date)
    response = requests.get(url, params=params)
    _check_response(response)

    df = _parse_sdmx_json(response.json())

    if cache and not df.empty:
        cache_dir.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache_path)
        logger.info("Cached FX data to %s", cache_path)

    return df


def fetch_eur_nok_daily_filled(
    start_date: str,
    end_date: str,
    cache: bool = True,
) -> pd.DataFrame:
    """Fetch EUR/NOK rates and forward-fill to a complete daily series.

    Fills weekends and holidays by carrying forward the last known rate.
    Useful for merging with daily or hourly data.

    Args:
        start_date: Start date as "YYYY-MM-DD".
        end_date: End date as "YYYY-MM-DD".
        cache: If True, use caching for the raw fetch.

    Returns:
        DataFrame with a complete DatetimeIndex (every calendar day)
        and column "eur_nok", forward-filled.
    """
    df = fetch_eur_nok(start_date, end_date, cache=cache)

    if df.empty:
        return df

    # Reindex to full calendar day range and forward-fill
    full_range = pd.date_range(start=start_date, end=end_date, freq="D")
    df = df.reindex(full_range)
    df.index.name = "date"
    df = df.ffill()

    # First days may be NaN if start_date is a weekend — backfill those
    df = df.bfill()

    return df


# --- Entry point for testing ---

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    # Test: fetch EUR/NOK from 2020 to current
    logger.info("Testing: Fetching EUR/NOK rates 2020-01-01 to 2026-02-22")
    df_raw = fetch_eur_nok("2020-01-01", "2026-02-22")

    if not df_raw.empty:
        print(f"\n--- Raw (business days only) ---")
        print(f"Shape: {df_raw.shape}")
        print(f"Date range: {df_raw.index.min()} to {df_raw.index.max()}")
        print(f"\nFirst 5 rows:")
        print(df_raw.head())
        print(f"\nLast 5 rows:")
        print(df_raw.tail())
        print(f"\nBasic stats:")
        print(df_raw.describe())
        print(f"\nMissing values: {df_raw.isna().sum().sum()}")
    else:
        print("No data returned!")

    # Also test the forward-filled version
    df_filled = fetch_eur_nok_daily_filled("2020-01-01", "2026-02-22")
    print(f"\n--- Forward-filled (all calendar days) ---")
    print(f"Shape: {df_filled.shape}")
    print(f"Missing values: {df_filled.isna().sum().sum()}")
    # Show a weekend to verify forward-fill
    print(f"\nWeekend check (Fri–Mon around 2025-02-07):")
    print(df_filled.loc["2025-02-06":"2025-02-10"])
