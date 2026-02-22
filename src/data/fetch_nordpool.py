"""
Fetch day-ahead electricity prices from hvakosterstrommen.no API.

Retrieves hourly day-ahead prices for Norwegian bidding zones (NO1–NO5).
The data originates from ENTSO-E (the EU transparency platform) and is
served by hvakosterstrommen.no — a free, public API with no authentication.

No API key required. No third-party library beyond requests/pandas.

Data availability:
    - Continuous from October 2021 to present (all 5 Norwegian zones)
    - Patchy/missing for 2020–Sept 2021
    - For earlier data, use ENTSO-E directly (fetch_electricity.py)

The API returns one day at a time per zone, so historical backfills loop
over dates. Completed years are cached as Parquet files in data/raw/.

Zone mapping:
    This project uses "NO_1" format (matching ENTSO-E / entsoe-py).
    The API uses "NO1" format in the URL path.
    Conversion is handled internally by this module.

Prices are returned in EUR/MWh (the API provides EUR/kWh, multiplied by 1000).
"""

import logging
import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BASE_URL = "https://www.hvakosterstrommen.no/api/v1/prices"

ZONES_PROJECT_TO_API: dict[str, str] = {
    "NO_1": "NO1",
    "NO_2": "NO2",
    "NO_3": "NO3",
    "NO_4": "NO4",
    "NO_5": "NO5",
}

ZONES_API_TO_PROJECT: dict[str, str] = {v: k for k, v in ZONES_PROJECT_TO_API.items()}

ALL_ZONES: list[str] = list(ZONES_PROJECT_TO_API.keys())

# Rate limiting — be polite to the free API
REQUEST_DELAY_SECONDS: float = 0.3

# Request timeout in seconds
REQUEST_TIMEOUT: int = 15


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _cache_path(zone: str, year: int) -> Path:
    """Build a standardized cache file path for a zone-year.

    Args:
        zone: Bidding zone in project format (e.g., "NO_1").
        year: Year of data.

    Returns:
        Path to the Parquet cache file in data/raw/.
    """
    return PROJECT_ROOT / "data" / "raw" / f"prices_{zone}_{year}.parquet"


# ---------------------------------------------------------------------------
# Single-day fetch and parse
# ---------------------------------------------------------------------------

def _fetch_one_day(api_zone: str, query_date: date) -> pd.DataFrame:
    """Fetch hourly prices for one zone for one day.

    Args:
        api_zone: Zone in API format (e.g., "NO1").
        query_date: Date to fetch.

    Returns:
        DataFrame with 24 rows (one per hour), columns:
        eur_per_mwh, nok_per_kwh, exr.
        DatetimeIndex in Europe/Oslo timezone.
        Empty DataFrame if no data for this date.
    """
    url = f"{BASE_URL}/{query_date.year}/{query_date.strftime('%m-%d')}_{api_zone}.json"

    resp = requests.get(url, timeout=REQUEST_TIMEOUT)

    if resp.status_code == 404:
        # No data for this date (common for pre-Oct 2021)
        return pd.DataFrame()

    resp.raise_for_status()

    records = resp.json()
    if not records:
        return pd.DataFrame()

    rows = []
    for record in records:
        rows.append({
            "timestamp": pd.Timestamp(record["time_start"]),
            "eur_per_mwh": record["EUR_per_kWh"] * 1000,
            "nok_per_kwh": record["NOK_per_kWh"],
            "exr": record["EXR"],
        })

    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["timestamp"] = df["timestamp"].dt.tz_convert("Europe/Oslo")
    df = df.set_index("timestamp").sort_index()
    return df


# ---------------------------------------------------------------------------
# Yearly fetching with caching
# ---------------------------------------------------------------------------

def _fetch_zone_year(
    zone: str,
    year: int,
    start_date: date,
    end_date: date,
    cache: bool = True,
) -> pd.DataFrame:
    """Fetch all prices for one zone for one year, with caching.

    Args:
        zone: Zone in project format (e.g., "NO_1").
        year: Year to fetch.
        start_date: Earliest date to include (may be mid-year).
        end_date: Latest date to include (may be mid-year).
        cache: If True, cache completed years as Parquet.

    Returns:
        DataFrame with hourly prices for the zone-year.
    """
    today = date.today()
    is_completed_year = year < today.year
    cp = _cache_path(zone, year)

    # Load from cache if available (only for completed years)
    if cache and is_completed_year and cp.exists():
        logger.info("Loading cached prices for %s %d", zone, year)
        return pd.read_parquet(cp)

    # Determine date range for this year
    year_start = max(start_date, date(year, 1, 1))
    year_end = min(end_date, date(year, 12, 31))
    n_days = (year_end - year_start).days + 1

    api_zone = ZONES_PROJECT_TO_API[zone]

    logger.info(
        "Fetching prices for %s: %s to %s (%d days)",
        zone, year_start, year_end, n_days,
    )

    daily_frames = []
    current = year_start
    consecutive_errors = 0
    fetched_days = 0

    while current <= year_end:
        try:
            df_day = _fetch_one_day(api_zone, current)
            if not df_day.empty:
                daily_frames.append(df_day)
                consecutive_errors = 0
                fetched_days += 1
            # 404 (empty) is not an error — just missing data
        except requests.exceptions.HTTPError as e:
            consecutive_errors += 1
            if consecutive_errors <= 3:
                logger.warning("HTTP error for %s on %s: %s", zone, current, e)
            elif consecutive_errors == 4:
                logger.warning(
                    "Multiple consecutive errors for %s — continuing silently", zone
                )
            # Exponential backoff on repeated errors
            if consecutive_errors >= 3:
                backoff = min(5 * (2 ** (consecutive_errors - 3)), 60)
                time.sleep(backoff)
        except requests.exceptions.RequestException as e:
            consecutive_errors += 1
            if consecutive_errors <= 3:
                logger.warning("Request error for %s on %s: %s", zone, current, e)
            if consecutive_errors >= 3:
                backoff = min(5 * (2 ** (consecutive_errors - 3)), 60)
                time.sleep(backoff)

        current += timedelta(days=1)
        time.sleep(REQUEST_DELAY_SECONDS)

    if not daily_frames:
        logger.warning("No price data for %s in %d", zone, year)
        return pd.DataFrame()

    year_df = pd.concat(daily_frames).sort_index()
    year_df = year_df[~year_df.index.duplicated(keep="first")]

    logger.info(
        "Fetched %s %d: %d days, %d rows",
        zone, year, fetched_days, len(year_df),
    )

    # Cache completed years
    if cache and is_completed_year:
        cache_dir = cp.parent
        cache_dir.mkdir(parents=True, exist_ok=True)
        year_df.to_parquet(cp)
        logger.info("Cached prices for %s %d → %s", zone, year, cp.name)

    return year_df


# ---------------------------------------------------------------------------
# Public fetch functions
# ---------------------------------------------------------------------------

def fetch_prices(
    start_date: str = "2021-10-01",
    end_date: str = "2026-02-22",
    zones: list[str] | None = None,
    cache: bool = True,
) -> pd.DataFrame:
    """Fetch day-ahead prices for Norwegian bidding zones.

    Fetches each zone independently (one API call per zone per day).
    Returns hourly prices in EUR/MWh.

    Note: Continuous data available from October 2021 onward.
    Earlier dates may have gaps or return empty.

    Args:
        start_date: Start date as "YYYY-MM-DD" (default: "2021-10-01").
        end_date: End date as "YYYY-MM-DD".
        zones: List of zones in project format (e.g., ["NO_1", "NO_2"]).
               Defaults to all 5 Norwegian zones.
        cache: If True, cache completed years as Parquet.

    Returns:
        DataFrame with hourly DatetimeIndex (Europe/Oslo) and one column
        per zone (NO_1, NO_2, ...) with prices in EUR/MWh.
    """
    if zones is None:
        zones = ALL_ZONES

    # Validate zones
    for z in zones:
        if z not in ZONES_PROJECT_TO_API:
            raise ValueError(
                f"Unknown zone: {z}. Valid zones: {list(ZONES_PROJECT_TO_API.keys())}"
            )

    start = date.fromisoformat(start_date)
    end = date.fromisoformat(end_date)

    zone_series = {}

    for zone in zones:
        logger.info("Fetching prices for %s (%s to %s)", zone, start_date, end_date)

        year_chunks = []
        year = start.year
        while year <= end.year:
            chunk = _fetch_zone_year(zone, year, start, end, cache=cache)
            if not chunk.empty:
                year_chunks.append(chunk)
            year += 1

        if year_chunks:
            zone_df = pd.concat(year_chunks).sort_index()
            zone_df = zone_df[~zone_df.index.duplicated(keep="first")]
            zone_series[zone] = zone_df["eur_per_mwh"]
            logger.info(
                "Prices %s: %d rows, range %s to %s",
                zone, len(zone_df), zone_df.index.min(), zone_df.index.max(),
            )
        else:
            logger.warning("No price data for %s", zone)

    if not zone_series:
        return pd.DataFrame()

    result = pd.DataFrame(zone_series)
    result.index.name = "timestamp"
    result = result.sort_index()

    logger.info(
        "Total prices: %d rows x %d zones, %d missing values",
        len(result), len(result.columns), result.isna().sum().sum(),
    )
    return result


def fetch_zone_prices(
    zone: str,
    start_date: str = "2021-10-01",
    end_date: str = "2026-02-22",
    cache: bool = True,
) -> pd.DataFrame:
    """Fetch full price detail for a single zone.

    Unlike fetch_prices() which returns only EUR/MWh per zone,
    this returns the full per-zone data including NOK price and
    exchange rate.

    Args:
        zone: Bidding zone in project format (e.g., "NO_5").
        start_date: Start date as "YYYY-MM-DD".
        end_date: End date as "YYYY-MM-DD".
        cache: If True, cache completed years as Parquet.

    Returns:
        DataFrame with columns: eur_per_mwh, nok_per_kwh, exr.
        Hourly DatetimeIndex (Europe/Oslo).
    """
    if zone not in ZONES_PROJECT_TO_API:
        raise ValueError(
            f"Unknown zone: {zone}. Valid zones: {list(ZONES_PROJECT_TO_API.keys())}"
        )

    start = date.fromisoformat(start_date)
    end = date.fromisoformat(end_date)

    year_chunks = []
    year = start.year
    while year <= end.year:
        chunk = _fetch_zone_year(zone, year, start, end, cache=cache)
        if not chunk.empty:
            year_chunks.append(chunk)
        year += 1

    if not year_chunks:
        return pd.DataFrame()

    result = pd.concat(year_chunks).sort_index()
    result = result[~result.index.duplicated(keep="first")]
    result.index.name = "timestamp"
    return result


# ---------------------------------------------------------------------------
# Entry point for testing
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    print("Day-Ahead Price Fetcher (hvakosterstrommen.no) — Test Run")
    print("=" * 60)

    # Step 1: Fetch a single day to verify API works
    print("\n--- Step 1: Fetch single day (2025-01-15, NO1) ---")
    try:
        df_day = _fetch_one_day("NO1", date(2025, 1, 15))
        if not df_day.empty:
            print(f"  Shape: {df_day.shape}")
            print(f"  Columns: {list(df_day.columns)}")
            print(f"  Price range: {df_day['eur_per_mwh'].min():.2f} – "
                  f"{df_day['eur_per_mwh'].max():.2f} EUR/MWh")
            print(f"\n{df_day.head()}")
        else:
            print("  No data returned")
    except Exception as e:
        print(f"  Error: {e}")

    # Step 2: Fetch 1 week for all zones
    print("\n--- Step 2: Fetch 1 week, all zones ---")
    try:
        df_week = fetch_prices("2025-01-13", "2025-01-19", cache=False)
        if not df_week.empty:
            print(f"Shape: {df_week.shape}")
            print(f"Columns: {list(df_week.columns)}")
            print(f"Date range: {df_week.index.min()} to {df_week.index.max()}")
            print(f"\nMissing values:\n{df_week.isna().sum()}")
            print(f"\nSample (first 5 rows):\n{df_week.head()}")
            print(f"\nDescriptive stats:\n{df_week.describe()}")
        else:
            print("No data returned")
    except Exception as e:
        print(f"Error: {e}")

    print("\nDone!")
