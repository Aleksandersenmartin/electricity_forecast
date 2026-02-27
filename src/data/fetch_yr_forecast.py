"""
Fetch weather forecasts from Yr Locationforecast 2.0 (MET Norway).

Retrieves hourly weather forecasts (~9 days ahead) for Norwegian bidding zone
representative locations. Used for forward-looking price predictions where
historical observations (Frost API) are not available.

API docs: https://api.met.no/weatherapi/locationforecast/2.0/documentation
No API key required — only a User-Agent header with app name + contact.

Columns are prefixed with ``yr_`` to distinguish from historical weather
observations (``temperature``, ``wind_speed``) in the feature matrix.
"""

import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# --- Configuration ---

YR_BASE_URL = "https://api.met.no/weatherapi/locationforecast/2.0/compact"

# User-Agent required by MET Norway ToS (app name + contact)
YR_USER_AGENT = "electricity-forecast github.com/Aleksandersenmartin"

# Representative coordinates per bidding zone (same locations as Frost stations)
ZONE_COORDINATES: dict[str, dict[str, float]] = {
    "NO_1": {"lat": 59.9423, "lon": 10.7200, "name": "Oslo (Blindern)"},
    "NO_2": {"lat": 58.2000, "lon": 8.0800, "name": "Kristiansand (Kjevik)"},
    "NO_3": {"lat": 63.4107, "lon": 10.4538, "name": "Trondheim (Voll)"},
    "NO_4": {"lat": 69.6489, "lon": 18.9551, "name": "Tromsø"},
    "NO_5": {"lat": 60.3830, "lon": 5.3327, "name": "Bergen (Florida)"},
}

# Mapping from Yr JSON field names to our column names
FIELD_MAP = {
    "air_temperature": "yr_temperature",
    "wind_speed": "yr_wind_speed",
    "wind_from_direction": "yr_wind_direction",
    "relative_humidity": "yr_humidity",
    "air_pressure_at_sea_level": "yr_pressure",
    "cloud_area_fraction": "yr_cloud_cover",
    "wind_speed_of_gust": "yr_wind_gust",
}


def _parse_timeseries(data: dict) -> pd.DataFrame:
    """Parse Yr Locationforecast JSON timeseries into a DataFrame.

    Args:
        data: Parsed JSON response from Locationforecast API.

    Returns:
        DataFrame with hourly DatetimeIndex (UTC) and yr_* columns.
    """
    timeseries = data.get("properties", {}).get("timeseries", [])
    if not timeseries:
        logger.warning("Empty timeseries in Yr response")
        return pd.DataFrame()

    rows = []
    for entry in timeseries:
        ts = pd.Timestamp(entry["time"])
        details = entry.get("data", {}).get("instant", {}).get("details", {})

        row: dict[str, float | pd.Timestamp] = {"time": ts}
        for yr_key, col_name in FIELD_MAP.items():
            row[col_name] = details.get(yr_key)

        # Precipitation from next_1_hours (not in instant details)
        next_1h = entry.get("data", {}).get("next_1_hours", {})
        precip = next_1h.get("details", {}).get("precipitation_amount")
        row["yr_precipitation_1h"] = precip

        rows.append(row)

    df = pd.DataFrame(rows)
    df = df.set_index("time")
    df.index = pd.DatetimeIndex(df.index, name="time")

    # Ensure UTC then convert to Europe/Oslo
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    df.index = df.index.tz_convert("Europe/Oslo")

    return df


def _resample_to_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """Resample forecast DataFrame to consistent hourly resolution.

    Yr provides 1-hour steps for the first ~48h, then 6-hour steps.
    This forward-fills the 6-hour gaps to produce hourly data.

    Args:
        df: DataFrame with irregular DatetimeIndex.

    Returns:
        DataFrame resampled to hourly frequency via forward-fill.
    """
    if df.empty:
        return df

    hourly_index = pd.date_range(
        start=df.index.min(),
        end=df.index.max(),
        freq="h",
        tz=df.index.tz,
    )
    df = df.reindex(hourly_index)
    df = df.ffill()
    df.index.name = "time"
    return df


def fetch_yr_forecast(zone: str, cache: bool = True) -> pd.DataFrame:
    """Fetch Yr Locationforecast for a Norwegian bidding zone.

    Retrieves ~9 days of hourly weather forecast data from MET Norway's
    Locationforecast 2.0 API for the representative location of the zone.

    Args:
        zone: Bidding zone identifier (e.g. "NO_5").
        cache: If True, cache result to data/raw/ and return cached
            data if less than 1 hour old.

    Returns:
        DataFrame with hourly DatetimeIndex (Europe/Oslo) and columns:
        yr_temperature, yr_wind_speed, yr_precipitation_1h,
        yr_cloud_cover, yr_humidity, yr_pressure, yr_wind_gust,
        yr_wind_direction.
    """
    if zone not in ZONE_COORDINATES:
        raise ValueError(
            f"Unknown zone '{zone}'. Must be one of: {list(ZONE_COORDINATES.keys())}"
        )

    coords = ZONE_COORDINATES[zone]
    cache_dir = PROJECT_ROOT / "data" / "raw"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"yr_forecast_{zone}.parquet"

    # Check cache freshness (Yr updates every ~1 hour)
    if cache and cache_path.exists():
        cache_age_seconds = time.time() - cache_path.stat().st_mtime
        if cache_age_seconds < 3600:  # Less than 1 hour old
            logger.info(
                "Using cached Yr forecast for %s (%s) — %.0f min old",
                zone, coords["name"], cache_age_seconds / 60,
            )
            return pd.read_parquet(cache_path)

    # Fetch from API
    params = {
        "lat": round(coords["lat"], 4),
        "lon": round(coords["lon"], 4),
    }
    headers = {
        "User-Agent": YR_USER_AGENT,
        "Accept": "application/json",
    }

    logger.info(
        "Fetching Yr forecast for %s (%s) at (%.4f, %.4f)",
        zone, coords["name"], coords["lat"], coords["lon"],
    )

    try:
        response = requests.get(
            YR_BASE_URL, params=params, headers=headers, timeout=30
        )
        response.raise_for_status()
    except requests.RequestException as e:
        logger.error("Yr API request failed for %s: %s", zone, e)
        # Return cached data even if stale, if available
        if cache_path.exists():
            logger.warning("Returning stale cache for %s", zone)
            return pd.read_parquet(cache_path)
        return pd.DataFrame()

    data = response.json()

    # Log API metadata
    expires = response.headers.get("Expires", "unknown")
    last_modified = response.headers.get("Last-Modified", "unknown")
    logger.info("Yr response: Expires=%s, Last-Modified=%s", expires, last_modified)

    # Parse and resample
    df = _parse_timeseries(data)
    if df.empty:
        logger.warning("No data parsed from Yr response for %s", zone)
        return df

    df = _resample_to_hourly(df)

    logger.info(
        "Yr forecast for %s: %d hours, %s to %s",
        zone, len(df), df.index.min(), df.index.max(),
    )

    # Cache
    if cache:
        df.to_parquet(cache_path)
        logger.info("Cached Yr forecast to %s", cache_path)

    return df


def fetch_all_yr_forecasts(cache: bool = True) -> dict[str, pd.DataFrame]:
    """Fetch Yr forecasts for all 5 Norwegian bidding zones.

    Args:
        cache: If True, cache each zone's result.

    Returns:
        Dict mapping zone name to forecast DataFrame.
    """
    results = {}
    for zone in ZONE_COORDINATES:
        results[zone] = fetch_yr_forecast(zone, cache=cache)
        time.sleep(0.5)  # Be polite to MET Norway API
    return results


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    # Quick test: fetch Bergen forecast
    df = fetch_yr_forecast("NO_5", cache=True)
    if not df.empty:
        print(f"\nYr forecast for NO_5 (Bergen):")
        print(f"  Rows: {len(df)}")
        print(f"  Columns: {list(df.columns)}")
        print(f"  Range: {df.index.min()} to {df.index.max()}")
        print(f"\nFirst 5 rows:")
        print(df.head())
        print(f"\nLast 5 rows:")
        print(df.tail())
    else:
        print("No data returned.")
