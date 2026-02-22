"""
Fetch weather data from MET Norway Frost API.

Retrieves historical hourly observations (temperature, wind, precipitation)
for weather stations mapped to Norwegian electricity bidding zones (NO1–NO5).

API docs: https://frost.met.no/api.html
Reference: See docs/frost_api_docs.md for response structures and element IDs.

Authentication: HTTP Basic Auth with FROST_CLIENT_ID as username, empty password.
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

FROST_BASE_URL = "https://frost.met.no"
FROST_CLIENT_ID = os.getenv("FROST_CLIENT_ID")

# Weather stations mapped to bidding zones
# See docs/frost_api_docs.md for coordinates and details
STATIONS = {
    "NO_1": "SN18700",    # Oslo - Blindern
    "NO_2": "SN39040",    # Kristiansand - Kjevik
    "NO_3": "SN68860",    # Trondheim - Voll
    "NO_4": "SN90450",    # Tromsø
    "NO_5": "SN50540",    # Bergen - Florida
}

# Elements relevant for electricity price forecasting
ELEMENTS = [
    "air_temperature",
    "wind_speed",
    "sum(precipitation_amount PT1H)",
]

# Project root for data paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def _check_response(response: requests.Response) -> None:
    """Check Frost API response and raise descriptive errors.

    Args:
        response: The requests Response object from Frost API.

    Raises:
        requests.HTTPError: With descriptive message based on status code.

    Status codes:
        200 — OK
        400 — Invalid parameter value or malformed request
        401 — Unauthorized (check FROST_CLIENT_ID)
        404 — No data found for the query
        500 — Internal server error
    """
    if response.status_code == 200:
        return

    error_messages = {
        400: "Invalid parameter value or malformed request",
        401: "Unauthorized — check FROST_CLIENT_ID in .env",
        404: "No data found for the query",
        500: "Frost API internal server error",
    }

    msg = error_messages.get(
        response.status_code,
        f"Unexpected status code: {response.status_code}",
    )

    # Try to extract detail from response body
    try:
        body = response.json()
        detail = body.get("error", {}).get("message", "")
        if detail:
            msg = f"{msg}: {detail}"
    except (ValueError, KeyError):
        pass

    logger.error("Frost API error %d: %s", response.status_code, msg)
    response.raise_for_status()


def fetch_observations(
    station_id: str,
    elements: list[str],
    start_date: str,
    end_date: str,
    time_resolution: str = "PT1H",
) -> pd.DataFrame:
    """Fetch historical observations from a single Frost API station.

    Handles pagination automatically by following nextLink until all data
    is retrieved. Fetches in yearly chunks to avoid timeouts on large ranges.

    Args:
        station_id: Frost station ID (e.g., "SN18700" for Oslo-Blindern).
        elements: List of element IDs to fetch (e.g., ["air_temperature", "wind_speed"]).
        start_date: Start date as "YYYY-MM-DD".
        end_date: End date as "YYYY-MM-DD".
        time_resolution: Time resolution (default "PT1H" for hourly).

    Returns:
        DataFrame with columns: one per element.
        Index is timezone-aware datetime (Europe/Oslo) named "timestamp".
    """
    if not FROST_CLIENT_ID:
        raise RuntimeError("FROST_CLIENT_ID not set in .env")

    # Break large date ranges into yearly chunks to avoid timeouts
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    all_records: list[dict] = []

    chunk_start = start
    while chunk_start < end:
        chunk_end = min(
            pd.Timestamp(f"{chunk_start.year}-12-31"),
            end,
        )

        logger.info(
            "Fetching %s from %s to %s",
            station_id,
            chunk_start.strftime("%Y-%m-%d"),
            chunk_end.strftime("%Y-%m-%d"),
        )

        records = _fetch_observations_chunk(
            station_id=station_id,
            elements=elements,
            start_date=chunk_start.strftime("%Y-%m-%d"),
            end_date=chunk_end.strftime("%Y-%m-%d"),
            time_resolution=time_resolution,
        )
        all_records.extend(records)

        # Move to next year
        chunk_start = pd.Timestamp(f"{chunk_start.year + 1}-01-01")

        # Be polite to the API
        if chunk_start < end:
            time.sleep(1)

    if not all_records:
        logger.warning("No observations returned for %s", station_id)
        return pd.DataFrame()

    return _records_to_dataframe(all_records)


def _fetch_observations_chunk(
    station_id: str,
    elements: list[str],
    start_date: str,
    end_date: str,
    time_resolution: str,
) -> list[dict]:
    """Fetch a single date-range chunk, handling pagination."""
    endpoint = f"{FROST_BASE_URL}/observations/v0.jsonld"
    params = {
        "sources": station_id,
        "elements": ",".join(elements),
        "referencetime": f"{start_date}/{end_date}",
        "timeresolutions": time_resolution,
    }

    records: list[dict] = []
    url = endpoint

    while True:
        response = requests.get(url, params=params, auth=(FROST_CLIENT_ID, ""))
        _check_response(response)

        data = response.json()
        records.extend(data.get("data", []))

        # Check for pagination
        next_link = None
        for link in data.get("links", []):
            if link.get("rel") == "next":
                next_link = link.get("href")
                break

        if not next_link:
            break

        # Follow the next link (it's a full URL with params baked in)
        url = next_link
        params = {}  # params are in the URL now

    logger.info("Got %d observation records for chunk", len(records))
    return records


def _records_to_dataframe(records: list[dict]) -> pd.DataFrame:
    """Convert Frost API observation records to a clean DataFrame.

    Args:
        records: List of observation dicts from Frost API.

    Returns:
        DataFrame indexed by timezone-aware timestamp (Europe/Oslo),
        with one column per element.
    """
    rows = []
    for record in records:
        ref_time = record["referenceTime"]
        row = {"timestamp": ref_time}
        for obs in record.get("observations", []):
            element_id = obs["elementId"]
            # Rename the long precipitation element for convenience
            if element_id == "sum(precipitation_amount PT1H)":
                element_id = "precipitation"
            row[element_id] = obs["value"]
        rows.append(row)

    df = pd.DataFrame(rows)

    if df.empty:
        return df

    # Parse timestamps and set timezone
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["timestamp"] = df["timestamp"].dt.tz_convert("Europe/Oslo")
    df = df.set_index("timestamp")

    # Sort and remove duplicates (can happen at chunk boundaries)
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="first")]

    return df


def fetch_zone_weather(
    zone: str,
    start_date: str,
    end_date: str,
    cache: bool = True,
) -> pd.DataFrame:
    """Fetch weather data for a bidding zone, with local caching.

    Args:
        zone: Bidding zone (e.g., "NO_5").
        start_date: Start date as "YYYY-MM-DD".
        end_date: End date as "YYYY-MM-DD".
        cache: If True, check data/raw/ before fetching and save after.

    Returns:
        DataFrame with hourly weather observations for the zone.

    Raises:
        ValueError: If zone is not in STATIONS.
    """
    if zone not in STATIONS:
        raise ValueError(
            f"Unknown zone '{zone}'. Valid zones: {list(STATIONS.keys())}"
        )

    cache_dir = PROJECT_ROOT / "data" / "raw"
    cache_path = cache_dir / f"weather_{zone}_{start_date}_{end_date}.parquet"

    if cache and cache_path.exists():
        logger.info("Loading cached weather data from %s", cache_path)
        return pd.read_parquet(cache_path)

    station_id = STATIONS[zone]
    df = fetch_observations(
        station_id=station_id,
        elements=ELEMENTS,
        start_date=start_date,
        end_date=end_date,
    )

    if cache and not df.empty:
        cache_dir.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache_path)
        logger.info("Cached weather data to %s", cache_path)

    return df


def fetch_all_zones(
    start_date: str,
    end_date: str,
) -> dict[str, pd.DataFrame]:
    """Fetch weather data for all 5 Norwegian bidding zones.

    Args:
        start_date: Start date as "YYYY-MM-DD".
        end_date: End date as "YYYY-MM-DD".

    Returns:
        Dict mapping zone names to DataFrames (e.g., {"NO_1": df, ...}).
    """
    results = {}
    for zone in STATIONS:
        logger.info("Fetching weather for zone %s", zone)
        results[zone] = fetch_zone_weather(zone, start_date, end_date)
        time.sleep(2)  # Be polite to the API between zones

    return results


# --- Entry point for testing ---

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    print(f"FROST_CLIENT_ID configured: {bool(FROST_CLIENT_ID)}")
    print(f"Stations: {list(STATIONS.keys())}")
    print(f"Elements: {ELEMENTS}")

    # Test: fetch Bergen (NO_5) data from 2020 to current
    logger.info("Testing: Fetching Bergen (NO_5) weather 2020-01-01 to 2026-02-22")
    df = fetch_zone_weather("NO_5", "2020-01-01", "2026-02-22")

    if not df.empty:
        print(f"\nShape: {df.shape}")
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
