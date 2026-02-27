"""
Fetch electricity data from ENTSO-E Transparency Platform.

Retrieves day-ahead prices, actual load, generation per type, and reservoir
filling for Norwegian bidding zones (NO1–NO5) using the entsoe-py wrapper.

API docs: See docs/entsoe_api_reference.md for full reference.
Authentication: API token via ENTSOE_API_KEY in .env.

Registration:
    1. Create account at https://transparency.entsoe.eu/
    2. Email transparency@entsoe.eu with subject "Restful API access"
    3. Wait 1–3 business days
    4. Generate token: My Account Settings → Web API Security Token
    5. Add to .env: ENTSOE_API_KEY=your-token-here
"""

import os
import time
import logging
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

# --- Configuration ---

ENTSOE_API_KEY = os.getenv("ENTSOE_API_KEY")

# Norwegian bidding zones (entsoe-py keys)
ZONES = {
    "NO_1": "Øst-Norge (Oslo)",
    "NO_2": "Sør-Norge (Kristiansand)",
    "NO_3": "Midt-Norge (Trondheim)",
    "NO_4": "Nord-Norge (Tromsø)",
    "NO_5": "Vest-Norge (Bergen)",
}

# Project root for data paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def _get_client():
    """Create and return an EntsoePandasClient.

    Raises:
        RuntimeError: If ENTSOE_API_KEY is not set in .env.
    """
    if not ENTSOE_API_KEY:
        raise RuntimeError(
            "ENTSOE_API_KEY not set in .env.\n"
            "To get an API key:\n"
            "  1. Register at https://transparency.entsoe.eu/\n"
            "  2. Email transparency@entsoe.eu with subject 'Restful API access'\n"
            "  3. Wait 1–3 business days\n"
            "  4. Generate token: My Account Settings → Web API Security Token\n"
            "  5. Add to .env: ENTSOE_API_KEY=your-token-here"
        )
    from entsoe import EntsoePandasClient

    return EntsoePandasClient(api_key=ENTSOE_API_KEY)


def _make_timestamps(start_date: str, end_date: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Convert date strings to timezone-aware Timestamps.

    Args:
        start_date: Start date as "YYYY-MM-DD".
        end_date: End date as "YYYY-MM-DD".

    Returns:
        Tuple of (start, end) as tz-aware pd.Timestamps (Europe/Oslo).
    """
    return (
        pd.Timestamp(start_date, tz="Europe/Oslo"),
        pd.Timestamp(end_date, tz="Europe/Oslo"),
    )


def _cache_path(label: str, zone: str, year: int) -> Path:
    """Build a standardized cache file path.

    Args:
        label: Data type label (e.g., "prices", "load", "generation").
        zone: Bidding zone (e.g., "NO_5").
        year: Year of data.

    Returns:
        Path to the Parquet cache file.
    """
    return PROJECT_ROOT / "data" / "raw" / f"entsoe_{label}_{zone}_{year}.parquet"


def _fetch_yearly_chunks(
    query_func,
    zone: str,
    start_date: str,
    end_date: str,
    label: str,
    cache: bool = True,
    **kwargs,
) -> pd.DataFrame:
    """Generic fetcher that splits requests into yearly chunks with caching.

    Args:
        query_func: The entsoe-py client method to call (e.g., client.query_day_ahead_prices).
        zone: Bidding zone (e.g., "NO_5").
        start_date: Start date as "YYYY-MM-DD".
        end_date: End date as "YYYY-MM-DD".
        label: Data type label for cache file naming.
        cache: If True, check/save cache files per year.
        **kwargs: Additional keyword arguments passed to query_func.

    Returns:
        Combined DataFrame/Series for the full date range.
    """
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    cache_dir = PROJECT_ROOT / "data" / "raw"
    cache_dir.mkdir(parents=True, exist_ok=True)

    all_chunks = []
    current_year = pd.Timestamp.now().year

    year = start.year
    while year <= end.year:
        chunk_start_str = f"{year}-01-01" if year > start.year else start_date
        chunk_end_str = f"{year}-12-31" if year < end.year else end_date
        is_current_year = (year == current_year)

        cp = _cache_path(label, zone, year)

        # Use cache for completed years only; always re-fetch the current year
        if cache and cp.exists() and not is_current_year:
            logger.info("Loading cached %s for %s %d", label, zone, year)
            chunk = pd.read_parquet(cp)
            all_chunks.append(chunk)
            year += 1
            continue

        ts_start, ts_end = _make_timestamps(chunk_start_str, chunk_end_str)
        logger.info("Fetching %s for %s: %s to %s", label, zone, chunk_start_str, chunk_end_str)

        try:
            result = query_func(zone, start=ts_start, end=ts_end, **kwargs)
        except Exception as e:
            logger.error("Failed to fetch %s for %s %d: %s", label, zone, year, e)
            year += 1
            continue

        # Normalize to DataFrame for consistent caching
        if isinstance(result, pd.Series):
            chunk = result.to_frame(label)
        else:
            chunk = result

        if cache and not chunk.empty:
            chunk.to_parquet(cp)
            logger.info("Cached %s for %s %d → %s", label, zone, year, cp.name)

        all_chunks.append(chunk)
        year += 1

        # Be polite to the API
        time.sleep(2)

    if not all_chunks:
        return pd.DataFrame()

    combined = pd.concat(all_chunks)
    combined = combined.sort_index()
    combined = combined[~combined.index.duplicated(keep="first")]
    return combined


# --- Public fetch functions ---


def fetch_prices(
    zone: str,
    start_date: str,
    end_date: str,
    cache: bool = True,
) -> pd.DataFrame:
    """Fetch day-ahead electricity prices for a bidding zone.

    Args:
        zone: Bidding zone (e.g., "NO_5" for Bergen).
        start_date: Start date as "YYYY-MM-DD".
        end_date: End date as "YYYY-MM-DD".
        cache: If True, use yearly Parquet caching.

    Returns:
        DataFrame with column "prices" (EUR/MWh), hourly datetime index (Europe/Oslo).
    """
    client = _get_client()
    return _fetch_yearly_chunks(
        client.query_day_ahead_prices,
        zone, start_date, end_date,
        label="prices",
        cache=cache,
    )


def fetch_load(
    zone: str,
    start_date: str,
    end_date: str,
    cache: bool = True,
) -> pd.DataFrame:
    """Fetch actual total load (consumption) for a bidding zone.

    Args:
        zone: Bidding zone (e.g., "NO_5").
        start_date: Start date as "YYYY-MM-DD".
        end_date: End date as "YYYY-MM-DD".
        cache: If True, use yearly Parquet caching.

    Returns:
        DataFrame with load columns (MW), hourly datetime index.
    """
    client = _get_client()
    return _fetch_yearly_chunks(
        client.query_load,
        zone, start_date, end_date,
        label="load",
        cache=cache,
    )


def _flatten_generation_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten MultiIndex columns from entsoe-py generation query.

    entsoe-py may return a MultiIndex with levels like
    ('Actual Aggregated', 'Hydro Water Reservoir'). This extracts only the
    'Actual Aggregated' rows (production) and flattens to single-level
    column names like 'Hydro Water Reservoir'.

    If columns are already a flat Index, returns unchanged.

    Args:
        df: DataFrame from client.query_generation().

    Returns:
        DataFrame with flat string column names (generation type names).
    """
    if df.empty:
        return df

    if isinstance(df.columns, pd.MultiIndex):
        # Keep only 'Actual Aggregated' (production), drop 'Actual Consumption'
        if "Actual Aggregated" in df.columns.get_level_values(0):
            df = df["Actual Aggregated"]
        else:
            # Fallback: just drop the first level
            df.columns = df.columns.droplevel(0)
        logger.info("Flattened MultiIndex generation columns → %d types", len(df.columns))

    return df


def fetch_generation(
    zone: str,
    start_date: str,
    end_date: str,
    cache: bool = True,
) -> pd.DataFrame:
    """Fetch actual generation per type for a bidding zone.

    Returns columns for each production type (Hydro Water Reservoir,
    Wind Onshore, etc.) in MW. Handles MultiIndex columns from entsoe-py
    by flattening to single-level type names.

    Args:
        zone: Bidding zone (e.g., "NO_5").
        start_date: Start date as "YYYY-MM-DD".
        end_date: End date as "YYYY-MM-DD".
        cache: If True, use yearly Parquet caching.

    Returns:
        DataFrame with one column per generation type (MW), hourly datetime index.
    """
    client = _get_client()
    result = _fetch_yearly_chunks(
        client.query_generation,
        zone, start_date, end_date,
        label="generation",
        cache=cache,
        psr_type=None,
    )
    return _flatten_generation_columns(result)


def fetch_reservoir_filling(
    start_date: str,
    end_date: str,
    cache: bool = True,
) -> pd.DataFrame:
    """Fetch reservoir filling levels for all of Norway.

    Note: ENTSO-E only provides reservoir data for all of Norway ("NO"),
    not per bidding zone. Resolution is weekly.

    Args:
        start_date: Start date as "YYYY-MM-DD".
        end_date: End date as "YYYY-MM-DD".
        cache: If True, use yearly Parquet caching.

    Returns:
        DataFrame with column "reservoir_filling" (%), weekly datetime index.
    """
    client = _get_client()
    return _fetch_yearly_chunks(
        client.query_aggregate_water_reservoirs_and_hydro_storage,
        "NO", start_date, end_date,
        label="reservoir_filling",
        cache=cache,
    )


def fetch_crossborder_flows(
    zone_from: str,
    zone_to: str,
    start_date: str,
    end_date: str,
    cache: bool = True,
) -> pd.DataFrame:
    """Fetch physical cross-border flows between two zones.

    Args:
        zone_from: Origin zone (e.g., "NO_5").
        zone_to: Destination zone (e.g., "NO_1").
        start_date: Start date as "YYYY-MM-DD".
        end_date: End date as "YYYY-MM-DD".
        cache: If True, use yearly Parquet caching.

    Returns:
        DataFrame with flow column (MW), hourly datetime index.
    """
    client = _get_client()
    label = f"flow_{zone_from}_{zone_to}"

    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    cache_dir = PROJECT_ROOT / "data" / "raw"
    cache_dir.mkdir(parents=True, exist_ok=True)

    all_chunks = []
    current_year = pd.Timestamp.now().year
    year = start.year

    while year <= end.year:
        chunk_start = f"{year}-01-01" if year > start.year else start_date
        chunk_end = f"{year}-12-31" if year < end.year else end_date
        is_current_year = (year == current_year)
        cp = cache_dir / f"entsoe_{label}_{year}.parquet"

        # Use cache for completed years only; always re-fetch the current year
        if cache and cp.exists() and not is_current_year:
            logger.info("Loading cached %s %d", label, year)
            all_chunks.append(pd.read_parquet(cp))
            year += 1
            continue

        ts_start, ts_end = _make_timestamps(chunk_start, chunk_end)
        logger.info("Fetching %s: %s to %s", label, chunk_start, chunk_end)

        try:
            result = client.query_crossborder_flows(zone_from, zone_to, start=ts_start, end=ts_end)
            if isinstance(result, pd.Series):
                chunk = result.to_frame(label)
            else:
                chunk = result

            if cache and not chunk.empty:
                chunk.to_parquet(cp)

            all_chunks.append(chunk)
        except Exception as e:
            logger.error("Failed to fetch %s %d: %s", label, year, e)

        year += 1
        time.sleep(2)

    if not all_chunks:
        return pd.DataFrame()

    combined = pd.concat(all_chunks).sort_index()
    combined = combined[~combined.index.duplicated(keep="first")]
    return combined


# Cable connections per Norwegian zone → foreign zone(s)
ZONE_CABLES = {
    "NO_1": ["SE_3"],
    "NO_2": ["DK_1", "NL", "DE_LU", "GB"],
    "NO_3": ["SE_2"],
    "NO_4": ["SE_1", "SE_2", "FI"],
    "NO_5": [],  # No international cables
}

# Internal connections between Norwegian bidding zones
ZONE_INTERNAL = {
    "NO_1": ["NO_2", "NO_3", "NO_5"],
    "NO_2": ["NO_1", "NO_5"],
    "NO_3": ["NO_1", "NO_4", "NO_5"],
    "NO_4": ["NO_3"],
    "NO_5": ["NO_1", "NO_2", "NO_3"],
}

# Foreign zones for which day-ahead prices are available on ENTSO-E.
# GB is excluded: left EU transparency platform, prices return NoMatchingDataError.
FOREIGN_PRICE_ZONES = ["DK_1", "NL", "DE_LU", "SE_1", "SE_2", "SE_3", "SE_4", "FI"]


def fetch_foreign_prices(
    foreign_zone: str,
    start_date: str,
    end_date: str,
    cache: bool = True,
) -> pd.DataFrame:
    """Fetch day-ahead prices for a foreign (non-Norwegian) zone.

    Used to compare Norwegian zone prices with cable endpoint prices
    for spread and arbitrage analysis.

    Args:
        foreign_zone: Foreign zone code (e.g., "DK_1", "SE_3", "NL").
        start_date: Start date as "YYYY-MM-DD".
        end_date: End date as "YYYY-MM-DD".
        cache: If True, use yearly Parquet caching.

    Returns:
        DataFrame with column "prices" (EUR/MWh), hourly datetime index.
    """
    client = _get_client()
    return _fetch_yearly_chunks(
        client.query_day_ahead_prices,
        foreign_zone, start_date, end_date,
        label=f"prices_{foreign_zone}",
        cache=cache,
    )


def fetch_all_entsoe(
    zone: str,
    start_date: str,
    end_date: str,
    cache: bool = True,
) -> dict[str, pd.DataFrame]:
    """Fetch all key ENTSO-E data for a zone.

    Fetches prices, load, and generation in one call. Reservoir filling
    is fetched for all of Norway (only available at national level).

    Args:
        zone: Bidding zone (e.g., "NO_5").
        start_date: Start date as "YYYY-MM-DD".
        end_date: End date as "YYYY-MM-DD".
        cache: If True, use yearly Parquet caching.

    Returns:
        Dict with keys: "prices", "load", "generation", "reservoir_filling".
    """
    results = {}

    logger.info("Fetching all ENTSO-E data for %s (%s to %s)", zone, start_date, end_date)

    for label, func in [
        ("prices", lambda: fetch_prices(zone, start_date, end_date, cache=cache)),
        ("load", lambda: fetch_load(zone, start_date, end_date, cache=cache)),
        ("generation", lambda: fetch_generation(zone, start_date, end_date, cache=cache)),
        ("reservoir_filling", lambda: fetch_reservoir_filling(start_date, end_date, cache=cache)),
    ]:
        try:
            results[label] = func()
            logger.info("✓ %s: %d rows", label, len(results[label]))
        except Exception as e:
            logger.error("✗ %s failed: %s", label, e)
            results[label] = pd.DataFrame()

    return results


# --- Entry point for testing ---

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    # Check API key status
    if ENTSOE_API_KEY:
        print(f"ENTSOE_API_KEY configured: True (length: {len(ENTSOE_API_KEY)})")
    else:
        print("ENTSOE_API_KEY not configured.")
        print()
        print("To set up:")
        print("  1. Register at https://transparency.entsoe.eu/")
        print("  2. Email transparency@entsoe.eu with subject 'Restful API access'")
        print("  3. Wait 1–3 business days for approval")
        print("  4. Go to My Account Settings → Web API Security Token → Generate")
        print("  5. Add to .env: ENTSOE_API_KEY=your-token-here")
        print()
        print("Once you have the key, run this script again to test.")
        exit(0)

    print(f"Zones: {list(ZONES.keys())}")
    print()

    # Test: fetch Bergen (NO_5) data for a small range first
    zone = "NO_5"
    start = "2024-01-01"
    end = "2024-02-01"

    print(f"=== Fetching all ENTSO-E data for {zone} ({start} to {end}) ===\n")
    data = fetch_all_entsoe(zone, start, end)

    for label, df in data.items():
        print(f"\n--- {label} ---")
        if df.empty:
            print("  No data returned")
            continue
        print(f"  Shape: {df.shape}")
        print(f"  Date range: {df.index.min()} to {df.index.max()}")
        print(f"  Columns: {list(df.columns)}")
        print(f"  Missing values: {df.isna().sum().sum()}")
        print(f"  First 3 rows:")
        print(df.head(3).to_string(max_cols=6))

    # Test foreign prices
    print("\n=== Testing foreign prices ===")
    for fz in FOREIGN_PRICE_ZONES[:3]:  # Test first 3
        try:
            fp = fetch_foreign_prices(fz, start, end, cache=True)
            print(f"  {fz}: {fp.shape[0]} rows, mean={fp.iloc[:, 0].mean():.1f} EUR/MWh")
        except Exception as e:
            print(f"  {fz}: ERROR - {e}")

    # Test crossborder flows
    print("\n=== Testing crossborder flows (NO_2 cables) ===")
    for cable_to in ZONE_CABLES["NO_2"]:
        try:
            flow = fetch_crossborder_flows("NO_2", cable_to, start, end, cache=True)
            col = flow.columns[0] if not flow.empty else "?"
            print(f"  NO_2→{cable_to}: {flow.shape[0]} rows, mean={flow.iloc[:, 0].mean():.1f} MW")
        except Exception as e:
            print(f"  NO_2→{cable_to}: ERROR - {e}")
