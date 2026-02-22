"""
Fetch reservoir filling data from NVE Magasinstatistikk API.

Retrieves weekly hydro reservoir filling levels per Norwegian bidding zone
(NO1–NO5) from NVE (Norges vassdrags- og energidirektorat).

This is the authoritative source for per-zone reservoir data. ENTSO-E only
has national-level data; NVE provides zone-level with richer metadata.

API docs: See docs/nve_magasin_api_reference.md
Swagger: https://biapi.nve.no/magasinstatistikk/swagger/index.html

Authentication: None required. Open data (NLOD license).
"""

import logging
from pathlib import Path

import requests
import pandas as pd

logger = logging.getLogger(__name__)

# --- Configuration ---

NVE_BASE_URL = "https://biapi.nve.no/magasinstatistikk/api/Magasinstatistikk"

# Map NVE omrnr to project zone names
ZONE_MAP = {
    1: "NO_1",  # Sørøst-Norge
    2: "NO_2",  # Sørvest-Norge
    3: "NO_3",  # Midt-Norge
    4: "NO_4",  # Nord-Norge
    5: "NO_5",  # Vest-Norge
}

# Project root for data paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def _check_response(response: requests.Response) -> None:
    """Check NVE API response and raise descriptive errors.

    Args:
        response: The requests Response object.

    Raises:
        requests.HTTPError: On HTTP errors.
    """
    if response.status_code == 200:
        return

    logger.error("NVE API error %d: %s", response.status_code, response.text[:200])
    response.raise_for_status()


def _parse_to_dataframe(data: list[dict], elspot_only: bool = True) -> pd.DataFrame:
    """Parse NVE JSON response into a clean DataFrame.

    Args:
        data: List of dicts from NVE API response.
        elspot_only: If True, filter to omrType == "EL" (elspot zones only).

    Returns:
        DataFrame with parsed dates, zone names, and clean column names.
    """
    df = pd.DataFrame(data)

    if df.empty:
        return df

    if elspot_only:
        df = df[df["omrType"] == "EL"].copy()

    # Map zone numbers to zone names
    df["zone"] = df["omrnr"].map(ZONE_MAP)

    # Parse date
    df["date"] = pd.to_datetime(df["dato_Id"])
    df["date"] = df["date"].dt.tz_localize("Europe/Oslo")

    # Select and rename columns for clarity
    df = df.rename(columns={
        "fyllingsgrad": "filling_pct",
        "kapasitet_TWh": "capacity_twh",
        "fylling_TWh": "filling_twh",
        "fyllingsgrad_forrige_uke": "filling_pct_prev_week",
        "endring_fyllingsgrad": "filling_change",
        "iso_aar": "year",
        "iso_uke": "week",
    })

    columns = [
        "date", "zone", "year", "week",
        "filling_pct", "filling_twh", "capacity_twh",
        "filling_pct_prev_week", "filling_change",
    ]
    df = df[[c for c in columns if c in df.columns]]
    df = df.set_index("date").sort_index()

    return df


def fetch_reservoir_all(cache: bool = True) -> pd.DataFrame:
    """Fetch all historical reservoir data from NVE (since 1995).

    Returns weekly filling data for all 5 elspot zones. This is a single
    API call that returns ~8000 rows — cache aggressively.

    Args:
        cache: If True, check data/raw/ before fetching and save after.

    Returns:
        DataFrame indexed by date (Europe/Oslo), with columns:
        zone, year, week, filling_pct, filling_twh, capacity_twh,
        filling_pct_prev_week, filling_change.
    """
    cache_dir = PROJECT_ROOT / "data" / "raw"
    cache_path = cache_dir / "reservoir_nve_all.parquet"

    if cache and cache_path.exists():
        logger.info("Loading cached reservoir data from %s", cache_path)
        return pd.read_parquet(cache_path)

    logger.info("Fetching all NVE reservoir data (since 1995)")
    response = requests.get(f"{NVE_BASE_URL}/HentOffentligData")
    _check_response(response)

    df = _parse_to_dataframe(response.json())

    if cache and not df.empty:
        cache_dir.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache_path)
        logger.info("Cached reservoir data to %s (%d rows)", cache_path, len(df))

    return df


def fetch_reservoir_latest() -> pd.DataFrame:
    """Fetch latest week reservoir data from NVE.

    Lightweight call — returns only the most recent week for all zones.
    Useful for dashboard updates. Not cached (always fresh).

    Returns:
        DataFrame with one row per zone for the latest week.
    """
    logger.info("Fetching latest NVE reservoir data")
    response = requests.get(f"{NVE_BASE_URL}/HentOffentligDataSisteUke")
    _check_response(response)

    return _parse_to_dataframe(response.json())


def fetch_reservoir_benchmarks(cache: bool = True) -> pd.DataFrame:
    """Fetch min/max/median reservoir benchmarks from NVE.

    Returns historical (20-year) min, max, and median filling per week
    per zone. Useful for computing deviation-from-normal features.

    Args:
        cache: If True, check data/raw/ before fetching and save after.

    Returns:
        DataFrame with columns: zone, week, min_filling_pct, median_filling_pct,
        max_filling_pct, and TWh equivalents.
    """
    cache_dir = PROJECT_ROOT / "data" / "raw"
    cache_path = cache_dir / "reservoir_nve_benchmarks.parquet"

    if cache and cache_path.exists():
        logger.info("Loading cached benchmark data from %s", cache_path)
        return pd.read_parquet(cache_path)

    logger.info("Fetching NVE reservoir benchmarks (20-year min/max/median)")
    response = requests.get(f"{NVE_BASE_URL}/HentOffentligDataMinMaxMedian")
    _check_response(response)

    df = pd.DataFrame(response.json())

    if df.empty:
        return df

    # Filter to elspot zones
    df = df[df["omrType"] == "EL"].copy()
    df["zone"] = df["omrnr"].map(ZONE_MAP)

    df = df.rename(columns={
        "iso_uke": "week",
        "minFyllingsgrad": "min_filling_pct",
        "minFyllingTWH": "min_filling_twh",
        "medianFyllingsGrad": "median_filling_pct",
        "medianFylling_TWH": "median_filling_twh",
        "maxFyllingsgrad": "max_filling_pct",
        "maxFyllingTWH": "max_filling_twh",
    })

    columns = [
        "zone", "week",
        "min_filling_pct", "median_filling_pct", "max_filling_pct",
        "min_filling_twh", "median_filling_twh", "max_filling_twh",
    ]
    df = df[columns].sort_values(["zone", "week"]).reset_index(drop=True)

    if cache and not df.empty:
        cache_dir.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache_path)
        logger.info("Cached benchmark data to %s", cache_path)

    return df


def fetch_zone_reservoir(
    zone: str,
    start_date: str,
    end_date: str,
    cache: bool = True,
) -> pd.DataFrame:
    """Fetch reservoir data for a single zone within a date range.

    Fetches the full dataset (cached) and filters by zone and date range.

    Args:
        zone: Bidding zone (e.g., "NO_5" for Bergen/Vest-Norge).
        start_date: Start date as "YYYY-MM-DD".
        end_date: End date as "YYYY-MM-DD".
        cache: If True, use caching for the full dataset fetch.

    Returns:
        DataFrame with weekly reservoir data for the specified zone and period.

    Raises:
        ValueError: If zone is not valid.
    """
    valid_zones = list(ZONE_MAP.values())
    if zone not in valid_zones:
        raise ValueError(f"Unknown zone '{zone}'. Valid zones: {valid_zones}")

    df = fetch_reservoir_all(cache=cache)

    if df.empty:
        return df

    # Filter by zone
    df = df[df["zone"] == zone].copy()

    # Filter by date range
    start = pd.Timestamp(start_date, tz="Europe/Oslo")
    end = pd.Timestamp(end_date, tz="Europe/Oslo")
    df = df[(df.index >= start) & (df.index <= end)]

    return df


def fetch_zone_reservoir_with_benchmarks(
    zone: str,
    start_date: str,
    end_date: str,
    cache: bool = True,
) -> pd.DataFrame:
    """Fetch reservoir data with deviation-from-normal features.

    Merges weekly filling data with 20-year min/max/median benchmarks
    to create features like filling_vs_median, filling_vs_min.

    Args:
        zone: Bidding zone (e.g., "NO_5").
        start_date: Start date as "YYYY-MM-DD".
        end_date: End date as "YYYY-MM-DD".
        cache: If True, use caching.

    Returns:
        DataFrame with reservoir data plus benchmark comparison columns:
        filling_vs_median, filling_vs_min, filling_vs_max.
    """
    df = fetch_zone_reservoir(zone, start_date, end_date, cache=cache)
    benchmarks = fetch_reservoir_benchmarks(cache=cache)

    if df.empty or benchmarks.empty:
        return df

    # Filter benchmarks for this zone
    bm = benchmarks[benchmarks["zone"] == zone].set_index("week")

    # Merge on ISO week number
    df = df.copy()
    df["filling_vs_median"] = df.apply(
        lambda r: r["filling_pct"] - bm.loc[r["week"], "median_filling_pct"]
        if r["week"] in bm.index else None,
        axis=1,
    )
    df["filling_vs_min"] = df.apply(
        lambda r: r["filling_pct"] - bm.loc[r["week"], "min_filling_pct"]
        if r["week"] in bm.index else None,
        axis=1,
    )
    df["filling_vs_max"] = df.apply(
        lambda r: r["filling_pct"] - bm.loc[r["week"], "max_filling_pct"]
        if r["week"] in bm.index else None,
        axis=1,
    )

    return df


# --- Entry point for testing ---

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    print("NVE Magasinstatistikk — no API key needed\n")

    # Test 1: Full dataset
    print("=== Full historical dataset ===")
    df_all = fetch_reservoir_all()
    print(f"Shape: {df_all.shape}")
    print(f"Date range: {df_all.index.min()} to {df_all.index.max()}")
    print(f"Zones: {sorted(df_all['zone'].unique())}")
    print()

    # Test 2: Bergen (NO_5) from 2020 to current
    zone = "NO_5"
    start = "2020-01-01"
    end = "2026-02-22"
    print(f"=== {zone} reservoir data ({start} to {end}) ===")
    df_zone = fetch_zone_reservoir(zone, start, end)
    print(f"Shape: {df_zone.shape}")
    print(f"Date range: {df_zone.index.min()} to {df_zone.index.max()}")
    print(f"\nFirst 5 rows:")
    print(df_zone.head())
    print(f"\nLast 5 rows:")
    print(df_zone.tail())
    print(f"\nBasic stats:")
    print(df_zone[["filling_pct", "filling_twh", "capacity_twh", "filling_change"]].describe())
    print(f"\nMissing values:")
    print(df_zone.isna().sum())

    # Test 3: With benchmarks
    print(f"\n=== {zone} with benchmark deviations ===")
    df_bm = fetch_zone_reservoir_with_benchmarks(zone, start, end)
    print(f"Extra columns: filling_vs_median, filling_vs_min, filling_vs_max")
    print(df_bm[["filling_pct", "filling_vs_median", "filling_vs_min", "filling_vs_max"]].tail(10))

    # Test 4: Latest week
    print("\n=== Latest week (all zones) ===")
    df_latest = fetch_reservoir_latest()
    print(df_latest[["zone", "filling_pct", "filling_twh", "filling_change"]].to_string(index=False))

    # Test 5: Benchmarks
    print("\n=== Benchmarks (sample: NO_5, weeks 1–5) ===")
    bm = fetch_reservoir_benchmarks()
    print(bm[(bm["zone"] == "NO_5") & (bm["week"] <= 5)].to_string(index=False))
