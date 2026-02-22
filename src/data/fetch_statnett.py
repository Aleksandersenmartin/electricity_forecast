"""
Fetch operational data from Statnett Driftsdata REST API.

Retrieves physical cross-border flows, production/consumption, real-time
Nordic power balance, and power situation assessments from Statnett
(Norway's Transmission System Operator).

API docs: See docs/statnett_api_reference.md
Base URL: https://driftsdata.statnett.no/restapi

Authentication: None required. Open data.
"""

import logging
from pathlib import Path

import requests
import pandas as pd

logger = logging.getLogger(__name__)

# --- Configuration ---

BASE_URL = "https://driftsdata.statnett.no/restapi"

# Project root for data paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Countries in the LatestDetailedOverview (index 1–7, 0=label, 8=total)
OVERVIEW_COUNTRIES = ["SE", "DK", "NO", "FI", "EE", "LT", "LV"]

# Data categories in the overview
OVERVIEW_CATEGORIES = [
    "ProductionData", "NuclearData", "HydroData", "ThermalData",
    "WindData", "NotSpecifiedData", "ConsumptionData", "NetExchangeData",
]


def _check_response(response: requests.Response) -> None:
    """Check Statnett API response and raise descriptive errors."""
    if response.status_code == 200:
        return
    logger.error("Statnett API error %d: %s", response.status_code, response.text[:200])
    response.raise_for_status()


def _timeseries_to_dataframe(
    data: dict,
    value_column: str,
) -> pd.DataFrame:
    """Convert Statnett timeseries JSON to a DataFrame.

    Statnett returns timeseries as:
        StartPointUTC (ms), EndPointUTC (ms), PeriodTickMs, <values array>

    Args:
        data: JSON response dict with StartPointUTC, PeriodTickMs, and values.
        value_column: Name for the values array key and resulting column.

    Returns:
        DataFrame indexed by date (Europe/Oslo) with one value column.
    """
    values = data.get(value_column, [])
    if not values:
        return pd.DataFrame(columns=[value_column])

    start_utc = pd.Timestamp(data["StartPointUTC"], unit="ms", tz="UTC")
    period_ms = data["PeriodTickMs"]

    if period_ms == 86400000:
        freq = "D"
    elif period_ms == 3600000:
        freq = "h"
    elif period_ms == 60000:
        freq = "min"
    elif period_ms == 1000:
        freq = "s"
    else:
        freq = pd.tseries.offsets.Milli(period_ms)

    index = pd.date_range(start_utc, periods=len(values), freq=freq)
    df = pd.DataFrame({value_column: values}, index=index)
    df.index = df.index.tz_convert("Europe/Oslo")
    df.index.name = "date"

    return df


# --- Physical Flows ---


def fetch_physical_flows(
    from_date: str = "2020-01-01",
    cache: bool = True,
) -> pd.DataFrame:
    """Fetch daily physical cross-border net exchange from Statnett.

    Returns Norway's aggregate net physical flow (MWh/day).
    Negative = net import, positive = net export.

    This is the TSO's own measurement — the most authoritative source
    for Norwegian cross-border physical flows.

    Args:
        from_date: Start date as "YYYY-MM-DD".
        cache: If True, check data/raw/ before fetching and save after.

    Returns:
        DataFrame indexed by date (Europe/Oslo) with column:
        net_exchange_mwh (negative = import, positive = export).
    """
    cache_dir = PROJECT_ROOT / "data" / "raw"
    cache_path = cache_dir / "statnett_physical_flows.parquet"

    if cache and cache_path.exists():
        logger.info("Loading cached physical flows from %s", cache_path)
        return pd.read_parquet(cache_path)

    logger.info("Fetching Statnett physical flows from %s", from_date)
    url = f"{BASE_URL}/Physicalflow/GetData?From={from_date}"
    response = requests.get(url, timeout=60)
    _check_response(response)

    data = response.json()
    df = _timeseries_to_dataframe(data, "PhysicalFlowNetExchange")
    df = df.rename(columns={"PhysicalFlowNetExchange": "net_exchange_mwh"})

    if cache and not df.empty:
        cache_dir.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache_path)
        logger.info("Cached physical flows to %s (%d rows)", cache_path, len(df))

    return df


# --- Production / Consumption ---


def fetch_production_consumption(
    from_date: str = "2020-01-01",
    cache: bool = True,
) -> pd.DataFrame:
    """Fetch daily Norwegian production and consumption from Statnett.

    Returns total production and consumption (MWh/day) for Norway.

    Args:
        from_date: Start date as "YYYY-MM-DD".
        cache: If True, check data/raw/ before fetching and save after.

    Returns:
        DataFrame indexed by date (Europe/Oslo) with columns:
        production_mwh, consumption_mwh, net_balance_mwh.
    """
    cache_dir = PROJECT_ROOT / "data" / "raw"
    cache_path = cache_dir / "statnett_prod_cons.parquet"

    if cache and cache_path.exists():
        logger.info("Loading cached production/consumption from %s", cache_path)
        return pd.read_parquet(cache_path)

    logger.info("Fetching Statnett production/consumption from %s", from_date)
    url = f"{BASE_URL}/ProductionConsumption/GetData?From={from_date}"
    response = requests.get(url, timeout=60)
    _check_response(response)

    data = response.json()

    # Both arrays share the same time axis
    prod = data.get("Production", [])
    cons = data.get("Consumption", [])

    if not prod:
        return pd.DataFrame(columns=["production_mwh", "consumption_mwh", "net_balance_mwh"])

    start_utc = pd.Timestamp(data["StartPointUTC"], unit="ms", tz="UTC")
    period_ms = data["PeriodTickMs"]
    freq = "D" if period_ms == 86400000 else pd.tseries.offsets.Milli(period_ms)

    index = pd.date_range(start_utc, periods=len(prod), freq=freq)

    df = pd.DataFrame({
        "production_mwh": prod,
        "consumption_mwh": cons,
    }, index=index)

    df.index = df.index.tz_convert("Europe/Oslo")
    df.index.name = "date"
    df["net_balance_mwh"] = df["production_mwh"] - df["consumption_mwh"]

    if cache and not df.empty:
        cache_dir.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache_path)
        logger.info("Cached prod/cons to %s (%d rows)", cache_path, len(df))

    return df


# --- Real-Time Overview ---


def _parse_overview_value(value_str: str) -> float | None:
    """Parse a formatted number string from LatestDetailedOverview.

    Values use non-breaking space (\xa0) as thousands separator.
    Returns None for empty or label entries.
    """
    if not value_str or value_str.strip() == "":
        return None
    # Remove non-breaking spaces and regular spaces used as thousands separators
    cleaned = value_str.replace("\xa0", "").replace(" ", "").replace(",", ".")
    try:
        return float(cleaned)
    except ValueError:
        return None


def fetch_latest_overview() -> pd.DataFrame:
    """Fetch real-time Nordic power balance snapshot from Statnett.

    Returns current production breakdown by type and country (MW).
    Countries: SE, DK, NO, FI, EE, LT, LV.
    Categories: production, nuclear, hydro, thermal, wind, other,
                consumption, net_exchange.

    This is NOT cached — always returns fresh real-time data.

    Returns:
        DataFrame with countries as index and generation types as columns (MW).
    """
    logger.info("Fetching Statnett real-time overview")
    url = f"{BASE_URL}/ProductionConsumption/GetLatestDetailedOverview"
    response = requests.get(url, timeout=30)
    _check_response(response)

    data = response.json()

    # Column name mapping
    col_map = {
        "ProductionData": "production_mw",
        "NuclearData": "nuclear_mw",
        "HydroData": "hydro_mw",
        "ThermalData": "thermal_mw",
        "WindData": "wind_mw",
        "NotSpecifiedData": "other_mw",
        "ConsumptionData": "consumption_mw",
        "NetExchangeData": "net_exchange_mw",
    }

    rows = {}
    for category, col_name in col_map.items():
        items = data.get(category, [])
        # Items 1–7 are country values (index 0 = label, 8 = total)
        for i, country in enumerate(OVERVIEW_COUNTRIES, start=1):
            if i < len(items):
                val = _parse_overview_value(items[i].get("value", ""))
                rows.setdefault(country, {})[col_name] = val

    df = pd.DataFrame.from_dict(rows, orient="index")
    df.index.name = "country"

    # Add timestamp
    measured_at = data.get("MeasuredAt")
    if measured_at:
        ts = pd.Timestamp(measured_at, unit="ms", tz="UTC").tz_convert("Europe/Oslo")
        df.attrs["measured_at"] = str(ts)
        logger.info("Overview measured at %s", ts)

    return df


# --- Power Situation ---


def fetch_power_situation() -> pd.DataFrame:
    """Fetch Statnett's power situation assessment per zone.

    Returns Statnett's qualitative assessment of the power situation
    for each Norwegian bidding zone (NO1–NO5).

    This is NOT cached — always returns fresh data.

    Returns:
        DataFrame with columns: zone, situation, description.
    """
    logger.info("Fetching Statnett power situation")
    url = f"{BASE_URL}/ElspotPowerSituation/GetPowerSituations/"
    response = requests.get(url, timeout=30)
    _check_response(response)

    data = response.json()

    records = []
    for item in data:
        zone = item.get("elspotId", "")
        # Normalize zone name to match project convention (NO1 -> NO_1)
        zone_normalized = zone[:2] + "_" + zone[2:] if len(zone) == 3 else zone
        records.append({
            "zone": zone_normalized,
            "situation": item.get("powerSituation", ""),
            "description_en": item.get("translatedText", {}).get("en", {}).get("shortDescription", ""),
            "description_no": item.get("translatedText", {}).get("no", {}).get("shortDescription", ""),
        })

    df = pd.DataFrame(records)
    return df


# --- Grid Frequency ---


def fetch_frequency(
    from_date: str,
    resolution: str = "minute",
) -> pd.DataFrame:
    """Fetch grid frequency data from Statnett.

    Returns frequency measurements (Hz). Target: 50.000 Hz.
    Deviations indicate supply/demand imbalance in the grid.

    Args:
        from_date: Start date as "YYYY-MM-DD".
        resolution: "minute" or "second".

    Returns:
        DataFrame indexed by timestamp (Europe/Oslo) with column:
        frequency_hz.
    """
    endpoint = "ByMinute" if resolution == "minute" else "BySecond"
    logger.info("Fetching Statnett frequency (%s) from %s", resolution, from_date)

    url = f"{BASE_URL}/Frequency/{endpoint}?From={from_date}"
    response = requests.get(url, timeout=60)
    _check_response(response)

    data = response.json()
    measurements = data.get("Measurements", [])

    if not measurements:
        return pd.DataFrame(columns=["frequency_hz"])

    start_utc = pd.Timestamp(data["StartPointUTC"], unit="ms", tz="UTC")
    period_ms = data["PeriodTickMs"]

    if period_ms == 60000:
        freq = "min"
    elif period_ms == 1000:
        freq = "s"
    else:
        freq = pd.tseries.offsets.Milli(period_ms)

    index = pd.date_range(start_utc, periods=len(measurements), freq=freq)
    df = pd.DataFrame({"frequency_hz": measurements}, index=index)
    df.index = df.index.tz_convert("Europe/Oslo")
    df.index.name = "datetime"

    return df


# --- Entry point for testing ---

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    print("Statnett Driftsdata API — no API key needed\n")

    # Test 1: Physical flows (2020–current)
    from_date = "2020-01-01"
    print(f"=== Physical cross-border flows ({from_date} to current) ===")
    df_flows = fetch_physical_flows(from_date=from_date, cache=False)
    print(f"Shape: {df_flows.shape}")
    print(f"Date range: {df_flows.index.min()} to {df_flows.index.max()}")
    print(f"\nFirst 5 rows:")
    print(df_flows.head())
    print(f"\nLast 5 rows:")
    print(df_flows.tail())
    print(f"\nBasic stats:")
    print(df_flows.describe())
    print(f"\nMissing values: {df_flows.isna().sum().to_dict()}")

    # Test 2: Production/consumption (2020–current)
    print(f"\n=== Production/Consumption ({from_date} to current) ===")
    df_pc = fetch_production_consumption(from_date=from_date, cache=False)
    print(f"Shape: {df_pc.shape}")
    print(f"Date range: {df_pc.index.min()} to {df_pc.index.max()}")
    print(f"\nFirst 5 rows:")
    print(df_pc.head())
    print(f"\nLast 5 rows:")
    print(df_pc.tail())
    print(f"\nBasic stats:")
    print(df_pc.describe())
    print(f"\nMissing values: {df_pc.isna().sum().to_dict()}")

    # Test 3: Real-time overview
    print("\n=== Real-time Nordic overview ===")
    df_overview = fetch_latest_overview()
    if "measured_at" in df_overview.attrs:
        print(f"Measured at: {df_overview.attrs['measured_at']}")
    print(df_overview.to_string())

    # Test 4: Power situation
    print("\n=== Power situation per zone ===")
    df_sit = fetch_power_situation()
    print(df_sit.to_string(index=False))

    # Test 5: Frequency (last few minutes)
    print("\n=== Grid frequency (last reading) ===")
    from datetime import date
    df_freq = fetch_frequency(from_date=str(date.today()), resolution="minute")
    print(f"Shape: {df_freq.shape}")
    if not df_freq.empty:
        print(f"Time range: {df_freq.index.min()} to {df_freq.index.max()}")
        print(f"Frequency: {df_freq['frequency_hz'].iloc[-1]:.4f} Hz")
        print(f"Stats: mean={df_freq['frequency_hz'].mean():.4f}, "
              f"min={df_freq['frequency_hz'].min():.4f}, "
              f"max={df_freq['frequency_hz'].max():.4f}")
