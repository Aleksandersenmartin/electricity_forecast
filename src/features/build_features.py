"""
Feature engineering pipeline for electricity price forecasting.

Merges all data sources (weather, FX, commodities, reservoir, Statnett,
and optionally ENTSO-E prices) into a single hourly-indexed feature matrix
per bidding zone.

Each build_* function creates one feature group. The orchestrator
build_feature_matrix() merges them all on an hourly Europe/Oslo spine.

Usage:
    python -m src.features.build_features
"""

import logging
from pathlib import Path

import holidays
import numpy as np
import pandas as pd

from src.data.fetch_metro import fetch_zone_weather
from src.data.fetch_fx import fetch_eur_nok_daily_filled
from src.data.fetch_commodity import fetch_commodities_daily_filled
from src.data.fetch_reservoir import fetch_zone_reservoir_with_benchmarks
from src.data.fetch_statnett import fetch_physical_flows, fetch_production_consumption

# ENTSO-E imports — primary data source for prices, load, generation, flows
try:
    from src.data.fetch_electricity import (
        fetch_prices as fetch_entsoe_prices,
        fetch_load as fetch_entsoe_load,
        fetch_generation as fetch_entsoe_generation,
        fetch_crossborder_flows as fetch_entsoe_flows,
        fetch_foreign_prices as fetch_entsoe_foreign_prices,
        ZONE_CABLES,
        FOREIGN_PRICE_ZONES,
    )
    ENTSOE_AVAILABLE = True
except Exception:
    ENTSOE_AVAILABLE = False

# Nord Pool — fallback if ENTSO-E key is not set
try:
    from src.data.fetch_nordpool import fetch_prices as fetch_nordpool_prices
    NORDPOOL_AVAILABLE = True
except Exception:
    NORDPOOL_AVAILABLE = False

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


# ---------------------------------------------------------------------------
# 1. Calendar Features
# ---------------------------------------------------------------------------

def build_calendar_features(hourly_index: pd.DatetimeIndex) -> pd.DataFrame:
    """Derive calendar/time features from an hourly DatetimeIndex.

    Args:
        hourly_index: Timezone-aware hourly DatetimeIndex (Europe/Oslo).

    Returns:
        DataFrame with columns: hour_of_day, day_of_week, month,
        week_of_year, is_weekend, is_holiday, is_business_hour.
    """
    no_holidays = holidays.Norway()

    df = pd.DataFrame(index=hourly_index)
    df["hour_of_day"] = hourly_index.hour
    df["day_of_week"] = hourly_index.dayofweek  # Monday=0
    df["month"] = hourly_index.month
    df["week_of_year"] = hourly_index.isocalendar().week.values
    df["is_weekend"] = (hourly_index.dayofweek >= 5).astype(int)
    df["is_holiday"] = hourly_index.date
    df["is_holiday"] = df["is_holiday"].apply(lambda d: int(d in no_holidays))
    df["is_business_hour"] = (
        (hourly_index.hour >= 8)
        & (hourly_index.hour <= 17)
        & (hourly_index.dayofweek < 5)
    ).astype(int)

    logger.info("Calendar features: %d rows, %d columns", len(df), len(df.columns))
    return df


# ---------------------------------------------------------------------------
# 2. Weather Features
# ---------------------------------------------------------------------------

def build_weather_features(
    zone: str,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """Load weather data and derive additional features.

    Args:
        zone: Bidding zone (e.g., "NO_5").
        start_date: Start date as "YYYY-MM-DD".
        end_date: End date as "YYYY-MM-DD".

    Returns:
        DataFrame with hourly weather + derived features:
        temperature, wind_speed, precipitation,
        temperature_lag_24h, temperature_rolling_24h_mean.
    """
    df = fetch_zone_weather(zone, start_date, end_date, cache=True)

    if df.empty:
        logger.warning("No weather data for %s", zone)
        return pd.DataFrame()

    # Derived features
    df["temperature_lag_24h"] = df["air_temperature"].shift(24)
    df["temperature_rolling_24h_mean"] = (
        df["air_temperature"].rolling(window=24, min_periods=1).mean()
    )

    # Rename for consistency with CLAUDE.md feature names
    df = df.rename(columns={
        "air_temperature": "temperature",
    })

    logger.info("Weather features (%s): %d rows, %d columns", zone, len(df), len(df.columns))
    return df


# ---------------------------------------------------------------------------
# 3. Commodity Features
# ---------------------------------------------------------------------------

def build_commodity_features(
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """Load commodity prices (daily, forward-filled) and derive trend features.

    Args:
        start_date: Start date as "YYYY-MM-DD".
        end_date: End date as "YYYY-MM-DD".

    Returns:
        DataFrame with daily index: ttf_gas_close, brent_oil_close,
        coal_close, ng_fut_close, ttf_gas_change_7d.
    """
    df = fetch_commodities_daily_filled(start_date, end_date, cache=True)

    if df.empty:
        logger.warning("No commodity data")
        return pd.DataFrame()

    # Select close prices and rename to clean names
    rename_map = {
        "TTF-GAS_close": "ttf_gas_close",
        "BRENTOIL-SPOT_close": "brent_oil_close",
        "COAL_close": "coal_close",
        "NG-FUT_close": "ng_fut_close",
    }

    cols = [c for c in rename_map if c in df.columns]
    result = df[cols].rename(columns=rename_map)

    # 7-day percentage change for TTF gas (trend signal)
    if "ttf_gas_close" in result.columns:
        result["ttf_gas_change_7d"] = result["ttf_gas_close"].pct_change(periods=7)

    logger.info("Commodity features: %d rows, %d columns", len(result), len(result.columns))
    return result


# ---------------------------------------------------------------------------
# 4. FX Features
# ---------------------------------------------------------------------------

def build_fx_features(
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """Load EUR/NOK exchange rate (daily, forward-filled).

    Args:
        start_date: Start date as "YYYY-MM-DD".
        end_date: End date as "YYYY-MM-DD".

    Returns:
        DataFrame with daily index and column: eur_nok.
    """
    df = fetch_eur_nok_daily_filled(start_date, end_date, cache=True)

    if df.empty:
        logger.warning("No FX data")
        return pd.DataFrame()

    logger.info("FX features: %d rows, %d columns", len(df), len(df.columns))
    return df


# ---------------------------------------------------------------------------
# 5. Reservoir Features
# ---------------------------------------------------------------------------

def build_reservoir_features(
    zone: str,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """Load NVE reservoir data (weekly) with benchmark deviations.

    Args:
        zone: Bidding zone (e.g., "NO_5").
        start_date: Start date as "YYYY-MM-DD".
        end_date: End date as "YYYY-MM-DD".

    Returns:
        DataFrame with weekly index: reservoir_filling_pct,
        reservoir_filling_twh, reservoir_filling_change,
        reservoir_vs_median, reservoir_vs_min.
    """
    df = fetch_zone_reservoir_with_benchmarks(zone, start_date, end_date, cache=True)

    if df.empty:
        logger.warning("No reservoir data for %s", zone)
        return pd.DataFrame()

    # Select and rename columns for feature matrix
    rename_map = {
        "filling_pct": "reservoir_filling_pct",
        "filling_twh": "reservoir_filling_twh",
        "filling_change": "reservoir_filling_change",
        "filling_vs_median": "reservoir_vs_median",
        "filling_vs_min": "reservoir_vs_min",
    }

    cols = [c for c in rename_map if c in df.columns]
    result = df[cols].rename(columns=rename_map)

    # Drop non-numeric columns that came along (zone, year, week are in parent df)
    logger.info("Reservoir features (%s): %d rows, %d columns", zone, len(result), len(result.columns))
    return result


# ---------------------------------------------------------------------------
# 6. Statnett Features
# ---------------------------------------------------------------------------

def build_statnett_features(
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """Load Statnett physical flows and production/consumption (daily).

    Args:
        start_date: Start date as "YYYY-MM-DD".
        end_date: End date as "YYYY-MM-DD".

    Returns:
        DataFrame with daily index: net_exchange_mwh, production_mwh,
        consumption_mwh, net_balance_mwh.
    """
    flows = fetch_physical_flows(cache=True)
    prod_cons = fetch_production_consumption(cache=True)

    if flows.empty and prod_cons.empty:
        logger.warning("No Statnett data")
        return pd.DataFrame()

    # Filter to date range
    start = pd.Timestamp(start_date, tz="Europe/Oslo")
    end = pd.Timestamp(end_date, tz="Europe/Oslo")

    result = pd.DataFrame()

    if not flows.empty:
        flows_filtered = flows[(flows.index >= start) & (flows.index <= end)]
        result = flows_filtered

    if not prod_cons.empty:
        pc_filtered = prod_cons[(prod_cons.index >= start) & (prod_cons.index <= end)]
        if result.empty:
            result = pc_filtered
        else:
            result = result.join(pc_filtered, how="outer")

    logger.info("Statnett features: %d rows, %d columns", len(result), len(result.columns))
    return result


# ---------------------------------------------------------------------------
# 7. ENTSO-E Load Features
# ---------------------------------------------------------------------------

def build_entsoe_load_features(
    zone: str,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """Fetch ENTSO-E actual load (MW) and derive lag features.

    Args:
        zone: Bidding zone (e.g., "NO_5").
        start_date: Start date as "YYYY-MM-DD".
        end_date: End date as "YYYY-MM-DD".

    Returns:
        DataFrame with columns: actual_load, load_lag_24h, load_lag_168h,
        load_rolling_24h_mean. Empty DataFrame if ENTSO-E unavailable.
    """
    if not ENTSOE_AVAILABLE:
        logger.info("ENTSO-E load: skipped (entsoe-py not available)")
        return pd.DataFrame()

    try:
        raw = fetch_entsoe_load(zone, start_date, end_date, cache=True)
    except Exception as e:
        logger.warning("ENTSO-E load unavailable for %s: %s", zone, e)
        return pd.DataFrame()

    if raw.empty:
        return pd.DataFrame()

    # Extract the load column (entsoe-py returns 'Actual Load')
    load_col = raw.columns[0] if len(raw.columns) > 0 else None
    if load_col is None:
        return pd.DataFrame()

    df = pd.DataFrame(index=raw.index)
    df["actual_load"] = raw[load_col]
    df["load_lag_24h"] = df["actual_load"].shift(24)
    df["load_lag_168h"] = df["actual_load"].shift(168)
    df["load_rolling_24h_mean"] = df["actual_load"].rolling(window=24, min_periods=1).mean()

    logger.info("ENTSO-E load features (%s): %d rows, %d columns", zone, len(df), len(df.columns))
    return df


# ---------------------------------------------------------------------------
# 8. ENTSO-E Generation Features
# ---------------------------------------------------------------------------

# PSR type groupings for aggregation
_HYDRO_TYPES = {"Hydro Water Reservoir", "Hydro Run-of-river and poundage", "Hydro Pumped Storage"}
_WIND_TYPES = {"Wind Onshore", "Wind Offshore"}


def build_entsoe_generation_features(
    zone: str,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """Fetch ENTSO-E generation per type and derive aggregate features.

    Aggregates generation into hydro, wind, and total. Computes shares.

    Args:
        zone: Bidding zone (e.g., "NO_5").
        start_date: Start date as "YYYY-MM-DD".
        end_date: End date as "YYYY-MM-DD".

    Returns:
        DataFrame with columns: generation_hydro, generation_wind,
        generation_total, hydro_share, wind_share.
        Empty DataFrame if ENTSO-E unavailable.
    """
    if not ENTSOE_AVAILABLE:
        logger.info("ENTSO-E generation: skipped (entsoe-py not available)")
        return pd.DataFrame()

    try:
        raw = fetch_entsoe_generation(zone, start_date, end_date, cache=True)
    except Exception as e:
        logger.warning("ENTSO-E generation unavailable for %s: %s", zone, e)
        return pd.DataFrame()

    if raw.empty:
        return pd.DataFrame()

    df = pd.DataFrame(index=raw.index)

    # Aggregate hydro types (B10 + B11 + B12)
    hydro_cols = [c for c in raw.columns if c in _HYDRO_TYPES]
    df["generation_hydro"] = raw[hydro_cols].sum(axis=1) if hydro_cols else 0

    # Aggregate wind types (B18 + B19)
    wind_cols = [c for c in raw.columns if c in _WIND_TYPES]
    df["generation_wind"] = raw[wind_cols].sum(axis=1) if wind_cols else 0

    # Total generation
    df["generation_total"] = raw.sum(axis=1)

    # Shares (avoid division by zero)
    df["hydro_share"] = df["generation_hydro"] / df["generation_total"].replace(0, np.nan)
    df["wind_share"] = df["generation_wind"] / df["generation_total"].replace(0, np.nan)

    logger.info("ENTSO-E generation features (%s): %d rows, %d columns", zone, len(df), len(df.columns))
    return df


# ---------------------------------------------------------------------------
# 9. ENTSO-E Cross-Border Flow Features
# ---------------------------------------------------------------------------

def build_entsoe_flow_features(
    zone: str,
    start_date: str,
    end_date: str,
    eur_nok_hourly: pd.Series | None = None,
) -> pd.DataFrame:
    """Fetch ENTSO-E crossborder flows and foreign prices per cable.

    For each cable connected to the zone, fetches:
    - Net flow (MW, positive = export from Norway)
    - Foreign zone day-ahead price (EUR/MWh)
    - Price spread (Norwegian zone minus foreign zone)
    - Foreign price in NOK/kWh (when FX available)

    Also computes aggregate features: total_net_export, n_cables_exporting.

    Args:
        zone: Norwegian bidding zone (e.g., "NO_2").
        start_date: Start date as "YYYY-MM-DD".
        end_date: End date as "YYYY-MM-DD".
        eur_nok_hourly: Optional hourly EUR/NOK exchange rate Series
            for converting foreign prices to NOK/kWh.

    Returns:
        DataFrame with per-cable flow/price/spread columns plus aggregates.
        Empty DataFrame if ENTSO-E unavailable or zone has no cables.
    """
    if not ENTSOE_AVAILABLE:
        logger.info("ENTSO-E flows: skipped (entsoe-py not available)")
        return pd.DataFrame()

    cables = ZONE_CABLES.get(zone, [])
    if not cables:
        logger.info("ENTSO-E flows: %s has no international cables", zone)
        return pd.DataFrame()

    # Create hourly spine for alignment
    hourly_index = pd.date_range(start=start_date, end=end_date, freq="h", tz="Europe/Oslo")

    df = pd.DataFrame(index=hourly_index)
    flow_cols = []

    for foreign_zone in cables:
        fz_lower = foreign_zone.lower().replace("_", "")
        prefix = f"flow_{zone.lower()}_{fz_lower}"

        # Fetch net flow (direction: Norwegian zone → foreign zone)
        try:
            flow_raw = fetch_entsoe_flows(zone, foreign_zone, start_date, end_date, cache=True)
            if not flow_raw.empty:
                flow_series = flow_raw.iloc[:, 0]
                # Resample to hourly if 15-min data (e.g., DE_LU)
                if len(flow_series) > len(hourly_index) * 1.2:
                    flow_series = flow_series.resample("h").mean()
                df[prefix] = flow_series.reindex(hourly_index, method="nearest", tolerance="1h")
                flow_cols.append(prefix)
        except Exception as e:
            logger.warning("Flow %s→%s failed: %s", zone, foreign_zone, e)

        # Fetch foreign zone price
        if foreign_zone in FOREIGN_PRICE_ZONES:
            try:
                fp_raw = fetch_entsoe_foreign_prices(foreign_zone, start_date, end_date, cache=True)
                if not fp_raw.empty:
                    price_series = fp_raw.iloc[:, 0]
                    price_col = f"price_{fz_lower}_eur_mwh"
                    df[price_col] = price_series.reindex(hourly_index, method="nearest", tolerance="1h")

                    # Convert to NOK/kWh if FX available
                    if eur_nok_hourly is not None:
                        fx_aligned = eur_nok_hourly.reindex(hourly_index, method="ffill")
                        nok_col = f"price_{fz_lower}_nok_kwh"
                        df[nok_col] = df[price_col] * fx_aligned / 1000
            except Exception as e:
                logger.warning("Foreign prices for %s failed: %s", foreign_zone, e)

    # Aggregate flow features
    if flow_cols:
        df["total_net_export"] = df[flow_cols].sum(axis=1)
        df["n_cables_exporting"] = (df[flow_cols] > 0).sum(axis=1)

    # Drop rows that are all NaN (outside data range)
    df = df.dropna(how="all")

    logger.info("ENTSO-E flow features (%s): %d rows, %d columns", zone, len(df), len(df.columns))
    return df


# ---------------------------------------------------------------------------
# 10. EUR → NOK Price Conversion
# ---------------------------------------------------------------------------

def _convert_eur_to_nok(
    price_eur_mwh: pd.Series,
    eur_nok: pd.Series,
) -> tuple[pd.Series, pd.Series]:
    """Convert EUR/MWh → NOK/MWh and NOK/kWh.

    Args:
        price_eur_mwh: Hourly prices in EUR/MWh.
        eur_nok: Hourly EUR/NOK exchange rate (forward-filled from daily).

    Returns:
        Tuple of (price_nok_mwh, price_nok_kwh) Series.
    """
    price_nok_mwh = price_eur_mwh * eur_nok
    price_nok_kwh = price_nok_mwh / 1000
    return price_nok_mwh, price_nok_kwh


def build_nok_price_features(price_nok_mwh: pd.Series) -> pd.DataFrame:
    """Build lag/rolling/diff features from NOK/MWh and NOK/kWh price series.

    Lags are computed FROM the NOK series (not EUR lags × FX), so each
    lagged value naturally uses the FX rate from that original timestamp.

    Args:
        price_nok_mwh: Hourly prices in NOK/MWh.

    Returns:
        DataFrame with 16 columns: lag/rolling/diff for both NOK/MWh and NOK/kWh.
    """
    price_nok_kwh = price_nok_mwh / 1000

    df = pd.DataFrame(index=price_nok_mwh.index)

    # NOK/MWh features
    df["price_nok_mwh_lag_1h"] = price_nok_mwh.shift(1)
    df["price_nok_mwh_lag_24h"] = price_nok_mwh.shift(24)
    df["price_nok_mwh_lag_168h"] = price_nok_mwh.shift(168)
    df["price_nok_mwh_rolling_24h_mean"] = price_nok_mwh.rolling(window=24, min_periods=1).mean()
    df["price_nok_mwh_rolling_24h_std"] = price_nok_mwh.rolling(window=24, min_periods=1).std()
    df["price_nok_mwh_rolling_168h_mean"] = price_nok_mwh.rolling(window=168, min_periods=1).mean()
    df["price_nok_mwh_diff_24h"] = price_nok_mwh - price_nok_mwh.shift(24)
    df["price_nok_mwh_diff_168h"] = price_nok_mwh - price_nok_mwh.shift(168)

    # NOK/kWh features
    df["price_nok_kwh_lag_1h"] = price_nok_kwh.shift(1)
    df["price_nok_kwh_lag_24h"] = price_nok_kwh.shift(24)
    df["price_nok_kwh_lag_168h"] = price_nok_kwh.shift(168)
    df["price_nok_kwh_rolling_24h_mean"] = price_nok_kwh.rolling(window=24, min_periods=1).mean()
    df["price_nok_kwh_rolling_24h_std"] = price_nok_kwh.rolling(window=24, min_periods=1).std()
    df["price_nok_kwh_rolling_168h_mean"] = price_nok_kwh.rolling(window=168, min_periods=1).mean()
    df["price_nok_kwh_diff_24h"] = price_nok_kwh - price_nok_kwh.shift(24)
    df["price_nok_kwh_diff_168h"] = price_nok_kwh - price_nok_kwh.shift(168)

    logger.info("NOK price features: %d rows, %d columns", len(df), len(df.columns))
    return df


# ---------------------------------------------------------------------------
# 8a. Price Features (placeholder — ENTSO-E key not yet available)
# ---------------------------------------------------------------------------

def build_price_features(prices: pd.Series | None = None) -> pd.DataFrame:
    """Build autoregressive price features from an existing price Series.

    Returns an empty DataFrame if prices is None (ENTSO-E not available yet).
    Once ENTSO-E data is available, pass the hourly price Series to get:
    lag, rolling, and diff features.

    Args:
        prices: Hourly price Series with timezone-aware DatetimeIndex,
                or None if not available.

    Returns:
        DataFrame with price lag/rolling/diff features, or empty DataFrame.
    """
    if prices is None:
        logger.info("Price features: skipped (no price data provided)")
        return pd.DataFrame()

    df = pd.DataFrame(index=prices.index)
    df["price_lag_1h"] = prices.shift(1)
    df["price_lag_24h"] = prices.shift(24)
    df["price_lag_168h"] = prices.shift(168)
    df["price_rolling_24h_mean"] = prices.rolling(window=24, min_periods=1).mean()
    df["price_rolling_24h_std"] = prices.rolling(window=24, min_periods=1).std()
    df["price_rolling_168h_mean"] = prices.rolling(window=168, min_periods=1).mean()
    df["price_diff_24h"] = prices - prices.shift(24)
    df["price_diff_168h"] = prices - prices.shift(168)

    logger.info("Price features: %d rows, %d columns", len(df), len(df.columns))
    return df


# ---------------------------------------------------------------------------
# 8b. Nord Pool Price Features
# ---------------------------------------------------------------------------

def build_nordpool_price_features(
    zone: str,
    start_date: str,
    end_date: str,
    eur_nok_hourly: pd.Series | None = None,
) -> pd.DataFrame:
    """Fetch Nord Pool prices and build autoregressive price features.

    Downloads day-ahead prices for all zones via the Nord Pool Data Portal,
    extracts the target zone's price, and derives lag/rolling/diff features
    using build_price_features(). When eur_nok_hourly is provided, also
    computes price_nok_mwh and price_nok_kwh base columns.

    Args:
        zone: Bidding zone (e.g., "NO_5").
        start_date: Start date as "YYYY-MM-DD".
        end_date: End date as "YYYY-MM-DD".
        eur_nok_hourly: Optional hourly EUR/NOK exchange rate Series.
            When provided, adds price_nok_mwh and price_nok_kwh columns.

    Returns:
        DataFrame with columns: price_eur_mwh, price_lag_1h, price_lag_24h,
        price_lag_168h, price_rolling_24h_mean, price_rolling_24h_std,
        price_rolling_168h_mean, price_diff_24h, price_diff_168h,
        and optionally price_nok_mwh, price_nok_kwh.
        Empty DataFrame if Nord Pool data is unavailable.
    """
    prices_all = fetch_nordpool_prices(start_date, end_date, cache=True)

    if prices_all.empty or zone not in prices_all.columns:
        logger.warning("No Nord Pool prices for %s", zone)
        return pd.DataFrame()

    zone_prices = prices_all[zone].rename("price_eur_mwh")

    # Build autoregressive features from the price series
    lag_features = build_price_features(zone_prices)

    # Combine raw price + derived features
    result = zone_prices.to_frame()
    if not lag_features.empty:
        result = result.join(lag_features, how="left")

    # Add NOK price base columns if FX rate is available
    if eur_nok_hourly is not None:
        # Align FX rate to price index
        aligned_fx = eur_nok_hourly.reindex(result.index, method="ffill")
        nok_mwh, nok_kwh = _convert_eur_to_nok(result["price_eur_mwh"], aligned_fx)
        result["price_nok_mwh"] = nok_mwh
        result["price_nok_kwh"] = nok_kwh
        logger.info("Added NOK price columns (price_nok_mwh, price_nok_kwh)")

    logger.info(
        "Nord Pool price features (%s): %d rows, %d columns",
        zone, len(result), len(result.columns),
    )
    return result


# ---------------------------------------------------------------------------
# 8c. ENTSO-E Price Features (primary source)
# ---------------------------------------------------------------------------

def build_entsoe_price_features(
    zone: str,
    start_date: str,
    end_date: str,
    eur_nok_hourly: pd.Series | None = None,
) -> pd.DataFrame:
    """Fetch ENTSO-E day-ahead prices and build autoregressive price features.

    Downloads day-ahead prices for a single zone via the ENTSO-E Transparency
    Platform, and derives lag/rolling/diff features using build_price_features().
    When eur_nok_hourly is provided, also computes price_nok_mwh and
    price_nok_kwh base columns.

    Args:
        zone: Bidding zone (e.g., "NO_5").
        start_date: Start date as "YYYY-MM-DD".
        end_date: End date as "YYYY-MM-DD".
        eur_nok_hourly: Optional hourly EUR/NOK exchange rate Series.
            When provided, adds price_nok_mwh and price_nok_kwh columns.

    Returns:
        DataFrame with columns: price_eur_mwh, price_lag_1h, price_lag_24h,
        price_lag_168h, price_rolling_24h_mean, price_rolling_24h_std,
        price_rolling_168h_mean, price_diff_24h, price_diff_168h,
        and optionally price_nok_mwh, price_nok_kwh.
        Empty DataFrame if ENTSO-E data is unavailable.
    """
    prices_raw = fetch_entsoe_prices(zone, start_date, end_date, cache=True)

    if prices_raw.empty:
        logger.warning("No ENTSO-E prices for %s", zone)
        return pd.DataFrame()

    # ENTSO-E returns a Series or single-column DataFrame
    if isinstance(prices_raw, pd.DataFrame):
        zone_prices = prices_raw.iloc[:, 0].rename("price_eur_mwh")
    else:
        zone_prices = prices_raw.rename("price_eur_mwh")

    # Build autoregressive features from the price series
    lag_features = build_price_features(zone_prices)

    # Combine raw price + derived features
    result = zone_prices.to_frame()
    if not lag_features.empty:
        result = result.join(lag_features, how="left")

    # Add NOK price base columns if FX rate is available
    if eur_nok_hourly is not None:
        aligned_fx = eur_nok_hourly.reindex(result.index, method="ffill")
        nok_mwh, nok_kwh = _convert_eur_to_nok(result["price_eur_mwh"], aligned_fx)
        result["price_nok_mwh"] = nok_mwh
        result["price_nok_kwh"] = nok_kwh
        logger.info("Added NOK price columns (price_nok_mwh, price_nok_kwh)")

    logger.info(
        "ENTSO-E price features (%s): %d rows, %d columns",
        zone, len(result), len(result.columns),
    )
    return result


# ---------------------------------------------------------------------------
# 9. Orchestrator — Merge all features
# ---------------------------------------------------------------------------

def _resample_to_hourly(
    df: pd.DataFrame,
    hourly_index: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Resample a daily/weekly DataFrame to hourly by forward-fill.

    Handles timezone-naive daily sources by localizing them first,
    then reindexing to the hourly spine and forward-filling.

    Args:
        df: DataFrame with DatetimeIndex (daily or weekly).
        hourly_index: Target hourly DatetimeIndex (Europe/Oslo).

    Returns:
        DataFrame reindexed to hourly_index, forward-filled.
    """
    if df.empty:
        return df

    idx = df.index

    # Make timezone-aware if needed
    if idx.tz is None:
        idx = idx.tz_localize("Europe/Oslo")
        df = df.copy()
        df.index = idx

    # Reindex to hourly and forward-fill
    df = df.reindex(hourly_index, method="ffill")

    return df


def build_feature_matrix(
    zone: str,
    start_date: str,
    end_date: str,
    prices: pd.Series | None = None,
) -> pd.DataFrame:
    """Build the complete feature matrix for a zone.

    Orchestrates all build_* functions, merges on an hourly spine,
    and caches the result to data/processed/.

    Args:
        zone: Bidding zone (e.g., "NO_5").
        start_date: Start date as "YYYY-MM-DD".
        end_date: End date as "YYYY-MM-DD".
        prices: Optional hourly price Series (from ENTSO-E).

    Returns:
        DataFrame with hourly index (Europe/Oslo) and all feature columns.
    """
    logger.info("Building feature matrix for %s (%s to %s)", zone, start_date, end_date)

    # Create hourly spine
    hourly_index = pd.date_range(
        start=start_date,
        end=end_date,
        freq="h",
        tz="Europe/Oslo",
    )
    hourly_index.name = "timestamp"

    # Start with calendar features (always available, no missing)
    result = build_calendar_features(hourly_index)

    # --- Hourly sources (direct join) ---

    # Weather
    try:
        weather = build_weather_features(zone, start_date, end_date)
        if not weather.empty:
            result = result.join(weather, how="left")
            logger.info("Merged weather: +%d columns", len(weather.columns))
    except Exception as e:
        logger.error("Failed to build weather features: %s", e)

    # --- FX (loaded early so hourly EUR/NOK is available for price conversion) ---

    fx_hourly_series = None
    try:
        fx = build_fx_features(start_date, end_date)
        if not fx.empty:
            fx_hourly = _resample_to_hourly(fx, hourly_index)
            result = result.join(fx_hourly, how="left")
            logger.info("Merged FX: +%d columns", len(fx.columns))
            # Save the hourly EUR/NOK series for price conversion
            if "eur_nok" in fx_hourly.columns:
                fx_hourly_series = fx_hourly["eur_nok"]
    except Exception as e:
        logger.error("Failed to build FX features: %s", e)

    # --- Price features (hourly) ---

    # Option A: Pre-built prices passed explicitly
    try:
        price_feats = build_price_features(prices)
        if not price_feats.empty:
            result = result.join(price_feats, how="left")
            logger.info("Merged explicit price features: +%d columns", len(price_feats.columns))
    except Exception as e:
        logger.debug("No explicit prices passed: %s", e)

    # Option B: ENTSO-E prices (primary source)
    if "price_eur_mwh" not in result.columns and ENTSOE_AVAILABLE:
        try:
            entsoe_price_feats = build_entsoe_price_features(
                zone, start_date, end_date,
                eur_nok_hourly=fx_hourly_series,
            )
            if not entsoe_price_feats.empty:
                result = result.join(entsoe_price_feats, how="left")
                logger.info("Merged ENTSO-E price features: +%d columns", len(entsoe_price_feats.columns))
        except Exception as e:
            logger.warning("ENTSO-E price features failed: %s", e)

    # Option C: Nord Pool prices (fallback if ENTSO-E unavailable)
    if "price_eur_mwh" not in result.columns and NORDPOOL_AVAILABLE:
        try:
            nordpool_feats = build_nordpool_price_features(
                zone, start_date, end_date,
                eur_nok_hourly=fx_hourly_series,
            )
            if not nordpool_feats.empty:
                result = result.join(nordpool_feats, how="left")
                logger.info("Merged Nord Pool price features (fallback): +%d columns", len(nordpool_feats.columns))
        except Exception as e:
            logger.warning("Nord Pool price features unavailable: %s", e)

    # NOK price lag/rolling/diff features (if NOK base columns are available)
    if "price_nok_mwh" in result.columns:
        try:
            nok_feats = build_nok_price_features(result["price_nok_mwh"])
            if not nok_feats.empty:
                result = result.join(nok_feats, how="left")
                logger.info("Merged NOK price features: +%d columns", len(nok_feats.columns))
        except Exception as e:
            logger.error("Failed to build NOK price features: %s", e)

    # --- ENTSO-E hourly sources (load, generation, flows) ---

    if ENTSOE_AVAILABLE:
        # ENTSO-E Load (per-zone consumption)
        try:
            entsoe_load = build_entsoe_load_features(zone, start_date, end_date)
            if not entsoe_load.empty:
                result = result.join(entsoe_load, how="left")
                logger.info("Merged ENTSO-E load: +%d columns", len(entsoe_load.columns))
        except Exception as e:
            logger.warning("ENTSO-E load features failed: %s", e)

        # ENTSO-E Generation (per-zone production mix)
        try:
            entsoe_gen = build_entsoe_generation_features(zone, start_date, end_date)
            if not entsoe_gen.empty:
                result = result.join(entsoe_gen, how="left")
                logger.info("Merged ENTSO-E generation: +%d columns", len(entsoe_gen.columns))
        except Exception as e:
            logger.warning("ENTSO-E generation features failed: %s", e)

        # ENTSO-E Cross-border flows (per-cable flows + foreign prices)
        try:
            entsoe_flows = build_entsoe_flow_features(
                zone, start_date, end_date,
                eur_nok_hourly=fx_hourly_series,
            )
            if not entsoe_flows.empty:
                result = result.join(entsoe_flows, how="left")
                logger.info("Merged ENTSO-E flows: +%d columns", len(entsoe_flows.columns))
        except Exception as e:
            logger.warning("ENTSO-E flow features failed: %s", e)

    # --- Daily sources (resample to hourly) ---

    # Commodities
    try:
        commodities = build_commodity_features(start_date, end_date)
        if not commodities.empty:
            commodities_hourly = _resample_to_hourly(commodities, hourly_index)
            result = result.join(commodities_hourly, how="left")
            logger.info("Merged commodities: +%d columns", len(commodities.columns))
    except Exception as e:
        logger.error("Failed to build commodity features: %s", e)

    # Statnett
    try:
        statnett = build_statnett_features(start_date, end_date)
        if not statnett.empty:
            statnett_hourly = _resample_to_hourly(statnett, hourly_index)
            result = result.join(statnett_hourly, how="left")
            logger.info("Merged Statnett: +%d columns", len(statnett.columns))
    except Exception as e:
        logger.error("Failed to build Statnett features: %s", e)

    # --- Weekly source (resample to hourly) ---

    # Reservoir
    try:
        reservoir = build_reservoir_features(zone, start_date, end_date)
        if not reservoir.empty:
            reservoir_hourly = _resample_to_hourly(reservoir, hourly_index)
            result = result.join(reservoir_hourly, how="left")
            logger.info("Merged reservoir: +%d columns", len(reservoir.columns))
    except Exception as e:
        logger.error("Failed to build reservoir features: %s", e)

    # --- Final cleanup ---

    # Forward-fill remaining gaps from resampling
    result = result.ffill()

    # Log summary
    n_missing = result.isna().sum()
    total_missing = n_missing.sum()
    logger.info(
        "Feature matrix: %d rows x %d columns, %d total missing values",
        len(result), len(result.columns), total_missing,
    )
    if total_missing > 0:
        for col, count in n_missing[n_missing > 0].items():
            logger.info("  %s: %d missing (%.1f%%)", col, count, 100 * count / len(result))

    # Cache to processed/
    cache_dir = PROJECT_ROOT / "data" / "processed"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"features_{zone}_{start_date}_{end_date}.parquet"
    result.to_parquet(cache_path)
    logger.info("Cached feature matrix to %s", cache_path)

    return result


# ---------------------------------------------------------------------------
# 10. All-zones orchestrator
# ---------------------------------------------------------------------------

ALL_ZONES = ["NO_1", "NO_2", "NO_3", "NO_4", "NO_5"]

ZONE_LABELS = {
    "NO_1": "Øst-Norge (Oslo)",
    "NO_2": "Sør-Norge (Kristiansand)",
    "NO_3": "Midt-Norge (Trondheim)",
    "NO_4": "Nord-Norge (Tromsø)",
    "NO_5": "Vest-Norge (Bergen)",
}


def build_all_zones_feature_matrix(
    start_date: str,
    end_date: str,
    prices_dict: dict[str, pd.Series] | None = None,
) -> dict[str, pd.DataFrame]:
    """Build feature matrices for all 5 Norwegian bidding zones.

    Calls build_feature_matrix() for each zone. Skips zones that fail
    (e.g., missing weather data) and logs a warning.

    Args:
        start_date: Start date as "YYYY-MM-DD".
        end_date: End date as "YYYY-MM-DD".
        prices_dict: Optional dict mapping zone names to hourly price Series.

    Returns:
        Dict mapping zone names to feature DataFrames.
        E.g., {"NO_1": df1, "NO_2": df2, ...}
    """
    results = {}

    for zone in ALL_ZONES:
        logger.info("Building features for %s (%s)", zone, ZONE_LABELS[zone])
        try:
            prices = prices_dict.get(zone) if prices_dict else None
            results[zone] = build_feature_matrix(zone, start_date, end_date, prices=prices)
        except Exception as e:
            logger.error("Failed to build features for %s: %s", zone, e)

    logger.info(
        "Built feature matrices for %d/%d zones: %s",
        len(results), len(ALL_ZONES), list(results.keys()),
    )
    return results


# ---------------------------------------------------------------------------
# 11. Visualization
# ---------------------------------------------------------------------------

def plot_feature_summary(df: pd.DataFrame, zone: str) -> None:
    """Generate a multi-panel summary visualization of the feature matrix.

    Creates 8 subplots showing time series, distributions, correlations,
    and data quality. Saves to artifacts/feature_summary_{zone}.png.

    Args:
        df: Feature matrix DataFrame (output of build_feature_matrix).
        zone: Bidding zone name (for titles and filename).
    """
    import matplotlib
    import matplotlib.pyplot as plt
    matplotlib.use("Agg")  # Non-interactive backend for saving plots

    artifacts_dir = PROJECT_ROOT / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(4, 2, figsize=(18, 22))
    fig.suptitle(f"Feature Summary — {zone}", fontsize=16, fontweight="bold", y=0.98)

    # --- Panel 1: Weather time series ---
    ax = axes[0, 0]
    if "temperature" in df.columns:
        # Downsample for readability
        daily = df[["temperature"]].resample("D").mean()
        ax.plot(daily.index, daily["temperature"], color="tab:red", linewidth=0.8, label="Temperature")
        ax.set_ylabel("Temperature (°C)", color="tab:red")
        ax.tick_params(axis="y", labelcolor="tab:red")

        if "wind_speed" in df.columns:
            ax2 = ax.twinx()
            daily_wind = df[["wind_speed"]].resample("D").mean()
            ax2.plot(daily_wind.index, daily_wind["wind_speed"], color="tab:blue", linewidth=0.8, alpha=0.7, label="Wind")
            ax2.set_ylabel("Wind speed (m/s)", color="tab:blue")
            ax2.tick_params(axis="y", labelcolor="tab:blue")

        ax.set_title("Weather: Temperature & Wind Speed (daily avg)")
    else:
        ax.text(0.5, 0.5, "No weather data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Weather")
    ax.grid(True, alpha=0.3)

    # --- Panel 2: Commodity prices ---
    ax = axes[0, 1]
    has_commodity = False
    if "ttf_gas_close" in df.columns:
        daily_gas = df[["ttf_gas_close"]].resample("D").last().dropna()
        ax.plot(daily_gas.index, daily_gas["ttf_gas_close"], color="tab:orange", linewidth=0.8, label="TTF Gas")
        ax.set_ylabel("TTF Gas (EUR/MWh)", color="tab:orange")
        ax.tick_params(axis="y", labelcolor="tab:orange")
        has_commodity = True

        if "brent_oil_close" in df.columns:
            ax2 = ax.twinx()
            daily_oil = df[["brent_oil_close"]].resample("D").last().dropna()
            ax2.plot(daily_oil.index, daily_oil["brent_oil_close"], color="tab:green", linewidth=0.8, alpha=0.7, label="Brent Oil")
            ax2.set_ylabel("Brent Oil (USD/bbl)", color="tab:green")
            ax2.tick_params(axis="y", labelcolor="tab:green")

    if has_commodity:
        ax.set_title("Commodity Prices: TTF Gas & Brent Oil")
    else:
        ax.text(0.5, 0.5, "No commodity data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Commodity Prices")
    ax.grid(True, alpha=0.3)

    # --- Panel 3: Reservoir filling ---
    ax = axes[1, 0]
    if "reservoir_filling_pct" in df.columns:
        daily_res = df[["reservoir_filling_pct"]].resample("D").last().dropna()
        ax.plot(daily_res.index, daily_res["reservoir_filling_pct"], color="tab:cyan", linewidth=1.2, label="Filling %")
        ax.set_ylabel("Filling (%)")
        ax.set_ylim(0, 105)

        if "reservoir_vs_median" in df.columns:
            daily_med = df[["reservoir_vs_median"]].resample("D").last().dropna()
            ax3 = ax.twinx()
            ax3.bar(daily_med.index, daily_med["reservoir_vs_median"], width=7, alpha=0.3, color="tab:purple", label="vs Median")
            ax3.set_ylabel("Deviation from median (%)", color="tab:purple")
            ax3.tick_params(axis="y", labelcolor="tab:purple")

        ax.set_title(f"Reservoir Filling — {zone}")
    else:
        ax.text(0.5, 0.5, "No reservoir data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Reservoir Filling")
    ax.grid(True, alpha=0.3)

    # --- Panel 4: Statnett balance ---
    ax = axes[1, 1]
    if "production_mwh" in df.columns and "consumption_mwh" in df.columns:
        daily_pc = df[["production_mwh", "consumption_mwh"]].resample("D").last().dropna()
        ax.bar(daily_pc.index, daily_pc["production_mwh"] / 1000, width=3, alpha=0.6, color="tab:green", label="Production")
        ax.bar(daily_pc.index, -daily_pc["consumption_mwh"] / 1000, width=3, alpha=0.6, color="tab:red", label="Consumption")
        ax.axhline(y=0, color="black", linewidth=0.5)
        ax.set_ylabel("GWh/day")
        ax.legend(loc="upper right", fontsize=8)
        ax.set_title("Statnett: Production vs Consumption")
    else:
        ax.text(0.5, 0.5, "No Statnett data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Statnett Balance")
    ax.grid(True, alpha=0.3)

    # --- Panel 5: EUR/NOK exchange rate ---
    ax = axes[2, 0]
    if "eur_nok" in df.columns:
        daily_fx = df[["eur_nok"]].resample("D").last().dropna()
        ax.plot(daily_fx.index, daily_fx["eur_nok"], color="tab:brown", linewidth=0.8, label="EUR/NOK")
        # 30-day moving average
        ma30 = daily_fx["eur_nok"].rolling(30, min_periods=1).mean()
        ax.plot(daily_fx.index, ma30, color="black", linewidth=1.2, alpha=0.7, label="30d MA")
        ax.set_ylabel("EUR/NOK")
        ax.legend(loc="upper left", fontsize=8)
        ax.set_title("EUR/NOK Exchange Rate")
    else:
        ax.text(0.5, 0.5, "No FX data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("EUR/NOK Exchange Rate")
    ax.grid(True, alpha=0.3)

    # --- Panel 6: Calendar patterns (hour x day_of_week heatmap) ---
    ax = axes[2, 1]
    if "hour_of_day" in df.columns and "temperature" in df.columns:
        # Pivot: average temperature by hour and day-of-week
        pivot = df.pivot_table(
            values="temperature", index="day_of_week", columns="hour_of_day", aggfunc="mean"
        )
        day_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlBu_r")
        ax.set_yticks(range(7))
        ax.set_yticklabels(day_labels)
        ax.set_xticks(range(0, 24, 3))
        ax.set_xlabel("Hour of day")
        plt.colorbar(im, ax=ax, label="Avg temperature (°C)")
        ax.set_title("Calendar: Avg Temperature by Hour & Day")
    else:
        ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Calendar Patterns")

    # --- Panel 7: Correlation heatmap ---
    ax = axes[3, 0]
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) > 2:
        # Sample to speed up correlation calculation
        sample = df[numeric_cols].sample(n=min(10000, len(df)), random_state=42)
        corr = sample.corr()

        im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
        ax.set_xticks(range(len(numeric_cols)))
        ax.set_yticks(range(len(numeric_cols)))
        ax.set_xticklabels(numeric_cols, rotation=90, fontsize=6)
        ax.set_yticklabels(numeric_cols, fontsize=6)
        plt.colorbar(im, ax=ax, label="Correlation")
        ax.set_title("Feature Correlation Matrix")
    else:
        ax.text(0.5, 0.5, "Not enough numeric features", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Correlation Matrix")

    # --- Panel 8: Missing data ---
    ax = axes[3, 1]
    missing_pct = (df.isna().sum() / len(df) * 100).sort_values(ascending=True)
    if missing_pct.sum() > 0:
        missing_nonzero = missing_pct[missing_pct > 0]
        if len(missing_nonzero) > 0:
            bars = ax.barh(range(len(missing_nonzero)), missing_nonzero.values, color="tab:red", alpha=0.7)
            ax.set_yticks(range(len(missing_nonzero)))
            ax.set_yticklabels(missing_nonzero.index, fontsize=7)
            ax.set_xlabel("Missing %")
            ax.set_title(f"Missing Data ({len(missing_nonzero)} features with gaps)")
        else:
            ax.text(0.5, 0.5, "No missing data!", ha="center", va="center",
                    transform=ax.transAxes, fontsize=14, color="green")
            ax.set_title("Missing Data")
    else:
        ax.text(0.5, 0.5, "No missing data!", ha="center", va="center",
                transform=ax.transAxes, fontsize=14, color="green")
        ax.set_title("Missing Data")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    save_path = artifacts_dir / f"feature_summary_{zone}.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    logger.info("Saved feature summary plot to %s", save_path)
    print(f"Visualization saved to {save_path}")


# ---------------------------------------------------------------------------
# Entry point for testing
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    start = "2020-01-01"
    end = "2026-02-22"

    print(f"Building feature matrices for all zones ({start} to {end})...\n")

    all_dfs = build_all_zones_feature_matrix(start, end)

    for zone, df in all_dfs.items():
        print(f"\n{'='*60}")
        print(f"FEATURE MATRIX SUMMARY — {zone} ({ZONE_LABELS[zone]})")
        print(f"{'='*60}")
        print(f"Shape: {df.shape[0]:,} rows x {df.shape[1]} columns")
        print(f"Date range: {df.index.min()} to {df.index.max()}")
        print(f"\nColumns ({len(df.columns)}):")
        for col in df.columns:
            missing = df[col].isna().sum()
            pct = 100 * missing / len(df)
            status = f"  {missing:>6,} missing ({pct:.1f}%)" if missing > 0 else "  OK"
            print(f"  {col:<35s}{status}")

        # Generate visualization
        print(f"\nGenerating visualization...")
        plot_feature_summary(df, zone)

    print(f"\nDone! Built features for {len(all_dfs)} zones.")
