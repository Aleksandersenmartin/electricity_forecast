"""Cable arbitrage analysis for cross-border electricity flows.

Detects wrong-direction flows (power flowing from expensive to cheap zone),
computes arbitrage revenue, and identifies suspicious trading patterns
on Norwegian interconnectors.

Usage:
    from src.anomaly.cable_arbitrage import (
        compute_cable_spreads,
        detect_wrong_direction_flows,
        compute_daily_arbitrage_revenue,
        build_cable_analysis,
    )
"""

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Cable capacities in MW (approximate, based on public data)
CABLE_CAPACITIES: dict[str, int] = {
    "NO_2_DK_1": 1700,     # Skagerrak 1-4
    "NO_2_NL": 700,        # NorNed
    "NO_2_DE_LU": 1400,    # NordLink
    "NO_2_GB": 1400,       # North Sea Link
    "NO_1_SE_3": 2145,     # Multiple cables
    "NO_3_SE_2": 1000,     # Nea-Järpströmmen etc.
    "NO_4_SE_1": 700,      # Northern cables
    "NO_4_SE_2": 300,      # Small interconnector
    "NO_4_FI": 100,        # Pasvik
}

# Map cable key to (Norwegian zone, Foreign zone) for entsoe-py calls
CABLE_ZONES: dict[str, tuple[str, str]] = {
    "NO_2_DK_1": ("NO_2", "DK_1"),
    "NO_2_NL": ("NO_2", "NL"),
    "NO_2_DE_LU": ("NO_2", "DE_LU"),
    "NO_2_GB": ("NO_2", "GB"),
    "NO_1_SE_3": ("NO_1", "SE_3"),
    "NO_3_SE_2": ("NO_3", "SE_2"),
    "NO_4_SE_1": ("NO_4", "SE_1"),
    "NO_4_SE_2": ("NO_4", "SE_2"),
    "NO_4_FI": ("NO_4", "FI"),
}


def compute_cable_spreads(
    no_prices: pd.Series,
    foreign_prices: pd.Series,
    flows: pd.Series,
    cable_capacity: int,
    threshold_eur: float = 2.0,
) -> pd.DataFrame:
    """Compute price spreads, flow analysis, and arbitrage metrics for a cable.

    Args:
        no_prices: Norwegian zone prices (EUR/MWh), hourly DatetimeIndex.
        foreign_prices: Foreign zone prices (EUR/MWh), hourly DatetimeIndex.
        flows: Cross-border flows (MW), positive = export from Norway.
        cable_capacity: Cable capacity in MW.
        threshold_eur: Minimum spread to flag wrong-direction (EUR/MWh).

    Returns:
        DataFrame with columns: no_price, foreign_price, spread, flow,
        wrong_direction, flow_spread_ratio, capacity_utilization,
        hourly_arbitrage_eur.
    """
    # Align all series to common index
    combined = pd.DataFrame({
        "no_price": no_prices,
        "foreign_price": foreign_prices,
        "flow": flows,
    }).dropna()

    if combined.empty:
        logger.warning("No overlapping data for cable spread computation")
        return pd.DataFrame()

    # Price spread: positive = Norway more expensive
    combined["spread"] = combined["no_price"] - combined["foreign_price"]

    # Wrong-direction flow detection
    # Exporting (flow > 0) when Norway is more expensive (spread > threshold)
    export_wrong = (combined["spread"] > threshold_eur) & (combined["flow"] > 0)
    # Importing (flow < 0) when Norway is cheaper (spread < -threshold)
    import_wrong = (combined["spread"] < -threshold_eur) & (combined["flow"] < 0)
    combined["wrong_direction"] = export_wrong | import_wrong

    # Flow-spread ratio (large flow + small spread = suspicious)
    combined["flow_spread_ratio"] = (
        combined["flow"].abs() / (combined["spread"].abs() + 0.01)
    )

    # Capacity utilization
    combined["capacity_utilization"] = combined["flow"].abs() / cable_capacity

    # Hourly arbitrage value in EUR
    # Negative = money flowing the wrong way (exporter losing money)
    combined["hourly_arbitrage_eur"] = combined["spread"] * combined["flow"]

    return combined


def detect_wrong_direction_flows(
    spreads_df: pd.DataFrame,
    threshold_eur: float = 2.0,
) -> pd.DataFrame:
    """Extract and rank wrong-direction flow events.

    Args:
        spreads_df: Output of compute_cable_spreads().
        threshold_eur: Minimum spread magnitude for flagging.

    Returns:
        DataFrame of wrong-direction events sorted by absolute EUR impact.
    """
    if spreads_df.empty or "wrong_direction" not in spreads_df.columns:
        return pd.DataFrame()

    wrong = spreads_df[spreads_df["wrong_direction"]].copy()

    if wrong.empty:
        return pd.DataFrame()

    wrong["abs_impact_eur"] = wrong["hourly_arbitrage_eur"].abs()
    wrong = wrong.sort_values("abs_impact_eur", ascending=False)

    return wrong


def compute_daily_arbitrage_revenue(
    spreads_df: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate hourly arbitrage to daily revenue.

    Args:
        spreads_df: Output of compute_cable_spreads().

    Returns:
        DataFrame with daily total_arbitrage_eur, wrong_direction_hours,
        wrong_direction_pct, mean_spread.
    """
    if spreads_df.empty:
        return pd.DataFrame()

    daily = spreads_df.resample("D").agg({
        "hourly_arbitrage_eur": "sum",
        "wrong_direction": "sum",
        "spread": "mean",
        "flow": "mean",
        "capacity_utilization": "mean",
    })

    daily = daily.rename(columns={
        "hourly_arbitrage_eur": "total_arbitrage_eur",
        "wrong_direction": "wrong_direction_hours",
        "spread": "mean_spread",
        "flow": "mean_flow",
        "capacity_utilization": "mean_utilization",
    })

    # Hours with data per day
    hours_per_day = spreads_df.resample("D").size()
    daily["wrong_direction_pct"] = (
        daily["wrong_direction_hours"] / hours_per_day * 100
    ).fillna(0)

    return daily


def build_cable_analysis(
    zone: str,
    start_date: str,
    end_date: str,
    cache: bool = True,
) -> dict[str, dict[str, Any]]:
    """Build complete cable analysis for a Norwegian zone.

    Fetches prices, flows, and computes arbitrage metrics for all
    international cables connected to the given zone.

    Args:
        zone: Norwegian zone (e.g., "NO_2").
        start_date: Start date (YYYY-MM-DD).
        end_date: End date (YYYY-MM-DD).
        cache: Whether to use cached data.

    Returns:
        Dict mapping cable_key → {spreads_df, daily_df, wrong_events, stats}.
    """
    from src.data.fetch_electricity import (
        ZONE_CABLES,
        fetch_crossborder_flows,
        fetch_foreign_prices,
        fetch_prices,
    )

    # Get cables for this zone
    foreign_zones = ZONE_CABLES.get(zone, [])
    if not foreign_zones:
        logger.info("Zone %s has no international cables", zone)
        return {}

    # Fetch Norwegian zone prices
    try:
        no_prices_df = fetch_prices(zone, start_date, end_date, cache=cache)
        no_prices = no_prices_df.iloc[:, 0] if isinstance(no_prices_df, pd.DataFrame) else no_prices_df
    except Exception as e:
        logger.error("Failed to fetch %s prices: %s", zone, e)
        return {}

    results: dict[str, dict[str, Any]] = {}

    for foreign_zone in foreign_zones:
        cable_key = f"{zone}_{foreign_zone}"
        capacity = CABLE_CAPACITIES.get(cable_key, 1000)

        logger.info("Analyzing cable %s (capacity %d MW)", cable_key, capacity)

        try:
            # Fetch foreign prices
            foreign_df = fetch_foreign_prices(
                foreign_zone, start_date, end_date, cache=cache,
            )
            foreign_prices = (
                foreign_df.iloc[:, 0]
                if isinstance(foreign_df, pd.DataFrame)
                else foreign_df
            )

            # Fetch cross-border flows
            flows_df = fetch_crossborder_flows(
                zone, foreign_zone, start_date, end_date, cache=cache,
            )
            flows = (
                flows_df.iloc[:, 0]
                if isinstance(flows_df, pd.DataFrame)
                else flows_df
            )

            # Compute spreads and arbitrage
            spreads_df = compute_cable_spreads(
                no_prices, foreign_prices, flows, capacity,
            )

            if spreads_df.empty:
                logger.warning("No data for cable %s", cable_key)
                continue

            # Daily aggregation
            daily_df = compute_daily_arbitrage_revenue(spreads_df)

            # Wrong-direction events
            wrong_events = detect_wrong_direction_flows(spreads_df)

            # Summary statistics
            stats = {
                "cable_key": cable_key,
                "no_zone": zone,
                "foreign_zone": foreign_zone,
                "capacity_mw": capacity,
                "total_hours": len(spreads_df),
                "wrong_direction_hours": int(spreads_df["wrong_direction"].sum()),
                "wrong_direction_pct": round(
                    spreads_df["wrong_direction"].mean() * 100, 2,
                ),
                "mean_spread_eur": round(spreads_df["spread"].mean(), 2),
                "mean_flow_mw": round(spreads_df["flow"].mean(), 1),
                "mean_utilization": round(
                    spreads_df["capacity_utilization"].mean(), 3,
                ),
                "total_wrong_way_eur": round(
                    wrong_events["hourly_arbitrage_eur"].sum(), 0,
                ) if not wrong_events.empty else 0,
            }

            results[cable_key] = {
                "spreads_df": spreads_df,
                "daily_df": daily_df,
                "wrong_events": wrong_events,
                "stats": stats,
            }

            logger.info(
                "Cable %s: %d hours, %.1f%% wrong-direction",
                cable_key, stats["total_hours"], stats["wrong_direction_pct"],
            )

        except Exception as e:
            logger.warning("Failed to analyze cable %s: %s", cable_key, e)
            continue

    return results
