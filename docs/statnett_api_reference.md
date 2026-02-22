# Statnett Driftsdata REST API Reference

## Overview

Statnett (Norway's Transmission System Operator / TSO) provides real-time and historical
operational data for the Nordic power system via an open REST API. No authentication required.

**Base URL:** `https://driftsdata.statnett.no/restapi`

**No API key needed.** Open data, JSON responses.

**Python package:** `pip install statnett-api-client` (convenience wrapper, may be outdated)

## Relevance for Our Project

| Endpoint | Use Case | Priority | Overlaps With |
|----------|----------|----------|---------------|
| PhysicalFlow | Cable arbitrage — real physical flows per cable | ⭐⭐⭐ HIGH | ENTSO-E crossborder flows |
| ProductionConsumption | Nordic-wide production/consumption by type | ⭐⭐ MEDIUM | ENTSO-E generation/load |
| Reservoir | Reservoir data (NVE proxy) | ⭐ LOW | NVE API (already have) |
| Frequency | Grid frequency (50Hz deviations) | ⭐ OPTIONAL | Unique — not in ENTSO-E |
| ElspotPowerSituation | Statnett's own power situation assessment | ⭐⭐ MEDIUM | Unique signal |
| Keyfigures | Historical key figures since 1974 | ⭐ CONTEXT | Good for long-term analysis |
| Download | Bulk CSV download by year | ⭐⭐ MEDIUM | Easier than API pagination |

## Endpoints

### 1. PhysicalFlow — Physical Cross-Border Flows ⭐⭐⭐

```
GET /restapi/Physicalflow/GetData?From={date}
```

**Parameters:**
- `From` — Start date (format: `2024-01-01`)

Returns physical power flows across all Nordic interconnectors.
This is the **key endpoint for cable arbitrage analysis** — shows actual MW flowing
on each cable, which can be compared against price differentials.

**Advantage over ENTSO-E:** Statnett is the actual operator of the Norwegian cables,
so this is the most authoritative source. May have higher resolution or more detail
on individual cable flows.

### 2. ProductionConsumption — Nordic Power Balance ⭐⭐

#### GetData (historical)
```
GET /restapi/ProductionConsumption/GetData?From={date}
```

Returns historical production and consumption data per country (NO, SE, DK, FI, EE, LT, LV).

#### GetLatestDetailedOverview (real-time snapshot)
```
GET /restapi/ProductionConsumption/GetLatestDetailedOverview
```

Returns current production breakdown by type and country:

**Response structure (from live data):**

```
Countries: SE, DK, NO, FI, EE, LT, LV
Categories:
  - ProductionData     — Total production (MW)
  - NuclearData        — Nuclear generation (SE, FI)
  - HydroData          — Hydro generation (NO dominates)
  - ThermalData        — Thermal/gas/coal generation
  - WindData           — Wind generation
  - NotSpecifiedData   — Other/unclassified
  - ConsumptionData    — Total consumption (MW)
  - NetExchangeData    — Net import/export (MW)
```

**Live example (fetched 2026-02-22):**

| Country | Production | Hydro | Wind | Thermal | Nuclear | Consumption | Net Exchange |
|---------|-----------|-------|------|---------|---------|-------------|-------------|
| NO | 15,566 | 13,508 | 1,842 | 166 | - | 20,200 | 4,634 import |
| SE | 20,685 | 10,286 | 1,944 | 3 | 7,060 | 18,273 | -2,412 export |
| DK | 3,374 | 1 | 1,710 | 1,501 | - | 5,493 | 2,119 import |
| FI | 10,132 | 1,476 | 1,951 | 1,951 | 4,234 | 11,631 | 1,499 import |

**Note:** This confirms Norway was a NET IMPORTER (4,634 MW) at that moment despite
having 15,566 MW production — consumption (20,200 MW) exceeded production.

#### Download (bulk CSV by year)
```
GET /restapi/download?datasource=productionconsumption&year={year}
```

Downloads a full year of production/consumption data as CSV. Available from 2012.
**This is the easiest way to get historical data for training.**

### 3. Reservoir — Reservoir Data ⭐

```
GET /restapi/Reservoir/LastWeekData/{weeks}    — Last N weeks
GET /restapi/Reservoir/                         — Graph data
GET /restapi/Reservoir/NveProxy                 — Proxied NVE data
```

**Note:** This is a proxy to NVE data. Since we already have direct NVE API access
(docs/nve_magasin_api_reference.md), use NVE directly for reservoir data.
Statnett Reservoir endpoint is useful only as a fallback.

### 4. Frequency — Grid Frequency ⭐ (Optional/Advanced)

#### By Second (real-time)
```
GET /restapi/Frequency/BySecond?From={date}
```

Returns grid frequency measurements per second. Target: 50.000 Hz.
Deviations indicate supply/demand imbalance.

**Response format:**
```json
{
  "StartPointUTC": 1771062744000,  // Unix timestamp (ms)
  "EndPointUTC": 1771062804000,
  "PeriodTickMs": 1000,            // 1 measurement per second
  "Measurements": [49.967, 49.968, 49.968, ...]
}
```

#### By Minute
```
GET /restapi/Frequency/ByMinute?From={date}
```

Same format, 1-minute resolution.

**Use case for our project:** Grid frequency deviations are an advanced feature.
Large deviations from 50Hz indicate sudden imbalances (generator trips, cable outages).
Could be useful for anomaly detection (Phase 6), but low priority for price forecasting.

### 5. Reserves — Balancing Reserves ⭐

```
GET /restapi/Reserves/PrimaryReservesPerDay?localDateTime={date}
GET /restapi/Reserves/PrimaryReservesPerWeek?years={year}
GET /restapi/Reserves/SecondaryReservesPerWeek?years={year}
```

Primary and secondary reserves procurement data. Shows how much balancing capacity
was procured — higher reserves may indicate expected volatility.

### 6. ElspotPowerSituation — Statnett's Assessment ⭐⭐

```
GET /restapi/ElspotPowerSituation/GetPowerSituations/
GET /restapi/ElspotPowerSituation/LastChanges/
```

Statnett's own assessment of the power situation. This is a qualitative/semi-quantitative
signal that Statnett publishes — could be a useful categorical feature for the model.

### 7. ElspotSeparatorLine — Price Zone Boundaries ⭐

```
GET /restapi/ElspotSeparatorLine/LastChanges/
GET /restapi/ElspotSeparatorLine/AsPng/?country=NO
```

Shows when elspot zone boundaries change. Historical changes are rare but important
as they affect which zone a generator/consumer belongs to.

### 8. Systemprice — Current System Price

```
GET /restapi/Systemprice/
```

Nord Pool system price for the current hour. Useful for live dashboard,
but for historical prices use ENTSO-E or Nord Pool directly.

### 9. Keyfigures — Historical Key Figures since 1974

```
GET /restapi/Keyfigures
```

Long-term historical key figures for the Norwegian/Nordic power system.
Useful for context and long-term trend analysis.

### 10. Rkom — Regulation Power Market Messages

```
GET /restapi/Rkom/Year/{year}
GET /restapi/Rkom/MetaData
```

Regulation power market (RKOM) messages and procurement data.
Shows balancing market activation — indicates real-time stress in the system.

## Recommended Integration Strategy

### Phase 1 (Data Foundation) — Add to fetch pipeline:

**fetch_statnett.py** with these functions:

```python
import requests
import pandas as pd

BASE_URL = "https://driftsdata.statnett.no/restapi"

def fetch_physical_flows(from_date: str = "2017-01-01") -> pd.DataFrame:
    """Fetch physical cross-border flows from Statnett.
    Complements/replaces ENTSO-E cross-border flow data.
    """
    url = f"{BASE_URL}/Physicalflow/GetData?From={from_date}"
    response = requests.get(url)
    response.raise_for_status()
    return pd.DataFrame(response.json())

def fetch_production_consumption(from_date: str = "2017-01-01") -> pd.DataFrame:
    """Fetch historical production and consumption by country and type."""
    url = f"{BASE_URL}/ProductionConsumption/GetData?From={from_date}"
    response = requests.get(url)
    response.raise_for_status()
    return pd.DataFrame(response.json())

def fetch_latest_overview() -> dict:
    """Fetch real-time Nordic power balance snapshot."""
    url = f"{BASE_URL}/ProductionConsumption/GetLatestDetailedOverview"
    response = requests.get(url)
    response.raise_for_status()
    return response.json()

def download_yearly_csv(year: int) -> pd.DataFrame:
    """Download bulk CSV for a given year (from 2012)."""
    url = f"{BASE_URL}/download?datasource=productionconsumption&year={year}"
    response = requests.get(url)
    response.raise_for_status()
    # Parse CSV content
    from io import StringIO
    return pd.read_csv(StringIO(response.text), sep=';')

def fetch_frequency_by_minute(from_date: str) -> pd.DataFrame:
    """Fetch grid frequency data (1-minute resolution)."""
    url = f"{BASE_URL}/Frequency/ByMinute?From={from_date}"
    response = requests.get(url)
    response.raise_for_status()
    data = response.json()
    # Convert Unix timestamps to datetime
    import numpy as np
    start = pd.Timestamp(data['StartPointUTC'], unit='ms', tz='UTC')
    measurements = data['Measurements']
    index = pd.date_range(start, periods=len(measurements), freq='min')
    return pd.DataFrame({'frequency_hz': measurements}, index=index)

def fetch_power_situation() -> dict:
    """Fetch Statnett's power situation assessment."""
    url = f"{BASE_URL}/ElspotPowerSituation/GetPowerSituations/"
    response = requests.get(url)
    response.raise_for_status()
    return response.json()
```

### Phase priorities:

1. **PhysicalFlow** → Cable arbitrage analysis (Phase 6b)
   - Compare Statnett physical flows vs ENTSO-E — use whichever has better granularity
   - Statnett is the TSO, so this is the "ground truth" for Norwegian cables

2. **Download CSV** → Bulk historical data (Phase 1)
   - Easier than API pagination for production/consumption backfill
   - Years 2012–2025 available

3. **ProductionConsumption** → Nordic balance features (Phase 2)
   - Swedish nuclear, Danish wind, Finnish thermal — all affect Norwegian prices
   - Net exchange per country = directional flow signal

4. **Frequency** → Advanced anomaly detection (Phase 6+)
   - Sudden frequency drops indicate generator/cable trips
   - Could explain sudden price spikes

5. **ElspotPowerSituation** → Categorical feature (Phase 4+)
   - Statnett's own risk assessment — unique signal

## Statnett vs ENTSO-E Comparison

| Data Type | Statnett | ENTSO-E |
|-----------|----------|---------|
| Physical flows | ✅ TSO source, authoritative | ✅ Good, pan-European |
| Production/consumption | ✅ Nordic countries, per type | ✅ All Europe, per type |
| Prices | ❌ Only system price (current hour) | ✅ Day-ahead per zone, historical |
| Reservoir | ⚠️ NVE proxy (use NVE directly) | ⚠️ Norway total only |
| Grid frequency | ✅ Per-second resolution | ❌ Not available |
| Balancing reserves | ✅ RKOM data | ⚠️ Limited |
| Power situation | ✅ Qualitative assessment | ❌ Not available |
| Historical depth | 2012+ | 2015+ |
| Authentication | None | API key required |

**Recommendation:** Use both. ENTSO-E for prices and European-wide data.
Statnett for physical flows (authoritative), frequency, and Nordic-specific signals.

## Data Notes

- All timestamps appear to use Unix millisecond format in JSON responses
- Real-time data updates every ~15 minutes (aligned with ENTSO-E 15-min periods)
- Historical data available from 2012 via download endpoint
- No documented rate limits, but be respectful — cache aggressively
- Response format is JSON (some endpoints also support XML via content negotiation)
- The `statnett-api-client` Python package exists on PyPI but is old (v0.1.5, 2019)
  — consider using it for convenience or implementing directly with requests