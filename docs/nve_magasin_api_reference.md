# NVE Magasinstatistikk API Reference

## Overview

NVE (Norges vassdrags- og energidirektorat) collects weekly water level measurements
from ~490 reservoirs to track the power situation in Norway. Total reservoir capacity
is approximately 87 TWh. Data is published every Wednesday at 13:00 CET.

**This is the authoritative source for Norwegian reservoir filling data — per elspot zone.**
ENTSO-E only has reservoir filling for all of Norway. NVE has it per zone (NO1–NO5).

## Base URL

```
https://biapi.nve.no/magasinstatistikk
```

## Authentication

**None required.** Open data, no API key needed.

Licensed under NLOD (Norsk lisens for offentlige data), compatible with CC BY 3.0.

## Endpoints

### 1. HentOffentligData — Full Historical Dataset

```
GET /api/Magasinstatistikk/HentOffentligData
```

Returns ALL weekly reservoir data since 1995. ~13,000+ rows.
No parameters — returns everything in one call.

**Response model: `MagasinstatistikkModel[]`**

```json
[
  {
    "dato_Id": "2024-01-07T00:00:00",    // Date identifier
    "omrType": "EL",                       // Area type: "EL" = elspot, "NO" = country, "VA" = vassdrag
    "omrnr": 1,                            // Area number (1=NO1, 2=NO2, 3=NO3, 4=NO4, 5=NO5)
    "iso_aar": 2024,                       // ISO year
    "iso_uke": 1,                          // ISO week number (1–53)
    "fyllingsgrad": 0.723,                 // Filling degree (0.0–1.0) — THIS IS THE KEY FEATURE
    "kapasitet_TWh": 11.54,               // Zone capacity in TWh
    "fylling_TWh": 8.34,                  // Current filling in TWh
    "neste_Publiseringsdato": "2024-01-17T13:00:00",  // Next publication date
    "fyllingsgrad_forrige_uke": 0.741,    // Previous week filling degree
    "endring_fyllingsgrad": -0.018        // Week-over-week change
  }
]
```

**Filtering by area type (`omrType`):**

| omrType | Description | omrnr values |
|---------|-------------|-------------|
| `NO` | Hele Norge (country total) | 0 |
| `EL` | Elspot-områder (price zones) | 1=NO1, 2=NO2, 3=NO3, 4=NO4, 5=NO5 |
| `VA` | Vassdragsområder (watershed areas) | 1–7 (older grouping) |

**For our project: Filter `omrType == "EL"` to get per-zone data.**

### 2. HentOffentligDataSisteUke — Latest Week Only

```
GET /api/Magasinstatistikk/HentOffentligDataSisteUke
```

Returns only the most recent week's data. Same response model as above.
Useful for real-time monitoring / dashboard updates.

### 3. HentOffentligDataMinMaxMedian — Historical Benchmarks

```
GET /api/Magasinstatistikk/HentOffentligDataMinMaxMedian
```

Returns min, max, and median filling for each week of the year,
based on the last 20 years. For seasonal comparison.

**Response model: `MagasinstatistikkOffentligMinMaxMedianModel[]`**

```json
[
  {
    "omrType": "EL",
    "omrnr": 1,                    // 1=NO1
    "iso_uke": 1,                  // Week 1
    "minFyllingsgrad": 0.45,       // Historical minimum filling (last 20 years)
    "minFyllingTWH": 5.19,
    "medianFyllingsGrad": 0.65,    // Historical median filling
    "medianFylling_TWH": 7.50,
    "maxFyllingsgrad": 0.82,       // Historical maximum filling
    "maxFyllingTWH": 9.47
  }
]
```

**Use case:** Compare current filling to historical norms.
If current filling < minFyllingsgrad → historically low → price signal.

### 4. HentOmråder — Area Definitions

```
GET /api/Magasinstatistikk/HentOmråder
```

Returns area definitions grouped by type.

**Response model: `CurrentAreas`**

```json
{
  "land": [
    { "navn": "Norge", "navn_langt": "Norge", "beskrivelse": "...", "omrType": "NO", "omrnr": 0 }
  ],
  "elspot": [
    { "navn": "NO1", "navn_langt": "Sørøst-Norge",  "beskrivelse": "...", "omrType": "EL", "omrnr": 1 },
    { "navn": "NO2", "navn_langt": "Sørvest-Norge",  "beskrivelse": "...", "omrType": "EL", "omrnr": 2 },
    { "navn": "NO3", "navn_langt": "Midt-Norge",     "beskrivelse": "...", "omrType": "EL", "omrnr": 3 },
    { "navn": "NO4", "navn_langt": "Nord-Norge",     "beskrivelse": "...", "omrType": "EL", "omrnr": 4 },
    { "navn": "NO5", "navn_langt": "Vest-Norge",     "beskrivelse": "...", "omrType": "EL", "omrnr": 5 }
  ],
  "vassdrag": [
    { "navn": "Område 1", "omrType": "VA", "omrnr": 1 },
    ...
  ]
}
```

## Elspot Zone Mapping

| omrnr | Zone | Name | Description | Capacity (approx) |
|-------|------|------|-------------|-------------------|
| 1 | NO1 | Sørøst-Norge | Eastern Norway (Buskerud northward) | ~12 TWh |
| 2 | NO2 | Sørvest-Norge | Vestfold, Telemark, Agder, Rogaland, southern Vestland | ~21 TWh |
| 3 | NO3 | Midt-Norge | Northern Vestland, Innlandet (west), Møre og Romsdal, Trøndelag | ~12 TWh |
| 4 | NO4 | Nord-Norge | Rest of Trøndelag and Northern Norway | ~15 TWh |
| 5 | NO5 | Vest-Norge | Central Vestland (up to Sognefjorden), western Buskerud | ~27 TWh |

**Note:** NO2 and NO5 together hold ~55% of Norway's reservoir capacity.
This explains why southern Norway price sensitivity to reservoir levels is so high.

## Python Implementation

```python
import requests
import pandas as pd

BASE_URL = "https://biapi.nve.no/magasinstatistikk"

def fetch_nve_reservoir_all() -> pd.DataFrame:
    """Fetch all historical reservoir data from NVE (since 1995)."""
    url = f"{BASE_URL}/api/Magasinstatistikk/HentOffentligData"
    response = requests.get(url)
    response.raise_for_status()
    data = response.json()
    df = pd.DataFrame(data)
    return df

def fetch_nve_reservoir_latest() -> pd.DataFrame:
    """Fetch latest week reservoir data."""
    url = f"{BASE_URL}/api/Magasinstatistikk/HentOffentligDataSisteUke"
    response = requests.get(url)
    response.raise_for_status()
    return pd.DataFrame(response.json())

def fetch_nve_reservoir_benchmarks() -> pd.DataFrame:
    """Fetch min/max/median benchmarks (last 20 years)."""
    url = f"{BASE_URL}/api/Magasinstatistikk/HentOffentligDataMinMaxMedian"
    response = requests.get(url)
    response.raise_for_status()
    return pd.DataFrame(response.json())

def fetch_nve_areas() -> dict:
    """Fetch area definitions."""
    url = f"{BASE_URL}/api/Magasinstatistikk/HentOmråder"
    response = requests.get(url)
    response.raise_for_status()
    return response.json()

# Filter for elspot zones only
df = fetch_nve_reservoir_all()
df_elspot = df[df['omrType'] == 'EL'].copy()

# Map omrnr to zone name
zone_map = {1: 'NO1', 2: 'NO2', 3: 'NO3', 4: 'NO4', 5: 'NO5'}
df_elspot['zone'] = df_elspot['omrnr'].map(zone_map)

# Parse date
df_elspot['date'] = pd.to_datetime(df_elspot['dato_Id'])

# Result: weekly filling per zone since 1995
# Columns: date, zone, fyllingsgrad, kapasitet_TWh, fylling_TWh, endring_fyllingsgrad
```

## Feature Engineering from NVE Data

For the forecasting model, extract these features per zone:

```python
# Weekly data → forward-fill to hourly for merge with price data
# (reservoir filling changes slowly, weekly granularity is fine)

# Feature 1: Filling degree (0–1)
reservoir_filling = fyllingsgrad  # Primary feature

# Feature 2: Week-over-week change
reservoir_filling_diff = endring_fyllingsgrad  # Already provided by API

# Feature 3: Deviation from historical median
# (merge with MinMaxMedian data)
reservoir_vs_median = fyllingsgrad - medianFyllingsGrad  # Positive = above normal

# Feature 4: Deviation from historical min/max
reservoir_vs_min = fyllingsgrad - minFyllingsgrad  # How far above record low
reservoir_vs_max = maxFyllingsgrad - fyllingsgrad  # How far below record high

# Feature 5: Filling in TWh (absolute, not just percentage)
reservoir_filling_twh = fylling_TWh

# Feature 6: Zone capacity utilization context
# NO2+NO5 filling is most impactful for southern Norway prices
reservoir_south = (fylling_TWh_NO2 + fylling_TWh_NO5) / (kapasitet_TWh_NO2 + kapasitet_TWh_NO5)
```

## Why NVE Instead of ENTSO-E for Reservoirs

| Aspect | ENTSO-E (A72) | NVE Magasinstatistikk |
|--------|--------------|----------------------|
| Geographic detail | Norway total only | Per elspot zone (NO1–NO5) |
| Update frequency | Weekly | Weekly (Wednesday 13:00) |
| Historical data | ~2015+ | Since 1995 |
| Additional fields | Just filling % | Capacity TWh, filling TWh, change, benchmarks |
| Authentication | API key required | No auth needed |
| Reliability | Sometimes missing | Official NVE source — very reliable |
| Min/max/median | Not available | Separate endpoint with 20-year benchmarks |

**Decision: Use NVE as primary reservoir source. Keep ENTSO-E as fallback only.**

## Data Quality Notes

- Data published every Wednesday at 13:00 CET
- NVE waits for >90% of measurements before publishing
- Filling may be recalculated slightly as late measurements arrive
- Capacity (TWh) is updated annually
- Historical data back to 1995 (elspot zone data back to 2016)
- Vassdrag (watershed) grouping differs from elspot zones — use `omrType == "EL"` only
- Negative `endring_fyllingsgrad` = drawdown (winter), positive = filling (spring/summer)

## Seasonal Patterns

Typical annual cycle for Norwegian reservoirs:
- **Week 15–25 (Apr–Jun):** Spring melt → rapid filling
- **Week 25–40 (Jun–Oct):** Peak filling, typically 80–95%
- **Week 40–52 (Oct–Dec):** Autumn drawdown begins
- **Week 1–15 (Jan–Apr):** Winter drawdown, lowest levels (~40–60%)

Price impact: Reservoir filling below median during drawdown season (Oct–Mar) 
signals potential scarcity → significant upward price pressure.

## Rate Limits

No documented rate limits, but:
- HentOffentligData returns ~13,000 rows — call once and cache
- HentOffentligDataSisteUke is lightweight — OK for frequent polling
- Be respectful — NVE is a government agency providing free data
- Cache data locally in Parquet files after first fetch

## Swagger Documentation

Interactive Swagger UI: https://biapi.nve.no/magasinstatistikk/swagger/index.html
OpenAPI spec: https://biapi.nve.no/magasinstatistikk/swagger/v1/swagger.json