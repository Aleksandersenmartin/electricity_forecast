# ENTSO-E Transparency Platform — API Reference

Complete reference for the ENTSO-E REST API as used in this project.
Covers all endpoints relevant for Norwegian electricity price forecasting.

API base URL: `https://web-api.tp.entsoe.eu/api`
Python wrapper: `entsoe-py` ([GitHub](https://github.com/EnergieID/entsoe-py))
Official docs: [Postman collection](https://documenter.getpostman.com/view/7009892/2s93JtP3F6)
Knowledge base: [Transparency Platform Zendesk](https://transparencyplatform.zendesk.com/hc/en-us/sections/12783116987028-Web-API)

---

## Table of Contents

1. [Authentication](#1-authentication)
2. [Request Format](#2-request-format)
3. [Norwegian Bidding Zones](#3-norwegian-bidding-zones)
4. [Data Domains Overview](#4-data-domains-overview)
5. [Load / Consumption](#5-load--consumption)
6. [Day-Ahead Prices](#6-day-ahead-prices)
7. [Generation](#7-generation)
8. [Reservoir Filling](#8-reservoir-filling)
9. [Cross-Border Flows & Transmission](#9-cross-border-flows--transmission)
10. [Wind & Solar Forecasts](#10-wind--solar-forecasts)
11. [DocumentType Codes](#11-documenttype-codes)
12. [ProcessType Codes](#12-processtype-codes)
13. [PSR Type Codes (Production Types)](#13-psr-type-codes-production-types)
14. [Business Type Codes](#14-business-type-codes)
15. [entsoe-py Method Reference](#15-entsoe-py-method-reference)
16. [Rate Limits & Best Practices](#16-rate-limits--best-practices)
17. [Known Issues & Data Quality](#17-known-issues--data-quality)
18. [Neighbouring Zones & Country Codes](#18-neighbouring-zones--country-codes)

---

## 1. Authentication

**Registration:**
1. Create account at https://transparency.entsoe.eu/
2. Password: min 14 characters, at least one special character (not letter, digit, or '@')
3. Email transparency@entsoe.eu with subject "Restful API access" and your registered email in body
4. Wait 1–3 business days
5. Generate token under "My Account Settings" → "Web API Security Token"

**Usage:**
- REST API: pass as query parameter `securityToken=YOUR_TOKEN`
- entsoe-py: pass to client constructor

```python
from entsoe import EntsoePandasClient
client = EntsoePandasClient(api_key=os.getenv("ENTSOE_API_KEY"))
```

---

## 2. Request Format

### REST API (raw)

```
GET https://web-api.tp.entsoe.eu/api
    ?documentType=A65
    &processType=A16
    &outBiddingZone_Domain=10YNO-1--------2
    &periodStart=202401010000
    &periodEnd=202401312300
    &securityToken=YOUR_TOKEN
```

**Date format:** `YYYYMMDDHHMM` (UTC)
**Response format:** XML (default) or JSON

### entsoe-py (recommended)

```python
import pandas as pd

start = pd.Timestamp("2024-01-01", tz="Europe/Oslo")
end = pd.Timestamp("2024-01-31", tz="Europe/Oslo")

# entsoe-py handles date formatting, pagination, and XML parsing
result = client.query_load("NO_1", start=start, end=end)
```

**entsoe-py advantages:**
- Automatic date format conversion
- Handles multi-year requests by splitting into chunks
- Parses XML into pandas Series/DataFrame
- Manages pagination for large responses
- Maps country/zone codes to EIC codes

---

## 3. Norwegian Bidding Zones

| Zone | Area | EIC Code | `entsoe-py` key | Major city |
|------|------|----------|-----------------|------------|
| NO1 | Øst-Norge | `10YNO-1--------2` | `NO_1` | Oslo |
| NO2 | Sør-Norge | `10YNO-2--------T` | `NO_2` | Kristiansand |
| NO3 | Midt-Norge | `10YNO-3--------J` | `NO_3` | Trondheim |
| NO4 | Nord-Norge | `10YNO-4--------9` | `NO_4` | Tromsø |
| NO5 | Vest-Norge | `10Y1001A1001A48H` | `NO_5` | Bergen |
| NO (all) | Norge | `10YNO-0--------C` | `NO` | — |

**Note:** `NO` (all of Norway) can be used for some queries (e.g., reservoir filling),
but most data is reported per bidding zone (NO1–NO5).

---

## 4. Data Domains Overview

ENTSO-E organizes data into domains. The ones relevant for this project:

| Domain | What | Key data items |
|--------|------|---------------|
| **Load** | Electricity consumption | Actual total load, load forecast |
| **Generation** | Electricity production | Actual generation per type, generation forecast, installed capacity |
| **Transmission** | Cross-border flows | Physical flows, scheduled exchanges, net transfer capacity |
| **Balancing** | Market balancing | Imbalance prices, imbalance volumes |
| **Outages** | Planned/unplanned outages | Generation unavailability, transmission unavailability |

### Priority for price forecasting (our use)

| Priority | Data | Why it matters |
|----------|------|---------------|
| 🔴 Must have | Day-ahead prices | Target variable |
| 🔴 Must have | Actual total load | Demand drives price |
| 🔴 Must have | Actual generation per type | Supply mix affects price |
| 🟡 Important | Load forecast (day-ahead) | Market prices expectations |
| 🟡 Important | Reservoir filling | Hydro availability dominates Norwegian prices |
| 🟡 Important | Cross-border flows | Import/export affects supply-demand balance |
| 🟡 Important | Wind & solar forecast | Renewable output suppresses prices |
| 🟢 Nice to have | Generation forecast | Expected total supply |
| 🟢 Nice to have | Installed generation capacity | Structural baseline |
| 🟢 Nice to have | Imbalance prices | Real-time market signal |

---

## 5. Load / Consumption

### Actual Total Load [6.1.A]

Actual electricity consumption per bidding zone per market time unit (hourly).
Defined as: total generation on TSO/DSO networks minus exchange balance minus storage absorption.

**REST API:**
```
documentType=A65
processType=A16         # A16 = Realised
outBiddingZone_Domain=10YNO-1--------2
```

**entsoe-py:**
```python
# Returns: pd.DataFrame with columns ['Actual Load']
load = client.query_load("NO_1", start=start, end=end)
```

**Unit:** MW
**Resolution:** Hourly (transitioning to 15-min in 2025)
**Available from:** ~2015 for Norway

### Day-Ahead Total Load Forecast [6.1.B]

Forecasted consumption published day-ahead.

**REST API:**
```
documentType=A65
processType=A01         # A01 = Day ahead
outBiddingZone_Domain=10YNO-1--------2
```

**entsoe-py:**
```python
# Returns: pd.DataFrame with columns ['Forecasted Load']
load_forecast = client.query_load_forecast("NO_1", start=start, end=end)
```

### Combined Load & Forecast

```python
# Utility method: merges actual load and day-ahead forecast into one DataFrame
load_combined = client.query_load_and_forecast("NO_1", start=start, end=end)
```

### Other Load Forecasts

| Forecast horizon | processType | entsoe-py method |
|-----------------|-------------|-----------------|
| Day ahead | A01 | `query_load_forecast()` |
| Week ahead | A31 | `query_load_forecast(process_type='A31')` |
| Month ahead | A32 | `query_load_forecast(process_type='A32')` |
| Year ahead | A33 | `query_load_forecast(process_type='A33')` |

---

## 6. Day-Ahead Prices

### Day-Ahead Prices [12.1.D]

Day-ahead electricity market prices per bidding zone.

**REST API:**
```
documentType=A44
in_Domain=10YNO-1--------2
out_Domain=10YNO-1--------2
```

**entsoe-py:**
```python
# Returns: pd.Series (EUR/MWh)
prices = client.query_day_ahead_prices("NO_1", start=start, end=end)
```

**Unit:** EUR/MWh
**Resolution:** Hourly (transitioning to 15-min in 2025)
**Published:** Around 12:42 CET daily
**Available from:** 2015 for Norway

**Note:** Prices are in EUR. Use Norges Bank FX rate (see fetch_fx.py) to convert to NOK.
Conversion: `price_nok_kwh = price_eur_mwh * eur_nok_rate / 1000`

---

## 7. Generation

### Actual Generation per Type [16.1.B&C]

Actual electricity generation broken down by production type (hydro, wind, gas, etc.).

**REST API:**
```
documentType=A75
processType=A16         # A16 = Realised
in_Domain=10YNO-1--------2
```

**entsoe-py:**
```python
# Returns: pd.DataFrame with one column per production type
# e.g., 'Hydro Water Reservoir', 'Wind Onshore', 'Fossil Gas', etc.
generation = client.query_generation("NO_1", start=start, end=end, psr_type=None)

# Filter by specific type:
hydro = client.query_generation("NO_1", start=start, end=end, psr_type='B12')
```

**Unit:** MW
**Resolution:** Hourly

### Norwegian Generation Mix (typical)

Norway's generation is dominated by hydropower (~90%):

| PSR Code | Type | Norway relevance |
|----------|------|-----------------|
| B12 | Hydro Water Reservoir | ~70% of Norwegian generation |
| B11 | Hydro Run-of-river and poundage | ~20% |
| B10 | Hydro Pumped Storage | Small but growing |
| B19 | Wind Onshore | ~5% and growing |
| B04 | Fossil Gas | Small (emergency/peak) |
| B01 | Biomass | Very small |
| B16 | Solar | Very small but growing |
| B18 | Wind Offshore | Emerging |

### Generation Forecast [14.1.C]

Day-ahead generation forecast.

**entsoe-py:**
```python
gen_forecast = client.query_generation_forecast("NO_1", start=start, end=end)
```

### Installed Generation Capacity per Type [14.1.A]

Total installed capacity per production type. Changes slowly (yearly).

**entsoe-py:**
```python
capacity = client.query_installed_generation_capacity("NO_1", start=start, end=end, psr_type=None)
```

### Actual Generation per Plant [16.1.A]

Generation from individual power plants (>100 MW installed capacity only).

**entsoe-py:**
```python
# Returns large DataFrame — one column per plant
per_plant = client.query_generation_per_plant("NO_1", start=start, end=end, psr_type=None)
```

**Note:** Only covers plants >100 MW. Not all Norwegian plants are included.

---

## 8. Reservoir Filling

### Reservoir Filling [16.1.D] — KEY for Norway

Water reservoir filling levels as a percentage. This is one of the **strongest price
drivers** in Norway's hydro-dominated system.

**REST API:**
```
documentType=A72
processType=A16
in_Domain=10YNO-0--------C      # Note: uses NO (all Norway), not per zone
```

**entsoe-py:**
```python
# Note: typically only available for whole Norway, not per bidding zone
# Returns: pd.Series (percentage filling)
reservoir = client.query_reservoir_filling("NO", start=start, end=end)
```

**Unit:** Percentage (%)
**Resolution:** Weekly
**Available from:** ~2015

**Why it matters:**
- High reservoir → cheap electricity (abundant hydro supply)
- Low reservoir → expensive electricity (scarcity premium)
- Seasonal pattern: fills in spring/summer (snowmelt), drains in winter
- Deviations from normal filling are a strong price signal

**Limitation:** Reported for all of Norway, not per bidding zone.
For zone-level reservoir data, check NVE (Norwegian Water Resources and Energy Directorate):
https://www.nve.no/energi/analyser-og-statistikk/magasinstatistikk/

---

## 9. Cross-Border Flows & Transmission

### Physical Cross-Border Flows [12.1.G]

Actual measured power flow between two areas.

**entsoe-py:**
```python
# Flow from NO1 to SE3 (Sweden zone 3)
flow = client.query_crossborder_flows("NO_1", "SE_3", start=start, end=end)

# Flow from NO2 to DK1 (Denmark zone 1) — NorNed/Skagerrak cables
flow = client.query_crossborder_flows("NO_2", "DK_1", start=start, end=end)

# All export flows from a zone
exports = client.query_physical_crossborder_allborders("NO_1", start=start, end=end, export=True)

# All import flows to a zone
imports = client.query_physical_crossborder_allborders("NO_1", start=start, end=end, export=False)
```

**Unit:** MW (positive = flow in direction specified)

### Key Norwegian Interconnections

| From | To | Cable/Connection | Capacity (MW) |
|------|----|-----------------|---------------|
| NO1 | SE3 | AC interconnectors | ~3,500 |
| NO2 | DK1 | Skagerrak 1-4 | ~1,700 |
| NO2 | NL | NorNed | 700 |
| NO2 | DE | NordLink | 1,400 |
| NO2 | GB | North Sea Link | 1,400 |
| NO3 | SE2 | AC interconnectors | ~1,000 |
| NO4 | SE1 | AC interconnectors | ~700 |
| NO4 | SE2 | AC interconnectors | ~300 |
| NO4 | FI | AC interconnectors | ~100 |
| NO1 | NO2 | Internal (Flesaker corridor) | varies |
| NO1 | NO3 | Internal | varies |
| NO1 | NO5 | Internal (Hallingdal corridor) | varies |
| NO2 | NO5 | Internal | varies |
| NO3 | NO4 | Internal | varies |
| NO3 | NO5 | Internal | varies |

**Why it matters:**
- Export from Norway → less supply locally → higher domestic price
- Import to Norway → more supply → lower domestic price
- Cable outages can cause significant price divergence between zones
- NO2 has most international cables → most exposed to continental prices

### Scheduled Commercial Exchanges [12.1.F]

Day-ahead scheduled exchanges (commercial, not physical).

```python
scheduled = client.query_scheduled_exchanges("NO_2", "DK_1", start=start, end=end)
```

### Net Transfer Capacity [12.1.H]

Day-ahead available transfer capacity between zones.

```python
ntc = client.query_net_transfer_capacity_dayahead("NO_2", "DK_1", start=start, end=end)
```

---

## 10. Wind & Solar Forecasts

### Wind & Solar Generation Forecast [14.1.D]

Day-ahead forecast for renewable generation.

**entsoe-py:**
```python
# Returns DataFrame with wind and solar forecast columns
ws_forecast = client.query_wind_and_solar_forecast("NO_1", start=start, end=end, psr_type=None)

# Filter specific type:
wind_forecast = client.query_wind_and_solar_forecast("NO_1", start=start, end=end, psr_type='B19')
```

### Intraday Wind & Solar Forecast

Updated forecast published closer to delivery.

```python
ws_intraday = client.query_intraday_wind_and_solar_forecast("NO_1", start=start, end=end, psr_type=None)
```

---

## 11. DocumentType Codes

Complete list of document types used in ENTSO-E API.
Codes marked with ✅ are used in this project.

| Code | Description | Used |
|------|-------------|------|
| A09 | Finalised schedule | |
| A11 | Aggregated energy data report | |
| A15 | Acquiring system operator reserve schedule | |
| A24 | Bid document | |
| A25 | Allocation result document | |
| A26 | Capacity document | |
| A31 | Agreed capacity | |
| A38 | Reserve allocation result document | |
| **A44** | **Price Document** | ✅ |
| A61 | Estimated Net Transfer Capacity | |
| A63 | Redispatch notice | |
| **A65** | **System total load** | ✅ |
| **A68** | **Installed generation per type** | ✅ |
| **A69** | **Wind and solar forecast** | ✅ |
| A70 | Load forecast margin | |
| **A71** | **Generation forecast** | ✅ |
| **A72** | **Reservoir filling information** | ✅ |
| A73 | Actual generation | |
| A74 | Wind and solar generation | |
| **A75** | **Actual generation per type** | ✅ |
| A76 | Load unavailability | |
| A77 | Production unavailability | |
| A78 | Transmission unavailability | |
| A79 | Offshore grid infrastructure unavailability | |
| A80 | Generation unavailability | |
| A81 | Contracted reserves | |
| A82 | Accepted offers | |
| A83 | Activated balancing quantities | |
| A84 | Activated balancing prices | |
| A85 | Imbalance prices | |
| A86 | Imbalance volume | |
| A87 | Financial situation | |
| **A88** | **Cross-border flows** | ✅ |
| A91 | Agreed capacity — explicit allocations | |
| A92 | Agreed capacity — implicit allocations | |
| A93 | Agreed capacity — resale | |
| A94 | Net position | |
| A95 | Congestion costs | |
| B01 | Day-ahead wholesale market | |
| B06 | Aggregated netted external TSO schedule | |
| B07 | Bid availability | |
| B08 | CGMES Document | |
| B09 | Aggregated fills | |
| B11 | Procurement capacity | |

---

## 12. ProcessType Codes

| Code | Description | Used with |
|------|-------------|-----------|
| **A01** | **Day ahead** | Load forecast, generation forecast |
| A02 | Intra day incremental | Intraday updates |
| A16 | **Realised / Actual** | Actual load, actual generation |
| A18 | Intraday total | |
| A31 | **Week ahead** | Week-ahead load forecast |
| A32 | **Month ahead** | Month-ahead load forecast |
| A33 | **Year ahead** | Year-ahead load forecast |
| A39 | Synchronisation process | |
| A40 | Intraday process | |
| A46 | Replacement reserve | |
| A47 | Manual frequency restoration reserve | |
| A51 | Automatic frequency restoration reserve | |
| A52 | Frequency containment reserve | |

---

## 13. PSR Type Codes (Production Types)

PSR = Power System Resource. Used with generation queries (documentType A73, A75).

### All PSR Types

| Code | Type | Norway? |
|------|------|---------|
| A03 | Mixed | |
| A04 | Generation | |
| A05 | Load | |
| B01 | Biomass | ⚡ Small |
| B02 | Fossil Brown coal/Lignite | ❌ |
| B03 | Fossil Coal-derived gas | ❌ |
| **B04** | **Fossil Gas** | ⚡ Small (peak/emergency) |
| B05 | Fossil Hard coal | ❌ |
| B06 | Fossil Oil | ⚡ Very small |
| B07 | Fossil Oil shale | ❌ |
| B08 | Fossil Peat | ❌ |
| B09 | Geothermal | ❌ |
| **B10** | **Hydro Pumped Storage** | ⚡ Small but growing |
| **B11** | **Hydro Run-of-river and poundage** | ✅ ~20% of generation |
| **B12** | **Hydro Water Reservoir** | ✅ ~70% of generation |
| B13 | Marine | ❌ |
| B14 | Nuclear | ❌ |
| B15 | Other renewable | ⚡ Small |
| **B16** | **Solar** | ⚡ Small but growing |
| B17 | Waste | ⚡ Very small |
| **B18** | **Wind Offshore** | ⚡ Emerging |
| **B19** | **Wind Onshore** | ✅ ~5% and growing |
| B20 | Other | |
| B21 | AC link | |
| B22 | DC link | |
| B23 | Substation | |
| B24 | Transformer | |

### Norwegian Generation Mix (typical share)

```
Hydro Water Reservoir (B12) ████████████████████████████████████  ~70%
Hydro Run-of-river (B11)   ██████████                            ~20%
Wind Onshore (B19)          ██▌                                   ~5%
Fossil Gas (B04)            █                                     ~2%
Other (B01, B16, etc.)      ▌                                     ~3%
```

---

## 14. Business Type Codes

Used as filter parameter in some queries.

| Code | Description |
|------|-------------|
| A01 | Production |
| A04 | Consumption |
| A14 | Aggregated consumption |
| A19 | Balance energy deviation |
| A25 | General capacity information |
| A29 | Already allocated capacity (AAC) |
| A93 | Imbalance |
| A95 | Contracted capacity |
| A96 | Maximum NTC |
| A97 | Manual frequency restoration reserve |
| B08 | Total nominated capacity |
| B09 | Net transfer capacity — nominated |
| B10 | Net transfer capacity — available |
| B11 | Already allocated capacity — AAC |
| C22 | Shared balancing reserve capacity |
| C24 | Actual reserve capacity |

---

## 15. entsoe-py Method Reference

### Methods returning pd.Series

| Method | documentType | Description |
|--------|-------------|-------------|
| `query_day_ahead_prices(zone, start, end)` | A44 | Day-ahead prices (EUR/MWh) |
| `query_load(zone, start, end)` | A65 + A16 | Actual total load (MW) |
| `query_load_forecast(zone, start, end)` | A65 + A01 | Day-ahead load forecast (MW) |
| `query_generation_forecast(zone, start, end)` | A71 | Total generation forecast (MW) |

### Methods returning pd.DataFrame

| Method | documentType | Description |
|--------|-------------|-------------|
| `query_load_and_forecast(zone, start, end)` | A65 | Combined actual + forecast load |
| `query_generation(zone, start, end, psr_type)` | A75 | Actual generation per type (MW) |
| `query_installed_generation_capacity(zone, start, end, psr_type)` | A68 | Installed capacity per type (MW) |
| `query_generation_per_plant(zone, start, end, psr_type)` | A73 | Generation per plant (MW) |
| `query_wind_and_solar_forecast(zone, start, end, psr_type)` | A69 | Wind/solar forecast (MW) |
| `query_intraday_wind_and_solar_forecast(zone, start, end, psr_type)` | A69 | Intraday renewable forecast |
| `query_crossborder_flows(zone_from, zone_to, start, end)` | A88 | Physical flows (MW) |
| `query_scheduled_exchanges(zone_from, zone_to, start, end)` | A09 | Scheduled commercial exchanges |
| `query_net_transfer_capacity_dayahead(zone_from, zone_to, start, end)` | A61 | Day-ahead NTC (MW) |
| `query_physical_crossborder_allborders(zone, start, end, export)` | A88 | All flows for a zone |
| `query_imbalance_prices(zone, start, end, psr_type)` | A85 | Imbalance prices |
| `query_reservoir_filling(zone, start, end)` | A72 | Reservoir filling (%) |

### Common parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `country_code` / zone | str | Zone code, e.g., `"NO_1"` |
| `start` | pd.Timestamp | Start time (must be tz-aware) |
| `end` | pd.Timestamp | End time (must be tz-aware) |
| `psr_type` | str or None | PSR type code (e.g., `"B12"` for hydro). `None` = all types |

### Example: fetch everything for one zone

```python
from entsoe import EntsoePandasClient
import pandas as pd
import os

client = EntsoePandasClient(api_key=os.getenv("ENTSOE_API_KEY"))

start = pd.Timestamp("2024-01-01", tz="Europe/Oslo")
end = pd.Timestamp("2024-01-31", tz="Europe/Oslo")
zone = "NO_1"

# Prices
prices = client.query_day_ahead_prices(zone, start=start, end=end)

# Load / consumption
actual_load = client.query_load(zone, start=start, end=end)
forecast_load = client.query_load_forecast(zone, start=start, end=end)

# Generation
generation = client.query_generation(zone, start=start, end=end, psr_type=None)
gen_forecast = client.query_generation_forecast(zone, start=start, end=end)

# Wind & solar
ws_forecast = client.query_wind_and_solar_forecast(zone, start=start, end=end, psr_type=None)

# Reservoir (whole Norway only)
reservoir = client.query_reservoir_filling("NO", start=start, end=end)

# Cross-border
flow_to_se3 = client.query_crossborder_flows("NO_1", "SE_3", start=start, end=end)
```

---

## 16. Rate Limits & Best Practices

### Rate limits

- ENTSO-E does not publish official rate limits
- In practice: ~400 requests per minute seems safe
- Large date ranges may time out — split into yearly chunks
- Add `time.sleep(1-2)` between requests to be safe

### Caching strategy

```python
import os
import time

def fetch_with_cache(zone, query_func, start, end, label):
    """Generic cache wrapper for ENTSO-E queries."""
    cache_path = f"data/raw/entsoe_{label}_{zone}_{start.year}.parquet"

    if os.path.exists(cache_path):
        return pd.read_parquet(cache_path)

    data = query_func(zone, start=start, end=end)
    data.to_frame().to_parquet(cache_path)
    time.sleep(2)
    return data
```

### Chunked fetching

For 8 years of data, fetch in yearly chunks:

```python
for year in range(2017, 2025):
    start = pd.Timestamp(f"{year}-01-01", tz="Europe/Oslo")
    end = pd.Timestamp(f"{year}-12-31 23:00", tz="Europe/Oslo")

    for zone in ["NO_1", "NO_2", "NO_3", "NO_4", "NO_5"]:
        prices = client.query_day_ahead_prices(zone, start=start, end=end)
        # ... save to parquet ...
        time.sleep(2)
```

### Data format tips

- Always use timezone-aware timestamps: `pd.Timestamp("2024-01-01", tz="Europe/Oslo")`
- Store as Parquet (preserves dtypes and timezones)
- entsoe-py returns data indexed by datetime — keep this index
- Some methods return Series, others DataFrames — check per method
- `query_generation()` returns MultiIndex columns (Actual Aggregated / Actual Consumption per type)

---

## 17. Known Issues & Data Quality

### General

- **Data gaps:** Some hours may be missing, especially for older data (pre-2017)
- **Inconsistencies:** ENTSO-E "Actual Total Load" can deviate >10% from other sources (Eurostat, TSO websites) for some countries. Norway is generally good quality.
- **Reporting delay:** Actual data may be delayed by hours or days
- **15-minute transition:** EU markets transitioning from 60-min to 15-min resolution in 2025. Design data pipeline to handle both.

### Norway-specific

- **Reservoir filling** only available for all of Norway, not per bidding zone. For zone-level data, check NVE.
- **Generation per plant** only covers plants >100 MW — many smaller hydro plants not included.
- **Wind/solar forecast** may return empty for some Norwegian zones (limited installed capacity historically).
- **NO5 (Bergen)** EIC code (`10Y1001A1001A48H`) is different format from NO1–NO4 — entsoe-py handles this, but watch out if using raw API.

### HTTP Error Codes

| Code | Meaning | What to do |
|------|---------|-----------|
| 200 | Success | Parse response |
| 400 | Invalid parameter | Check parameter names/values |
| 401 | Unauthorized | Check API token |
| 403 | Forbidden | Token may be expired or rate-limited |
| 404 | No data found | Data may not exist for this zone/period |
| 409 | Too many requests | Slow down, add sleep |
| 500 | Server error | Retry after delay |

---

## 18. Neighbouring Zones & Country Codes

Useful for cross-border flow queries.

### Nordic countries

| Country | Zone | EIC Code | `entsoe-py` key |
|---------|------|----------|-----------------|
| Sweden 1 | SE1 (Luleå) | `10Y1001A1001A44P` | `SE_1` |
| Sweden 2 | SE2 (Sundsvall) | `10Y1001A1001A45N` | `SE_2` |
| Sweden 3 | SE3 (Stockholm) | `10Y1001A1001A46L` | `SE_3` |
| Sweden 4 | SE4 (Malmö) | `10Y1001A1001A47J` | `SE_4` |
| Denmark 1 | DK1 (West) | `10YDK-1--------W` | `DK_1` |
| Denmark 2 | DK2 (East) | `10YDK-2--------M` | `DK_2` |
| Finland | FI | `10YFI-1--------U` | `FI` |

### Other connected countries

| Country | EIC Code | `entsoe-py` key | Connection to Norway |
|---------|----------|-----------------|---------------------|
| Netherlands | `10YNL----------L` | `NL` | NorNed cable (NO2) |
| Germany/Lux | `10Y1001A1001A83F` | `DE_LU` | NordLink cable (NO2) |
| Great Britain | `10Y1001A1001A92E` | `GB` | North Sea Link (NO2) |

---

*This document is a reference for the project. Update as you discover new endpoints
or data quality issues. See also: entsoe-py source code at
https://github.com/EnergieID/entsoe-py/blob/master/entsoe/mappings.py for the
complete list of codes and mappings.*