# CommodityPriceAPI — API Reference

Complete reference for the CommodityPriceAPI REST API as used in this project.
Covers endpoints, authentication, energy-relevant symbols, JSON response formats,
and data fetching strategies for Norwegian electricity price forecasting.

API base URL: `https://api.commoditypriceapi.com/v2`
Website: [commoditypriceapi.com](https://commoditypriceapi.com)
Documentation: [commoditypriceapi.com/#documentation](https://commoditypriceapi.com/#documentation)
Symbols list: [commoditypriceapi.com/symbols](https://commoditypriceapi.com/symbols)

---

## Table of Contents

1. [Authentication](#1-authentication)
2. [Request Format & Base URL](#2-request-format--base-url)
3. [Endpoints Overview](#3-endpoints-overview)
4. [Latest Rates Endpoint](#4-latest-rates-endpoint)
5. [Historical Rates Endpoint](#5-historical-rates-endpoint)
6. [Timeseries Endpoint](#6-timeseries-endpoint)
7. [Fluctuation Endpoint](#7-fluctuation-endpoint)
8. [Symbols Endpoint](#8-symbols-endpoint)
9. [Usage Endpoint](#9-usage-endpoint)
10. [Energy Symbols — Relevant for This Project](#10-energy-symbols--relevant-for-this-project)
11. [All Supported Symbols (Complete List)](#11-all-supported-symbols-complete-list)
12. [Update Intervals & Data Frequency](#12-update-intervals--data-frequency)
13. [Quote Currencies](#13-quote-currencies)
14. [Error Codes & Handling](#14-error-codes--handling)
15. [Rate Limits & Subscription Plans](#15-rate-limits--subscription-plans)
16. [Data Fetching Strategy for This Project](#16-data-fetching-strategy-for-this-project)
17. [Python Implementation Patterns](#17-python-implementation-patterns)

---

## 1. Authentication

**Registration:**
1. Sign up at https://commoditypriceapi.com/auth/signup (no credit card required)
2. Get your API key from the Dashboard: https://commoditypriceapi.com/dashboard
3. Free trial available — unlimited for integrations

**Sending the API key (two methods):**

```
# Method 1: Query parameter
GET https://api.commoditypriceapi.com/v2/latest?apiKey=YOUR_KEY&symbols=NG-FUT

# Method 2: Request header (recommended — keeps key out of logs)
GET https://api.commoditypriceapi.com/v2/latest?symbols=NG-FUT
Header: x-api-key: YOUR_KEY
```

**Environment variable setup (.env):**
```
COMMODITY_API_KEY=your_api_key_here
```

**Security:** Never expose API key in client-side JavaScript or public repositories.

---

## 2. Request Format & Base URL

**Base URL:** `https://api.commoditypriceapi.com/v2`

**Date format:** `YYYY-MM-DD` (e.g., `2024-06-15`)

**Symbols format:** Comma-separated string (e.g., `NG-FUT,BRENTOIL-SPOT,TTF-GAS`)

**Common parameters across all endpoints:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `apiKey` | string | Yes* | Your API key (*or pass via `x-api-key` header) |
| `symbols` | string | Yes | Comma-separated list of commodity symbols |
| `quote` | string | No | Target currency code (e.g., `EUR`, `NOK`, `USD`). Premium/Plus plans only |

---

## 3. Endpoints Overview

| Endpoint | Method | Path | Description |
|----------|--------|------|-------------|
| **Latest** | GET | `/v2/latest` | Current/most recent rates |
| **Historical** | GET | `/v2/{date}` | Rates for a specific date (OHLC) |
| **Timeseries** | GET | `/v2/timeseries` | Daily rates between two dates (max 1 year) |
| **Fluctuation** | GET | `/v2/fluctuation` | Rate change between two dates |
| **Symbols** | GET | `/v2/symbols` | List all supported commodities |
| **Usage** | GET | `/v2/usage` | Account usage statistics |

---

## 4. Latest Rates Endpoint

Returns the most recent price for specified commodities.
Depending on subscription, rates may be delayed up to 10 minutes.

**Request:**
```
GET /v2/latest?apiKey=YOUR_KEY&symbols=NG-FUT,BRENTOIL-SPOT,TTF-GAS
```

**Optional:** `&quote=EUR` to get prices in EUR (Premium/Plus only)

**Response JSON:**
```json
{
  "success": true,
  "timestamp": 1708617600,
  "rates": {
    "NG-FUT": 1.875,
    "BRENTOIL-SPOT": 82.45,
    "TTF-GAS": 28.35
  },
  "metaData": {
    "NG-FUT": {
      "unit": "MMBtu",
      "currency": "USD"
    },
    "BRENTOIL-SPOT": {
      "unit": "Bbl",
      "currency": "USD"
    },
    "TTF-GAS": {
      "unit": "MWh",
      "currency": "EUR"
    }
  }
}
```

**Response fields:**

| Field | Type | Description |
|-------|------|-------------|
| `success` | boolean | `true` if request was successful |
| `timestamp` | number | Unix timestamp of response |
| `rates` | object | Current price for each requested symbol |
| `metaData` | object | Unit and default currency per symbol |

**Note:** Latest rates for deprecated symbols are not available. Check historical rates instead.

---

## 5. Historical Rates Endpoint

Returns OHLC (Open, High, Low, Close) prices for a specific date.
Historical data available from **1990-01-01** for most commodities.

**Request:**
```
GET /v2/2024-06-15?apiKey=YOUR_KEY&symbols=NG-FUT,BRENTOIL-SPOT,TTF-GAS
```

**Response JSON:**
```json
{
  "success": true,
  "date": "2024-06-15",
  "rates": {
    "NG-FUT": {
      "date": "2024-06-15",
      "open": 2.756,
      "high": 2.812,
      "low": 2.701,
      "close": 2.789
    },
    "BRENTOIL-SPOT": {
      "date": "2024-06-14",
      "open": 82.34,
      "high": 83.10,
      "low": 81.90,
      "close": 82.75
    },
    "TTF-GAS": {
      "date": "2024-06-15",
      "open": 34.50,
      "high": 35.20,
      "low": 34.10,
      "close": 34.85
    }
  }
}
```

**Response fields:**

| Field | Type | Description |
|-------|------|-------------|
| `success` | boolean | `true` if request was successful |
| `date` | string | Requested date (YYYY-MM-DD) |
| `rates.{SYMBOL}.date` | string | Actual date data is from (may differ if unavailable for requested date) |
| `rates.{SYMBOL}.open` | number | Opening price |
| `rates.{SYMBOL}.high` | number | Highest price of the day |
| `rates.{SYMBOL}.low` | number | Lowest price of the day |
| `rates.{SYMBOL}.close` | number | Closing price |

**Important notes:**
- If rate for a specific date is unavailable, the API returns the **most recent available rate** along with its corresponding date. Always check the `date` field inside each rate.
- For **monthly commodities** (World Bank/IMF sourced), only closing rates are available.
- Weekend/holiday dates will return the last trading day's data.

---

## 6. Timeseries Endpoint

Returns daily rates between two dates. Maximum date range: **1 year**.

**Request:**
```
GET /v2/timeseries?apiKey=YOUR_KEY&symbols=NG-FUT,TTF-GAS&startDate=2024-01-01&endDate=2024-06-30
```

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `symbols` | string | Yes | Comma-separated symbols |
| `startDate` | string | Yes | Start date (YYYY-MM-DD) |
| `endDate` | string | Yes | End date (YYYY-MM-DD) |

**Response JSON:**
```json
{
  "success": true,
  "startDate": "2024-01-01",
  "endDate": "2024-06-30",
  "rates": {
    "2024-01-02": {
      "NG-FUT": {
        "open": 2.551,
        "high": 2.612,
        "low": 2.498,
        "close": 2.589
      },
      "TTF-GAS": {
        "open": 32.10,
        "high": 33.40,
        "low": 31.80,
        "close": 33.15
      }
    },
    "2024-01-03": {
      "NG-FUT": {
        "open": 2.589,
        "high": 2.650,
        "low": 2.540,
        "close": 2.621
      },
      "TTF-GAS": {
        "open": 33.15,
        "high": 33.90,
        "low": 32.50,
        "close": 33.45
      }
    }
  }
}
```

**Constraints:**
- `endDate - startDate` must be ≤ 365 days
- `startDate` cannot be after `endDate`
- Monthly commodities return only closing rates

---

## 7. Fluctuation Endpoint

Shows how commodity prices changed between two dates — useful for quick trend analysis.

**Request:**
```
GET /v2/fluctuation?apiKey=YOUR_KEY&symbols=NG-FUT,BRENTOIL-SPOT&startDate=2024-01-01&endDate=2024-06-30
```

**Response JSON:**
```json
{
  "success": true,
  "startDate": "2024-01-01",
  "endDate": "2024-06-30",
  "rates": {
    "NG-FUT": {
      "startRate": 2.551,
      "endRate": 2.789,
      "change": 0.238,
      "changePercent": 9.33
    },
    "BRENTOIL-SPOT": {
      "startRate": 77.04,
      "endRate": 82.75,
      "change": 5.71,
      "changePercent": 7.41
    }
  }
}
```

**Response fields:**

| Field | Type | Description |
|-------|------|-------------|
| `rates.{SYMBOL}.startRate` | number | Rate at start date |
| `rates.{SYMBOL}.endRate` | number | Rate at end date |
| `rates.{SYMBOL}.change` | number | Absolute change (negative = decrease) |
| `rates.{SYMBOL}.changePercent` | number | Percentage change (negative = decrease) |

---

## 8. Symbols Endpoint

Returns complete information about all supported commodities.

**Request:**
```
GET /v2/symbols?apiKey=YOUR_KEY
```

**Response:** JSON object with metadata per symbol (name, category, unit, currency, update interval, exchange).

---

## 9. Usage Endpoint

Returns account usage statistics.

**Request:**
```
GET /v2/usage?apiKey=YOUR_KEY
```

**Response JSON:**
```json
{
  "plan": "lite",
  "quota": 1000,
  "used": 142
}
```

---

## 10. Energy Symbols — Relevant for This Project

These are the symbols most relevant for Norwegian electricity price forecasting,
since natural gas, oil, and coal prices are key drivers of European power prices.

### Primary Energy Symbols (use these)

| Symbol | Name | Unit | Currency | Interval | Why relevant |
|--------|------|------|----------|----------|-------------|
| **TTF-GAS** | TTF Gas (Dutch) | MWh | EUR | 1 min | 🔴 **#1 driver** — European gas benchmark, sets marginal power price |
| **NG-FUT** | Natural Gas Futures (US) | MMBtu | USD | 1 min | Global gas reference (Henry Hub) |
| **NG-SPOT** | Natural Gas Spot (US) | MMBtu | USD | 1 min | US spot reference |
| **BRENTOIL-SPOT** | Crude Oil Brent Spot | Barrel | USD | 1 min | North Sea oil benchmark, relevant for Norway |
| **BRENTOIL-FUT** | Crude Oil Brent Futures | Barrel | USD | 1 min | Brent futures |
| **COAL** | Coal | Metric Ton | USD | 1 min | Coal generation affects power price |
| **UXA** | Uranium | Pound | USD | 1 min | Nuclear base load factor |

### Secondary Energy Symbols (lower priority)

| Symbol | Name | Unit | Currency | Interval |
|--------|------|------|----------|----------|
| WTIOIL-SPOT | Crude Oil WTI Spot | Barrel | USD | 1 min |
| WTIOIL-FUT | Crude Oil WTI Futures | Barrel | USD | 1 min |
| NG-EU | Natural Gas, Europe (World Bank) | MMBtu | USD | Monthly |
| NG-US | Natural Gas, US (World Bank) | MMBtu | USD | Monthly |
| LNG | Liquefied Natural Gas, Japan | MMBtu | USD | Monthly |
| UK-GAS | UK Gas | Therm | USD | 1 min |
| HO-SPOT | Heating Oil Spot | Gallon | USD | 1 min |
| HO-FUT | Heating Oil Futures | Gallon | USD | 1 min |
| LGO | Gas Oil | 100 Tonnes | USD | 1 min |
| RB1COAL | Coal, South Africa (World Bank) | Metric Ton | USD | Monthly |
| COAL-AU | Coal, Australia (World Bank) | Metric Ton | USD | Monthly |
| DBLC1 | Crude Oil Dubai (World Bank) | Barrel | USD | Monthly |

### Why these matter for Norwegian electricity prices

Norway's electricity is ~90% hydro, but prices are determined by the European market:

1. **TTF Gas (TTF-GAS)** is the most important commodity for European power prices.
   Gas-fired power plants often set the marginal price. When gas is expensive,
   electricity becomes expensive — even in hydro-rich Norway (via cross-border cables).

2. **Brent Oil (BRENTOIL-SPOT/FUT)** is relevant because:
   - Norway is a major oil/gas producer — economic linkage
   - Oil price affects gas price (partial correlation)
   - Relevant for gas-indexed contracts

3. **Coal (COAL)** matters because coal power plants are marginal generators in
   Germany/Poland, affecting NO2 and NO5 prices via NordLink and Skagerrak cables.

4. **EU ETS carbon prices** (not available in this API) are also a major driver.
   Consider supplementing with another source for EU Emissions Allowance (EUA) prices.

### Missing: EU Carbon Prices (EUA)

CommodityPriceAPI does **not** include EU ETS carbon allowance prices.
This is a significant gap for electricity price forecasting.
Consider supplementing with:
- Ember (free): https://ember-energy.org/data/carbon-price-viewer/
- ICAP: https://icapcarbonaction.com/en/ets-prices
- Sandbag/Ember API
- European Energy Exchange (EEX) data

---

## 11. All Supported Symbols (Complete List)

### Energy (17 symbols)

| # | Symbol | Name | Unit | Currency | Interval |
|---|--------|------|------|----------|----------|
| 1 | NG-FUT | Natural Gas Futures | MMBtu | USD | 1 min |
| 4 | WTIOIL-FUT | Crude Oil WTI Futures | Barrel | USD | 1 min |
| 5 | BRENTOIL-SPOT | Crude Oil Brent Spot | Barrel | USD | 1 min |
| 6 | LGO | Gas Oil | 100 Tonnes | USD | 1 min |
| 8 | RB-SPOT | RBOB Gasoline Spot | Gallon | USD | 1 min |
| 30 | NG-SPOT | Natural Gas Spot | MMBtu | USD | 1 min |
| 33 | HO-SPOT | Heating Oil Spot | Gallon | USD | 1 min |
| 37 | UXA | Uranium | Pound | USD | 1 min |
| 39 | PROP | Propane | Gallon | USD | 1 min |
| 40 | METH | Methanol | Metric Ton | USD | 1 min |
| 41 | URAL-OIL | Ural Oil | Barrel | USD | 1 min |
| 42 | COAL | Coal | Metric Ton | USD | 1 min |
| 67 | TTF-GAS | TTF Gas | MWh | EUR | 1 min |
| 73 | UK-GAS | UK Gas | Therm | USD | 1 min |
| 75 | RB-FUT | RBOB Gasoline Futures | Gallon | USD | 1 min |
| 136 | WTIOIL-SPOT | Crude Oil WTI Spot | Barrel | USD | 1 min |
| 137 | BRENTOIL-FUT | Crude Oil Brent Futures | Barrel | USD | 1 min |
| 147 | HO-FUT | Heating Oil Futures | Gallon | USD | 1 min |

### Energy — Monthly (World Bank/IMF sourced)

| # | Symbol | Name | Unit | Currency | Interval |
|---|--------|------|------|----------|----------|
| 76 | DBLC1 | Crude Oil Dubai | Barrel | USD | Monthly |
| 77 | NG-EU | Natural Gas, Europe | MMBtu | USD | Monthly |
| 78 | RB1COAL | Coal, South Africa | Metric Ton | USD | Monthly |
| 79 | COAL-AU | Coal, Australia | Metric Ton | USD | Monthly |
| 80 | NG-US | Natural Gas, US | MMBtu | USD | Monthly |
| 98 | LNG | Liquefied Natural Gas, Japan | MMBtu | USD | Monthly |

### Metals (20 symbols)

| Symbol | Name | Unit | Currency |
|--------|------|------|----------|
| XAU | Gold | Troy Ounce | USD |
| XAG | Silver | Troy Ounce | USD |
| PL | Platinum | Troy Ounce | USD |
| PA | Palladium | Troy Ounce | USD |
| HG-SPOT / HG-FUT | Copper Spot/Futures | Pound | USD |
| AL-SPOT / AL-FUT | Aluminium Spot/Futures | Metric Ton | USD |
| LEAD-SPOT / LEAD-FUT | Lead Spot/Futures | Metric Ton | USD |
| NICKEL-SPOT / NICKEL-FUT | Nickel Spot/Futures | Metric Ton | USD |
| TIN | Tin | Metric Ton | USD |
| ZINC | Zinc | Metric Ton | USD |
| TIOC | Iron Ore 62% FE | Metric Ton | USD |
| STEEL | Steel | Metric Ton | CNY |
| HRC-STEEL | Hot-Rolled Coil Steel | Metric Ton | USD |
| LC | Lithium | Metric Ton | CNY |
| TITAN | Titanium | Kilogram | CNY |
| MG | Magnesium | Metric Ton | CNY |

### Agriculture (30+ symbols)

Includes: CORN, ZW-SPOT/FUT (Wheat), SOYBEAN-SPOT/FUT, OAT-SPOT/FUT, CT (Cotton),
CC (Cocoa), CA (Coffee Arabica), OJ (Orange Juice), RR-SPOT/FUT (Rough Rice),
CANOLA, PO (Palm Oil), ZL (Soybean Oil), ZM (Soybean Meal), SUNF (Sunflower Oil),
MILK, BUTTER, POTATO, CHE (Cheese), TEA, RSO (Rapeseed Oil), etc.

### Livestock (8 symbols)

Includes: SALMON, POUL (Poultry), BEEF, CHKN (Chicken), LAMB, EGGS-CH, EGGS-US,
PORK, FC1 (Feeder Cattle), LC1 (Cattle), LHOGS (Lean Hogs), SHRIMP

### Industrial (14 symbols)

Includes: ETHANOL, RUBBER, NAPHTHA, COB (Cobalt), XRH (Rhodium), POL (Polyethylene),
PVC, PYL (Polypropylene), SODASH, NDYM (Neodymium), TEL (Tellurium), GA (Gallium),
INDIUM, DIAPH (Diammonium Phosphate), UREA, BIT (Bitumen), K-PULP, LB-SPOT/FUT (Lumber)

---

## 12. Update Intervals & Data Frequency

Symbols have different update frequencies depending on their source:

| Interval | Description | Count | Examples |
|----------|-------------|-------|---------|
| **1 Second** | Near real-time | 2 | XAU, XAG (precious metals) |
| **1 Minute** | High frequency | ~100 | Most daily-traded commodities (energy, metals, agriculture) |
| **10 Minutes** | Moderate frequency | ~4 | TIOC, FC1, LC1, LHOGS |
| **Monthly** | World Bank/IMF data | ~30 | NG-EU, COAL-AU, DBLC1, BEEF, BANANA-EU, etc. |

**Important for our project:**
- Daily-frequency symbols (1 min interval): Use the **Historical** or **Timeseries** endpoint
  to get daily OHLC data. These symbols have data for every trading day.
- Monthly symbols: Only have one data point per month (closing rate). Available via
  Historical and Timeseries endpoints, but be aware of sparse data when merging
  with daily electricity prices.

---

## 13. Quote Currencies

The `quote` parameter allows converting commodity prices to any of 175+ currencies.

**Usage:**
```
GET /v2/latest?apiKey=YOUR_KEY&symbols=TTF-GAS&quote=NOK
```

**Availability:** Premium and Plus plans only. Lite users get the default currency per symbol.

**Relevant currencies for this project:**

| Code | Currency | Use case |
|------|----------|----------|
| USD | US Dollar | Default for most commodities |
| EUR | Euro | Default for TTF-GAS; ENTSO-E prices are in EUR |
| NOK | Norwegian Krone | Norwegian market perspective |
| GBP | British Pound | UK gas (NBP) |

**Note:** If on the Lite plan, fetch in default currency and convert using Norges Bank
FX data (already available via `fetch_fx.py`).

---

## 14. Error Codes & Handling

### HTTP Status Codes

| Code | Error Type | Message |
|------|-----------|---------|
| **200** | Success | — |
| **400** | VALIDATION_ERROR | Various: missing symbols, invalid date format, date range exceeded |
| **401** | API_KEY_NOT_FOUND | API key missing from request |
| **402** | PAYMENT_REQUIRED | Trial expired, no subscription, or max symbols exceeded |
| **403** | LIMIT_REACHED | Usage limit reached — upgrade plan |
| **404** | USER_NOT_FOUND | Invalid API key |
| **404** | SYMBOL_NOT_FOUND | Unsupported symbol |
| **404** | QUOTE_NOT_FOUND | Invalid quote currency |
| **404** | RATE_NOT_FOUND | No rate found for specified date |
| **404** | DATA_NOT_FOUND | No data for specified date range |
| **429** | TOO_MANY_REQUESTS | Rate limit exceeded — wait and retry |
| **500** | SERVER_ERROR | Server-side error — retry later |

### Error Response Format

```json
{
  "timestamp": "2024-06-15T12:00:00.000Z",
  "path": "/v2/latest",
  "code": "VALIDATION_ERROR",
  "error": "VALIDATION_ERROR",
  "message": "Symbols are required in the query parameters"
}
```

### Error Handling Strategy

```python
import requests
import time

def fetch_commodity(url, params, max_retries=3):
    for attempt in range(max_retries):
        response = requests.get(url, params=params)

        if response.status_code == 200:
            data = response.json()
            if data.get("success"):
                return data
            else:
                raise ValueError(f"API error: {data}")

        elif response.status_code == 429:
            wait = 2 ** attempt * 60  # Exponential backoff (minutes)
            print(f"Rate limited. Waiting {wait}s...")
            time.sleep(wait)

        elif response.status_code in (401, 402, 403, 404):
            error = response.json()
            raise ValueError(f"{error['error']}: {error['message']}")

        elif response.status_code >= 500:
            time.sleep(5)  # Server error — retry
            continue

    raise RuntimeError(f"Failed after {max_retries} retries")
```

---

## 15. Rate Limits & Subscription Plans

### Plans (as of 2026)

| Feature | Lite (Free Trial) | Plus | Premium |
|---------|-------------------|------|---------|
| Requests/month | Limited | Higher | Highest |
| Update frequency | Up to 10 min delay | Near real-time | Real-time |
| Custom quote currency | ❌ | ✅ | ✅ |
| Timeseries | ✅ | ✅ | ✅ |
| Historical | ✅ | ✅ | ✅ |
| Max symbols per request | Limited | Higher | Highest |

**Rate limit:** If you receive HTTP 429, wait at least 60 seconds before retrying.

### Best practices

- Batch symbols in single requests (e.g., `symbols=NG-FUT,TTF-GAS,BRENTOIL-SPOT`)
  rather than one request per symbol
- Cache aggressively — daily commodity data doesn't change after market close
- For historical backfill, use the Timeseries endpoint (up to 1 year per request)
  rather than individual Historical requests per day
- Track usage via the Usage endpoint to avoid hitting limits

---

## 16. Data Fetching Strategy for This Project

### What to fetch

For electricity price forecasting, fetch these symbols daily:

```python
PRIMARY_SYMBOLS = ["TTF-GAS", "BRENTOIL-SPOT", "NG-FUT", "COAL"]
SECONDARY_SYMBOLS = ["NG-SPOT", "UK-GAS", "BRENTOIL-FUT", "WTIOIL-SPOT"]
ALL_SYMBOLS = PRIMARY_SYMBOLS + SECONDARY_SYMBOLS
```

### Backfill strategy (2017–2025)

Use the Timeseries endpoint in yearly chunks (max 365 days per request):

```python
for year in range(2017, 2026):
    start_date = f"{year}-01-01"
    end_date = f"{year}-12-31"
    cache_path = f"data/raw/commodity_{year}.parquet"

    if os.path.exists(cache_path):
        continue

    response = requests.get(
        f"{BASE_URL}/timeseries",
        params={
            "apiKey": api_key,
            "symbols": ",".join(PRIMARY_SYMBOLS),
            "startDate": start_date,
            "endDate": end_date,
        }
    )
    # ... parse and save to parquet ...
    time.sleep(2)
```

### Daily update strategy

Once backfill is done, fetch latest rates daily or use Historical endpoint:

```python
# Option A: Latest rates
response = requests.get(
    f"{BASE_URL}/latest",
    params={"apiKey": api_key, "symbols": ",".join(PRIMARY_SYMBOLS)}
)

# Option B: Yesterday's historical OHLC
yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
response = requests.get(
    f"{BASE_URL}/{yesterday}",
    params={"apiKey": api_key, "symbols": ",".join(PRIMARY_SYMBOLS)}
)
```

### Merging with electricity data

Commodity prices are daily, electricity prices are hourly:

```python
# Forward-fill daily commodity prices to match hourly electricity data
commodity_daily = pd.read_parquet("data/raw/commodity_2024.parquet")
commodity_daily.index = pd.to_datetime(commodity_daily.index)

# Create hourly index matching electricity data
hourly_index = pd.date_range("2024-01-01", "2024-12-31", freq="h", tz="Europe/Oslo")

# Reindex and forward-fill
commodity_hourly = commodity_daily.reindex(hourly_index.normalize()).ffill()
commodity_hourly.index = hourly_index
```

### Weekend/holiday handling

Markets are closed on weekends and holidays. Use forward-fill:

```python
# After parsing timeseries, fill gaps
df = df.asfreq("D")  # Ensure daily frequency
df = df.ffill()       # Forward-fill weekends and holidays
```

---

## 17. Python Implementation Patterns

### fetch_commodity.py — Skeleton

```python
"""
Fetch commodity prices from CommodityPriceAPI.
See docs/commodity_price_api.md for API reference.
"""

import os
import time
import requests
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

# ---------------------
# Configuration
# ---------------------
BASE_URL = "https://api.commoditypriceapi.com/v2"
API_KEY = os.getenv("COMMODITY_API_KEY")

# Symbols relevant for electricity price forecasting
PRIMARY_SYMBOLS = ["TTF-GAS", "BRENTOIL-SPOT", "NG-FUT", "COAL"]
SECONDARY_SYMBOLS = ["NG-SPOT", "UK-GAS", "BRENTOIL-FUT", "WTIOIL-SPOT"]

# ---------------------
# Helper
# ---------------------
def _make_request(endpoint: str, params: dict) -> dict:
    """Make authenticated request to CommodityPriceAPI."""
    params["apiKey"] = API_KEY
    response = requests.get(f"{BASE_URL}/{endpoint}", params=params)
    response.raise_for_status()
    data = response.json()
    if not data.get("success", False):
        raise ValueError(f"API error: {data}")
    return data


# ---------------------
# Core Functions
# ---------------------
def fetch_latest(symbols: list[str] = PRIMARY_SYMBOLS) -> pd.DataFrame:
    """Fetch latest commodity rates."""
    data = _make_request("latest", {"symbols": ",".join(symbols)})
    # ... parse into DataFrame ...
    pass


def fetch_historical(date: str, symbols: list[str] = PRIMARY_SYMBOLS) -> pd.DataFrame:
    """Fetch OHLC rates for a specific date (YYYY-MM-DD)."""
    data = _make_request(date, {"symbols": ",".join(symbols)})
    # ... parse into DataFrame with OHLC columns ...
    pass


def fetch_timeseries(
    start_date: str,
    end_date: str,
    symbols: list[str] = PRIMARY_SYMBOLS
) -> pd.DataFrame:
    """Fetch daily timeseries between two dates (max 1 year apart)."""
    data = _make_request("timeseries", {
        "symbols": ",".join(symbols),
        "startDate": start_date,
        "endDate": end_date,
    })
    # ... parse into DataFrame with date index ...
    pass


def fetch_all_years(
    start_year: int = 2017,
    end_year: int = 2025,
    symbols: list[str] = PRIMARY_SYMBOLS
) -> None:
    """Backfill commodity data in yearly chunks with caching."""
    for year in range(start_year, end_year + 1):
        cache_path = f"data/raw/commodity_{year}.parquet"
        if os.path.exists(cache_path):
            print(f"  Cached: {cache_path}")
            continue

        print(f"  Fetching: {year}...")
        df = fetch_timeseries(f"{year}-01-01", f"{year}-12-31", symbols)
        df.to_parquet(cache_path)
        time.sleep(2)  # Be nice to the API


# ---------------------
# Entry point
# ---------------------
if __name__ == "__main__":
    fetch_all_years()
```

### Parsing Timeseries Response

```python
def parse_timeseries(data: dict) -> pd.DataFrame:
    """
    Parse timeseries JSON response into a clean DataFrame.

    Returns DataFrame with:
    - DatetimeIndex (daily)
    - Columns like 'TTF-GAS_close', 'TTF-GAS_open', 'BRENTOIL-SPOT_close', etc.
    """
    records = []
    for date_str, symbols_data in data["rates"].items():
        row = {"date": pd.to_datetime(date_str)}
        for symbol, ohlc in symbols_data.items():
            if isinstance(ohlc, dict):
                row[f"{symbol}_open"] = ohlc.get("open")
                row[f"{symbol}_high"] = ohlc.get("high")
                row[f"{symbol}_low"] = ohlc.get("low")
                row[f"{symbol}_close"] = ohlc.get("close")
            else:
                # Latest endpoint returns simple values
                row[f"{symbol}_close"] = ohlc
        records.append(row)

    df = pd.DataFrame(records)
    df = df.set_index("date").sort_index()
    return df
```

---

## Quick Reference Card

```
Base URL:     https://api.commoditypriceapi.com/v2
Auth:         apiKey=... (query param) or x-api-key: ... (header)
Date format:  YYYY-MM-DD
Symbols:      Comma-separated (e.g., TTF-GAS,NG-FUT,BRENTOIL-SPOT)

Key symbols for electricity forecasting:
  TTF-GAS        European gas benchmark (EUR/MWh) — #1 driver
  BRENTOIL-SPOT  North Sea oil (USD/Barrel)
  NG-FUT         US natural gas futures (USD/MMBtu)
  COAL           Coal (USD/Metric Ton)

Endpoints:
  GET /v2/latest?symbols=...              → Current rates
  GET /v2/{YYYY-MM-DD}?symbols=...        → Historical OHLC
  GET /v2/timeseries?symbols=...&startDate=...&endDate=...  → Daily series (max 1yr)
  GET /v2/fluctuation?symbols=...&startDate=...&endDate=... → Change analysis
  GET /v2/symbols                         → All available symbols
  GET /v2/usage                           → Account usage
```

---

*This document is a reference for the project. Update as you discover data quality
issues or API changes. See also: docs/entsoe_api_reference.md for electricity data
and docs/frost_api_notes.md for weather data.*