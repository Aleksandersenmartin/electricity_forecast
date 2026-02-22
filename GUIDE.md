# Getting Started Guide — Electricity Price Forecasting

A hands-on reference for building a Nordic electricity price forecasting project with Python, ML, and Streamlit — using Claude Code as your learning partner.

---

## Table of Contents

1. [Project Structure](#1-project-structure)
2. [CLAUDE.md Best Practices](#2-claudemd-best-practices)
3. [Claude Code Workflow Tips](#3-claude-code-workflow-tips)
4. [Settings & Permissions](#4-settings--permissions)
5. [Data Sources](#5-data-sources)
6. [Data Caching Strategy](#6-data-caching-strategy)
7. [ML Approach](#7-ml-approach)
8. [Streamlit App & Deployment](#8-streamlit-app--deployment)
9. [Suggested Learning Path](#9-suggested-learning-path)
10. [MCP Server Setup (Optional)](#10-mcp-server-setup-optional)

---

## 1. Project Structure

```
electricity_forecast/
├── CLAUDE.md                # Context for Claude Code
├── GUIDE.md                 # This file (your reference)
├── README.md                # Project readme (create when ready)
├── requirements.txt         # pip dependencies
├── .env                     # API keys (never commit — add to .gitignore)
├── .env.example             # Template showing required env vars
├── .gitignore
├── .claudeignore            # Keeps large files out of Claude Code context
│
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── fetch_electricity.py  # ENTSO-E day-ahead prices (all zones)
│   │   ├── fetch_metro.py        # Weather (Frost API historical + Yr forecast)
│   │   ├── fetch_fx.py           # EUR/NOK exchange rates (Norges Bank)
│   │   └── fetch_commodity.py    # Natural gas / oil prices (CommodityPriceAPI)
│   │
│   ├── features/
│   │   ├── __init__.py
│   │   └── build_features.py     # Feature engineering (lags, rolling stats, etc.)
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── train.py              # Training pipeline
│   │   ├── evaluate.py           # Metrics and evaluation
│   │   └── predict.py            # Inference / forecasting
│   │
│   └── utils/
│       ├── __init__.py
│       └── config.py             # Shared configuration, paths, constants
│
├── data/
│   ├── raw/                      # Untouched downloaded data
│   └── processed/                # Cleaned, merged datasets
│
├── artifacts/                    # Saved model artifacts (.pkl, .joblib)
│
├── hooks/                        #Reusable hooks
├── app/
│   └── streamlit_app.py          # Streamlit dashboard
│
├── notebooks/
│   └── exploration.ipynb         # EDA and experimentation
│
├── docs/                         # Additional documentation and notes
│
└── tests/
    └── test_features.py          # Unit tests
```

**Key principles:**
- `src/data/` — one file per data source, each with a clear `fetch_*()` function
- `src/features/` — transform raw data into model-ready features
- `src/models/` — training, evaluation, prediction as separate concerns
- `data/raw/` vs `data/processed/` — never overwrite raw data
- `artifacts/` — saved models, separate from source code in `src/models/`
- `notebooks/` — for exploration only; move reusable code into `src/`

---

## 2. CLAUDE.md Best Practices

CLAUDE.md is loaded into every Claude Code conversation. It's your way of giving Claude persistent context about your project.

### What to include

| Section | Why it matters |
|---|---|
| **Project overview** | One paragraph so Claude understands the domain |
| **Architecture** | Folder layout, key modules, how things connect |
| **Commands** | How to run, test, lint — so Claude can verify its work |
| **Conventions** | Naming, formatting, patterns you want followed |
| **Domain rules** | Things Claude can't know without being told (rate limits, timezone rules, etc.) |
| **Current status** | What's built, what's in progress — update as you go |
| **Working style** | How you want to interact (explain first, don't auto-commit, etc.) |

### Tips for effectiveness

- **Keep it under 150 lines.** Claude reads this every time — bloat slows things down.
- **Update it regularly.** As your project grows, the CLAUDE.md should reflect reality.
- **Be specific.** "Use pandas for data, scikit-learn for models" is better than "use standard libraries."
- **Include commands.** If Claude knows how to run tests (`pytest tests/`), it can verify its own work.
- **State what NOT to do.** "Don't auto-commit" or "Don't add type hints unless I ask" prevents unwanted changes.
- **Keep CLAUDE.md and GUIDE.md consistent.** If you change a data source or convention in one, update the other.

---

## 3. Claude Code Workflow Tips

### Use Plan Mode for learning

Plan mode (press `Shift+Tab` twice) is your best friend for learning. Instead of Claude writing code immediately, it:
1. Explores your codebase
2. Proposes an approach
3. Waits for your approval

**How to use it for learning:**
```
You: "I want to fetch ENTSO-E day-ahead prices for all 5 Norwegian zones.
      Don't write the code — explain the API, show me the data format,
      and outline the steps. I'll implement it myself."
```

### Effective prompting patterns

| Pattern | Example |
|---|---|
| **Explain, don't implement** | "Explain how the entsoe-py wrapper handles bidding zones" |
| **Review my code** | "Review fetch_electricity.py — what could be improved?" |
| **Teach the concept** | "Explain what feature engineering means for time series" |
| **Scaffold then fill** | "Create the file structure, I'll write the functions" |
| **Debug together** | "This returns empty data — help me debug step by step" |
| **Compare approaches** | "Should I use XGBoost or LightGBM for this? Explain trade-offs" |

### Useful slash commands

| Command | What it does |
|---|---|
| `/plan` | Enter plan mode — Claude researches before acting |
| `/commit` | Stage and commit changes with a good message |
| `/clear` | Clear conversation context (start fresh) |
| `/help` | Show all available commands |
| `/compact` | Summarize conversation to free up context space |

### The "teach me" workflow

1. Ask Claude to **explain** the concept
2. Ask Claude to **create a skeleton** (empty functions with docstrings)
3. **You write** the implementation
4. Ask Claude to **review** your code
5. Ask Claude to help you **write tests**
6. Iterate

---

## 4. Settings & Permissions

### Project settings (`.claude/settings.json`)

```json
{
  "permissions": {
    "allow": [
      "Bash(source .venv/bin/activate && python *)",
      "Bash(pytest *)",
      "Bash(pip install *)",
      "Bash(streamlit run *)",
      "Bash(ruff check *)"
    ],
    "deny": [
      "Bash(rm -rf *)",
      "Bash(git push *)"
    ]
  }
}
```

### Recommended `.gitignore`

```
# Environment
.venv/
.env
__pycache__/

# Data (too large for git)
data/raw/
data/processed/
artifacts/*.pkl
artifacts/*.joblib

# IDE
.vscode/
.idea/

# Claude Code
.claude/
```

### Recommended `.claudeignore`

```
data/raw/
data/processed/
artifacts/
notebooks/.ipynb_checkpoints/
.env
*.sqlite
*.db
__pycache__/
*.pyc
.pytest_cache/
```

### Environment variables (`.env`)

Only three keys needed — Norges Bank and Yr require no API keys:

```
ENTSOE_API_KEY=your-token-here
FROST_CLIENT_ID=your-client-id-here
COMMODITY_API_KEY=your-key-here
```

Load them with `python-dotenv`:
```python
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("ENTSOE_API_KEY")
```

---

## 5. Data Sources

### Authentication overview

| Source | Auth method | Registration |
|--------|-----------|--------------|
| **hvakosterstrommen.no** (day-ahead prices) | None (public API) | None |
| **ENTSO-E** | API token in query params | Register at transparency.entsoe.eu → email transparency@entsoe.eu |
| **Frost API** (historical weather) | HTTP Basic Auth (client ID as username, empty password) | Register at frost.met.no with email |
| **Yr / Locationforecast** (forecast weather) | No key — custom `User-Agent` header required | None |
| **Norges Bank** (FX rates) | Fully open — no auth at all | None |
| **CommodityPriceAPI** | API key in query params | Register at commoditypriceapi.com |

### Day-Ahead Prices — hvakosterstrommen.no — `fetch_nordpool.py`

**Primary source for day-ahead prices.** No API key required. Free public API that serves
ENTSO-E price data for Norwegian bidding zones.

| Detail | Info |
|---|---|
| **What** | Day-ahead prices per zone (NO1–NO5) in EUR and NOK |
| **API** | `https://www.hvakosterstrommen.no/api/v1/prices/{YEAR}/{MM}-{DD}_{ZONE}.json` |
| **Auth** | None — fully open, no registration |
| **Granularity** | Hourly (one API call per zone per day) |
| **Format** | EUR/kWh and NOK/kWh (module converts to EUR/MWh) |
| **Historical data** | Continuous from October 2021; patchy before that |

**Data source:** hvakosterstrommen.no serves ENTSO-E day-ahead prices converted via
Norges Bank exchange rates. Nord Pool's own Data Portal API now requires a paid
subscription (1,200+ EUR/year), so this free alternative is used instead.

```python
from src.data.fetch_nordpool import fetch_prices, fetch_zone_prices

# All 5 zones at once (returns DataFrame with NO_1..NO_5 columns, EUR/MWh)
prices = fetch_prices("2021-10-01", "2026-02-22")

# Full detail for a single zone (EUR/MWh, NOK/kWh, exchange rate)
detail = fetch_zone_prices("NO_5", "2024-01-01", "2024-12-31")
```

**Zone format:** The API uses `"NO1"` (no underscore). The module maps to/from the project
format `"NO_1"` automatically — you always use `"NO_1"` in your code.

**Caching:** Completed years are cached as `data/raw/prices_{zone}_{year}.parquet`.
Current year re-fetches on each run. First full backfill for all 5 zones takes ~30 min
(0.3s per daily API call × ~1,600 days × 5 zones).

### ENTSO-E Transparency Platform — `fetch_electricity.py`

Provides load, generation by type, and cross-border flows — data not available from
hvakosterstrommen.no. Also has prices (same underlying data). Requires an API key.

ENTSO-E is the upstream source for Nord Pool day-ahead prices.

| Detail | Info |
|---|---|
| **What** | Day-ahead prices, actual generation per type, load, cross-border flows |
| **API** | REST API — free, requires registration |
| **Sign up** | https://transparency.entsoe.eu/ → register → email transparency@entsoe.eu |
| **Python package** | `pip install entsoe-py` |
| **Granularity** | Hourly (transitioning to 15-minute in 2025) |
| **Format** | EUR/MWh |

**Norwegian bidding zones:**

| Zone | Area | EIC Code | `entsoe-py` key |
|------|------|----------|-----------------|
| NO1 | Øst-Norge (Oslo) | `10YNO-1--------2` | `NO_1` |
| NO2 | Sør-Norge (Kristiansand) | `10YNO-2--------T` | `NO_2` |
| NO3 | Midt-Norge (Trondheim) | `10YNO-3--------J` | `NO_3` |
| NO4 | Nord-Norge (Tromsø) | `10YNO-4--------9` | `NO_4` |
| NO5 | Vest-Norge (Bergen) | `10Y1001A1001A48H` | `NO_5` |

```python
from entsoe import EntsoePandasClient
import pandas as pd

client = EntsoePandasClient(api_key=os.getenv("ENTSOE_API_KEY"))

start = pd.Timestamp("2024-01-01", tz="Europe/Oslo")
end = pd.Timestamp("2024-01-31", tz="Europe/Oslo")

zones = ["NO_1", "NO_2", "NO_3", "NO_4", "NO_5"]
all_prices = {}
for zone in zones:
    all_prices[zone] = client.query_day_ahead_prices(zone, start=start, end=end)
```

### MET Norway Weather Data — `fetch_metro.py`

MET Norway provides several free data services. You need two of them:

| Service | URL | What | Use case |
|---------|-----|------|----------|
| **Frost API** | frost.met.no | Historical observations from weather stations | Training data (8 years) |
| **Yr / Locationforecast** | api.met.no/weatherapi/locationforecast/2.0 | Weather forecast (up to 9 days) | Live forecasting in dashboard |
| **seKlima** | seklima.met.no | Web GUI for observations & statistics | Manual exploration, not API |

All MET Norway data is free under Creative Commons license. Credit: "Data from MET Norway."

#### Frost API (historical observations)

Requires registration — go to https://frost.met.no/, register with email, receive client ID.

```python
import requests

client_id = os.getenv("FROST_CLIENT_ID")
endpoint = "https://frost.met.no/observations/v0.jsonld"

params = {
    "sources": "SN18700",           # Oslo - Blindern
    "elements": "air_temperature,wind_speed,sum(precipitation_amount PT1H)",
    "referencetime": "2024-01-01/2024-01-31",
    "timeresolutions": "PT1H",
}

# HTTP Basic Auth: client_id as username, empty password
resp = requests.get(endpoint, params=params, auth=(client_id, ""))
data = resp.json()
```

**Key weather stations per bidding zone:**

| Zone | Station | Frost ID |
|------|---------|----------|
| NO1 (Oslo) | Oslo - Blindern | SN18700 |
| NO2 (Kristiansand) | Kristiansand - Kjevik | SN39040 |
| NO3 (Trondheim) | Trondheim - Voll | SN68860 |
| NO4 (Tromsø) | Tromsø | SN90450 |
| NO5 (Bergen) | Bergen - Florida | SN50540 |

**Useful Frost elements for price forecasting:**

| Element | Description | Relevance |
|---------|-------------|-----------|
| `air_temperature` | Temperature (°C) | Heating demand drives prices |
| `wind_speed` | Wind speed (m/s) | Wind generation suppresses prices |
| `sum(precipitation_amount PT1H)` | Hourly rainfall (mm) | Hydro inflow indicator |
| `surface_snow_thickness` | Snow depth (cm) | Future hydro inflow |
| `cloud_area_fraction` | Cloud cover (%) | Solar generation proxy |

#### Yr / Locationforecast (weather forecasts)

No API key needed — but you **must** set a custom `User-Agent` header with your app name and contact info. Without this you get `403 Forbidden`.

```python
import requests

headers = {
    "User-Agent": "electricity-forecast github.com/aleksandersenmartin"
}

# Compact format — enough for most needs
url = "https://api.met.no/weatherapi/locationforecast/2.0/compact"
params = {"lat": 59.91, "lon": 10.75}  # Oslo

resp = requests.get(url, params=params, headers=headers)
forecast = resp.json()
```

**Important Yr/Locationforecast rules:**
- Always use HTTPS
- Don't exceed 20 requests/second
- Respect `Expires` header — don't re-fetch before it expires
- Don't use more than 4 decimal places in coordinates (caching)
- Forecast data only (up to ~9 days ahead) — not historical

### Norges Bank Exchange Rates — `fetch_fx.py`

**Fully open — no API key, no registration, no auth.** Just call the URL.

ENTSO-E returns prices in EUR/MWh. You need EUR/NOK to convert.

| Detail | Info |
|---|---|
| **What** | Daily exchange rates (EUR/NOK, USD/NOK, etc.) |
| **API** | Norges Bank Data Warehouse — SDMX REST API |
| **Docs** | https://www.norges-bank.no/en/topics/statistics/open-data/guide-data-warehouse/ |
| **Query builder** | https://app.norges-bank.no/query/#/en/ |
| **Auth** | None |
| **Granularity** | Daily (no weekends/holidays — forward-fill needed) |

```python
import requests

# EUR/NOK exchange rate — no auth needed
url = "https://data.norges-bank.no/api/data/EXR/B.EUR.NOK.SP"
params = {
    "startPeriod": "2017-01-01",
    "endPeriod": "2025-01-31",
    "format": "sdmx-json",
}

resp = requests.get(url, params=params)
data = resp.json()
```

**Available formats:**
- `sdmx-json` — JSON (recommended for Python)
- `csv` — one observation per row
- `csv-ts` — one time series per row
- `excel-both` — Excel format

**Tip:** Use the [query builder tool](https://app.norges-bank.no/query/#/en/) to explore available data series and generate API URLs.

### Price Units & Conversion

Norwegian electricity prices exist in three units depending on audience:

| Unit | Column | Audience | Typical range |
|------|--------|----------|---------------|
| **EUR/MWh** | `price_eur_mwh` | Market/trading | 20–300 |
| **NOK/MWh** | `price_nok_mwh` | Industry/grid operators | 200–3,500 |
| **NOK/kWh** | `price_nok_kwh` | Consumer bills | 0.20–3.50 |

**Conversion formulas:**
```python
price_nok_mwh = price_eur_mwh * eur_nok    # EUR/NOK from Norges Bank
price_nok_kwh = price_nok_mwh / 1000        # MWh → kWh
```

**Worked example** (January 2024 average for NO_5 Bergen):
```
price_eur_mwh = 65.2 EUR/MWh
eur_nok       = 11.35
price_nok_mwh = 65.2 × 11.35 = 740 NOK/MWh
price_nok_kwh = 740 / 1000   = 0.740 NOK/kWh

→ A 2 kW heater costs 0.740 × 2 = 1.48 NOK/hour to run
```

**When to use each unit:**
- **EUR/MWh for modeling** — market standard, used by ENTSO-E and Nord Pool
- **NOK/kWh for consumer reporting** — what appears on electricity bills and Streamlit dashboards
- **NOK/MWh for industry** — used by grid operators and large consumers

**Modeling note:** EUR and NOK features are ~r>0.99 correlated (the EUR/NOK rate changes slowly compared to daily price swings). Use EUR/MWh features for model training. NOK features are included for dashboards and consumer-facing reporting only.

### Commodity Prices — `fetch_commodity.py`

| Detail | Info |
|---|---|
| **What** | Natural gas, oil (Brent/WTI), coal |
| **API** | CommodityPriceAPI REST |
| **Docs** | https://commoditypriceapi.com/#documentation |
| **Auth** | API key in query params |
| **Granularity** | Daily |

```python
import requests

api_key = os.getenv("COMMODITY_API_KEY")
url = f"https://commoditypriceapi.com/api/latest?access_key={api_key}&base=USD&symbols=NG,BRENT"
resp = requests.get(url)
data = resp.json()
```

**Alternative (free, no API key):**
```python
import yfinance as yf
gas = yf.download("NG=F", start="2017-01-01", end="2025-01-31")    # Natural gas futures
oil = yf.download("BZ=F", start="2017-01-01", end="2025-01-31")    # Brent crude futures
```

---

## 6. Data Caching Strategy

Fetching 8 years of hourly data for 5 zones means ~350,000 rows of price data alone. Build smart caching from day one.

### Principles

1. **Never re-fetch data you already have.** Check `data/raw/` before calling the API.
2. **Fetch incrementally.** Only request the missing date range.
3. **Store raw data immutably.** Save to `data/raw/` as-is. Processing happens separately.
4. **Use Parquet** (not CSV) — faster, smaller, preserves dtypes and timezones. Install with `pip install pyarrow`.

### Chunked fetching for large date ranges

ENTSO-E can time out on very large queries. Fetch in yearly chunks:

```python
import pandas as pd
import time
import os
from entsoe import EntsoePandasClient

client = EntsoePandasClient(api_key=os.getenv("ENTSOE_API_KEY"))
zones = ["NO_1", "NO_2", "NO_3", "NO_4", "NO_5"]

for year in range(2017, 2025):
    start = pd.Timestamp(f"{year}-01-01", tz="Europe/Oslo")
    end = pd.Timestamp(f"{year}-12-31 23:00", tz="Europe/Oslo")

    for zone in zones:
        filepath = f"data/raw/electricity_{zone}_{year}.parquet"

        if os.path.exists(filepath):
            print(f"Skipping {zone} {year} — already cached")
            continue

        prices = client.query_day_ahead_prices(zone, start=start, end=end)
        prices.to_frame("price_eur_mwh").to_parquet(filepath)
        print(f"Saved {zone} {year}")
        time.sleep(2)  # Be polite to the API
```

### Incremental update pattern

```python
def fetch_electricity(zone: str, start: str, end: str) -> pd.DataFrame:
    """Fetch ENTSO-E prices with local cache."""

    cache_path = f"data/raw/electricity_{zone}.parquet"

    if os.path.exists(cache_path):
        cached = pd.read_parquet(cache_path)
        last_date = cached.index.max()

        if last_date >= pd.Timestamp(end, tz="Europe/Oslo"):
            return cached

        new_data = client.query_day_ahead_prices(
            zone, start=last_date + pd.Timedelta(hours=1), end=end
        )
        combined = pd.concat([cached, new_data])
        combined.to_parquet(cache_path)
        return combined

    data = client.query_day_ahead_prices(zone, start=start, end=end)
    data.to_frame("price_eur_mwh").to_parquet(cache_path)
    return data
```

---

## 7. ML Approach

### Recommended progression (simple to complex)

#### Step 1: Baseline — Naive + Linear Regression

- **Naive baseline:** Tomorrow's price = same hour last week. Surprisingly hard to beat.
- **Linear regression:** Predict price using lag-1, lag-24, day of week, month.

These give you a floor to measure all other models against.

#### Step 2: Feature-rich — XGBoost / LightGBM / CatBoost

Gradient boosted trees work well for tabular data:
- Price lags (1h, 24h, 48h, 168h)
- Rolling statistics (mean, std of last 24h, 7 days)
- Weather features (temperature, wind speed, precipitation)
- Commodity prices (gas, oil — lagged since they're daily)
- FX rate (EUR/NOK)
- Calendar features (hour of day, day of week, month, holidays)

```python
features = [
    "price_lag_1h", "price_lag_24h", "price_lag_168h",
    "price_rolling_mean_24h", "price_rolling_std_24h",
    "temperature", "wind_speed", "precipitation",
    "gas_price_lag_1d", "oil_price_lag_1d",
    "hour_of_day", "day_of_week", "month", "is_weekend", "is_holiday",
]
```

#### Step 3: Ensemble

Combine XGBoost, LightGBM, and CatBoost predictions (simple average or weighted).

#### Step 4: Time series models (optional, later)

- **Prophet** — good for seasonality, easy to use
- **LSTM / Transformer** — neural networks for sequences (bigger learning curve)

### Key ML concepts

| Concept | Why it matters |
|---|---|
| **Train/test split for time series** | Always split by time, never randomly. |
| **Feature engineering** | Lags, rolling stats, and calendar features matter more than model choice. |
| **Cross-validation** | Use `TimeSeriesSplit` from scikit-learn, not regular k-fold. |
| **Evaluation metrics** | MAE and RMSE are standard for price forecasting. |
| **Overfitting** | If train error ≪ test error, your model memorized noise. |
| **Feature importance** | XGBoost/LightGBM show which features matter — use this to iterate. |
| **Naive baseline** | Always compare against "same hour last week." |

### Libraries to install

```
# Core
pip install pandas numpy scikit-learn pyarrow

# ML models
pip install xgboost lightgbm catboost

# Data sources
pip install entsoe-py requests python-dotenv

# Visualization & dashboard
pip install streamlit plotly matplotlib seaborn

# Optional
pip install yfinance   # Alternative commodity data
pip install ruff       # Linting
```

---

## 8. Streamlit App & Deployment

### Start simple

```python
# app/streamlit_app.py
import streamlit as st
import pandas as pd

st.title("Electricity Price Forecast — Nordic")

df = pd.read_parquet("data/processed/prices_all_zones.parquet")

zone = st.selectbox("Bidding Zone", ["NO1", "NO2", "NO3", "NO4", "NO5"])
st.line_chart(df[df["zone"] == zone].set_index("timestamp")["price_eur_mwh"])
```

Run with: `streamlit run app/streamlit_app.py`

### Build up incrementally

1. **v1:** Historical prices with zone selector
2. **v2:** Weather data overlay
3. **v3:** Model predictions vs actual
4. **v4:** Feature importance plot
5. **v5:** Date range picker and refresh
6. **v6:** Anomaly detection highlights

### Deployment options

| Platform | Cost | Notes |
|---|---|---|
| **Streamlit Community Cloud** | Free | Connect to GitHub, auto-deploys |
| **Railway / Render** | Free tier | More control, scheduled jobs |
| **Local only** | Free | `streamlit run` on your machine |

---

## 9. Suggested Learning Path

### Phase 1: Foundation (week 1–2)
- [ ] Set up project structure (folders, venv, requirements.txt)
- [ ] Register for ENTSO-E API key (email transparency@entsoe.eu)
- [ ] Register for Frost API (frost.met.no — just needs email)
- [ ] Register for CommodityPriceAPI
- [ ] Write `src/data/fetch_electricity.py` — fetch prices for all 5 zones
- [ ] Implement caching (check existing data before fetching)
- [ ] Fetch 8 years of historical data (2017–2025) in yearly chunks
- [ ] Save raw data to `data/raw/` as Parquet
- [ ] Explore data in a notebook (`notebooks/exploration.ipynb`)

### Phase 2: More Data (week 2–3)
- [ ] Write `src/data/fetch_metro.py` — fetch historical weather from Frost API (one station per zone)
- [ ] Write `src/data/fetch_fx.py` — fetch EUR/NOK from Norges Bank (no key needed)
- [ ] Write `src/data/fetch_commodity.py` — fetch gas/oil prices
- [ ] Merge all data sources by timestamp
- [ ] Save merged data to `data/processed/`

### Phase 3: Feature Engineering (week 3–4)
- [ ] Create lag features (price_lag_1h, price_lag_24h, price_lag_168h)
- [ ] Create rolling statistics (mean, std over 24h and 168h windows)
- [ ] Add calendar features (hour, day of week, month, is_weekend, is_holiday)
- [ ] Handle missing values (interpolation for weather, forward-fill for FX/commodities)
- [ ] Write `src/features/build_features.py`

### Phase 3.5: Statistical Inference (week 4)

Before jumping into modeling, apply formal statistical methods to understand the data.
This validates assumptions, identifies key relationships, and informs feature selection.

See `notebooks/08_statistical_inference_analysis.ipynb` for full implementation.

- [x] Price distribution analysis — test normality (Shapiro-Wilk, Anderson-Darling), measure skewness/kurtosis, QQ plots
- [x] Seasonal decomposition — STL (trend + seasonal + residual), Kruskal-Wallis day-of-week test
- [x] Reservoir-price relationship — Spearman correlation, Granger causality (lags 1–8 weeks)
- [x] Export/import patterns — Mann-Whitney U test (export vs import day prices), export intensity
- [x] Commodity passthrough — OLS regression (price ~ TTF + Brent + coal + FX), cross-correlation lags, structural break analysis (pre/post energy crisis)
- [x] Zone decoupling — inter-zone correlation matrix, ADF stationarity test on price spreads, North-South divide
- [x] Autocorrelation — ADF & KPSS stationarity tests, ACF/PACF plots (168h lags), differencing analysis

**Statistical methods quick reference:**

| Method | Library call | What it tests | Key output |
|---|---|---|---|
| Shapiro-Wilk | `scipy.stats.shapiro()` | Is data normally distributed? | W stat, p-value |
| Anderson-Darling | `scipy.stats.anderson()` | Normality (sensitive to tails) | Test stat vs critical values |
| Skewness/Kurtosis | `scipy.stats.skew/kurtosis()` | Distribution shape | Scalar (0 = normal) |
| KDE | `scipy.stats.gaussian_kde()` | Non-parametric density estimate | Smooth density curve |
| STL | `statsmodels.tsa.seasonal.STL()` | Decompose trend + seasonal + residual | Three component series |
| Kruskal-Wallis | `scipy.stats.kruskal()` | Do group medians differ? (non-parametric ANOVA) | H stat, p-value |
| Spearman ρ | `scipy.stats.spearmanr()` | Monotonic (non-linear) correlation | ρ (-1 to +1), p-value |
| Pearson r | `scipy.stats.pearsonr()` | Linear correlation | r (-1 to +1), p-value |
| Granger causality | `statsmodels.tsa.stattools.grangercausalitytests()` | Does X help predict Y? | F-stat, p-value per lag |
| Mann-Whitney U | `scipy.stats.mannwhitneyu()` | Are two groups from same distribution? | U stat, p-value |
| OLS regression | `statsmodels.api.OLS()` | Linear model: Y = Xβ + ε | R², coefficients, t-stats |
| ADF test | `statsmodels.tsa.stattools.adfuller()` | Is series stationary? (H₀: unit root) | ADF stat, p-value |
| KPSS test | `statsmodels.tsa.stattools.kpss()` | Is series stationary? (H₀: stationary) | KPSS stat, p-value |
| ACF | `statsmodels.tsa.stattools.acf()` | Autocorrelation at each lag | Correlation values |
| PACF | `statsmodels.tsa.stattools.pacf()` | Direct (partial) autocorrelation | Correlation values |

### Phase 4: Modeling (week 4–6)
- [ ] Implement train/test split by time (e.g., train 2017–2023, test 2024)
- [ ] Build naive baseline (same hour last week)
- [ ] Train linear regression
- [ ] Evaluate baselines with MAE and RMSE
- [ ] Train XGBoost model
- [ ] Train LightGBM and CatBoost
- [ ] Build simple ensemble (average of three models)
- [ ] Compare all models, analyze feature importance
- [ ] Write `src/models/train.py` and `src/models/evaluate.py`

### Phase 5: Streamlit App (week 6–7)
- [ ] Basic app with price chart and zone selector
- [ ] Prediction overlay (actual vs predicted)
- [ ] Feature importance plot
- [ ] Date range picker
- [ ] Deploy to Streamlit Community Cloud (optional)

### Phase 6: Anomaly Detection & Iteration
- [ ] XmR control charts on price residuals
- [ ] Flag anomalous periods in the dashboard
- [ ] Add Yr/Locationforecast for live weather predictions
- [ ] Try additional features (cross-border flows, hydro reservoir data)
- [ ] Automated daily data refresh
- [ ] Confidence intervals on predictions
- [ ] Experiment with 15-minute resolution data

---

## 10. MCP Server Setup (Optional)

Set these up later — focus on building the data pipeline first.

### Fetch MCP Server (web requests)
```json
{
  "fetch": {
    "command": "uvx",
    "args": ["mcp-server-fetch"]
  }
}
```

### SQLite MCP Server
```json
{
  "sqlite": {
    "command": "uvx",
    "args": ["mcp-server-sqlite", "--db-path", "data/cache.db"]
  }
}
```

### GitHub MCP Server
```json
{
  "github": {
    "command": "npx",
    "args": ["-y", "@modelcontextprotocol/server-github"],
    "env": {
      "GITHUB_PERSONAL_ACCESS_TOKEN": "<your-token>"
    }
  }
}
```

Configure in `~/.claude/settings.json` (global) or `.claude/settings.json` (project-level).

---

## Quick Reference — Asking Claude Code for Help

```
# Learning a concept
"Explain time series cross-validation. Why can't I use random splits?"

# Getting started on a task
"I want to write fetch_electricity.py for all 5 Norwegian zones.
Enter plan mode and help me design it, but I'll write the code myself."

# Reviewing your code
"Review src/data/fetch_electricity.py — is my error handling correct?
What edge cases am I missing?"

# Debugging
"fetch_metro.py returns NaN for some hours. Help me debug — don't fix it,
just help me understand why."

# Comparing approaches
"Should I use XGBoost or LightGBM for hourly price prediction?
What are the trade-offs?"
```

---

*This guide is your roadmap. Work through it at your own pace, use Claude Code to learn as you go, and update both CLAUDE.md and this guide as your project evolves.*