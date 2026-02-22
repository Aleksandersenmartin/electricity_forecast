# Nordic Electricity Price Forecast

ML-based forecasting of day-ahead electricity prices for Norwegian bidding zones (NO1–NO5). Uses weather, fuel prices, FX rates, reservoir levels, and grid data to predict prices on the Nord Pool market.

## Project Status

**Phase 1 (Data Foundation) complete** — all data fetchers implemented and tested. Next: feature engineering and baseline models.

| Module | Source | Status | Auth |
|--------|--------|--------|------|
| `fetch_metro.py` | MET Norway Frost API | Tested | `FROST_CLIENT_ID` |
| `fetch_fx.py` | Norges Bank | Tested | None |
| `fetch_commodity.py` | yfinance + CommodityPriceAPI | Tested | `COMMODITY_API_KEY` |
| `fetch_electricity.py` | ENTSO-E Transparency | Code complete | `ENTSOE_API_KEY` (pending) |
| `fetch_reservoir.py` | NVE Magasinstatistikk | Tested | None |
| `fetch_statnett.py` | Statnett Driftsdata | Tested | None |

## Quick Start

```bash
# Clone and set up
git clone https://github.com/Aleksandersenmartin/electricity_forecast.git
cd electricity_forecast
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Configure API keys
cp .env.example .env
# Edit .env with your keys (see "API Keys" below)

# Test a fetcher (weather data for Bergen, no ENTSO-E key needed)
python -m src.data.fetch_metro
```

## API Keys

Store all keys in `.env` (never commit this file).

| Variable | Source | How to Get |
|----------|--------|------------|
| `FROST_CLIENT_ID` | MET Norway Frost API | Register at [frost.met.no](https://frost.met.no/) with email |
| `ENTSOE_API_KEY` | ENTSO-E Transparency | Register at [transparency.entsoe.eu](https://transparency.entsoe.eu/), request token via email |
| `COMMODITY_API_KEY` | CommodityPriceAPI | Register at [commoditypriceapi.com](https://commoditypriceapi.com/) |

Three data sources require no authentication: Norges Bank (FX), NVE (reservoirs), and Statnett (grid data).

## Project Structure

```
src/
    data/                  # One fetcher per data source
        fetch_metro.py         Weather (temperature, wind, precipitation)
        fetch_fx.py            EUR/NOK exchange rates
        fetch_commodity.py     Gas, oil, coal prices
        fetch_electricity.py   Day-ahead prices, load, generation, flows
        fetch_reservoir.py     Hydro reservoir filling per zone
        fetch_statnett.py      Cross-border flows, production/consumption
    features/
        build_features.py      Feature engineering (TODO)
    models/
        train.py               Training pipeline (TODO)
        evaluate.py            Metrics and evaluation (TODO)
        predict.py             Inference (TODO)
    utils/
        config.py              Shared configuration (TODO)

data/
    raw/                   Cached API responses (Parquet)
    processed/             Cleaned, merged datasets (TODO)

artifacts/                 Saved models (TODO)
app/
    streamlit_app.py       Dashboard (TODO)
notebooks/                 Exploration and EDA
docs/                      API reference documentation
tests/                     Unit tests (TODO)
```

## Data Sources

### Weather — MET Norway Frost API (`fetch_metro.py`)

Historical weather observations (hourly) from representative stations per bidding zone.

| Zone | Station | Location |
|------|---------|----------|
| NO_1 | SN18700 | Blindern (Oslo) |
| NO_2 | SN39040 | Kjevik (Kristiansand) |
| NO_3 | SN68860 | Værnes (Trondheim) |
| NO_4 | SN90450 | Tromsø |
| NO_5 | SN50540 | Florida (Bergen) |

**Elements:** `air_temperature` (°C), `wind_speed` (m/s), `precipitation` (mm/h)

```python
from src.data.fetch_metro import fetch_zone_weather
df = fetch_zone_weather("NO_5", "2020-01-01", "2026-02-22")
# 53,712 rows, hourly, cached as Parquet
```

### FX Rates — Norges Bank (`fetch_fx.py`)

Daily EUR/NOK exchange rate. Business days only from source; forward-filled for weekends/holidays.

```python
from src.data.fetch_fx import fetch_eur_nok_daily_filled
df = fetch_eur_nok_daily_filled("2020-01-01", "2026-02-22")
# 1,548+ rows (business days), continuous after forward-fill
```

### Commodity Prices (`fetch_commodity.py`)

Daily OHLC prices for key energy commodities via yfinance (historical backfill) and CommodityPriceAPI (latest).

| Symbol | Ticker | Description |
|--------|--------|-------------|
| TTF-GAS | TTF=F | Dutch TTF natural gas (primary electricity price driver) |
| BRENTOIL-SPOT | BZ=F | Brent crude oil |
| NG-FUT | NG=F | US natural gas futures |
| COAL | MTF=F | Rotterdam coal futures |

```python
from src.data.fetch_commodity import fetch_commodities
df = fetch_commodities("2020-01-01", "2026-02-22")
# 1,545 rows, 16 columns (OHLC × 4 symbols)
```

### Electricity Market — ENTSO-E (`fetch_electricity.py`)

Day-ahead prices, load, generation by type, and cross-border flows for Norwegian bidding zones. Requires API key.

```python
from src.data.fetch_electricity import fetch_prices, fetch_all_entsoe

# Single zone
df = fetch_prices("NO_5", "2020-01-01", "2024-12-31")

# Everything for one zone
data = fetch_all_entsoe("NO_5", "2020-01-01", "2024-12-31")
# Returns dict: {prices, load, generation, reservoir, crossborder_flows}
```

**Norwegian bidding zones (EIC codes):**

| Zone | Region | EIC Code |
|------|--------|----------|
| NO_1 | Øst-Norge (Oslo) | `10YNO-1--------2` |
| NO_2 | Sør-Norge (Kristiansand) | `10YNO-2--------T` |
| NO_3 | Midt-Norge (Trondheim) | `10YNO-3--------J` |
| NO_4 | Nord-Norge (Tromsø) | `10YNO-4--------9` |
| NO_5 | Vest-Norge (Bergen) | `10Y1001A1001A48H` |

### Reservoir Filling — NVE (`fetch_reservoir.py`)

Weekly hydro reservoir filling per zone since 1995. Includes 20-year min/max/median benchmarks for deviation features.

```python
from src.data.fetch_reservoir import fetch_zone_reservoir_with_benchmarks
df = fetch_zone_reservoir_with_benchmarks("NO_5", "2020-01-01", "2026-02-22")
# 320 weekly rows with filling_pct, filling_vs_median, filling_vs_min, filling_vs_max
```

### Grid Data — Statnett (`fetch_statnett.py`)

Operational data from Norway's TSO: physical cross-border flows, production/consumption, real-time Nordic balance, power situation assessments, and grid frequency.

```python
from src.data.fetch_statnett import (
    fetch_physical_flows,
    fetch_production_consumption,
    fetch_latest_overview,
    fetch_power_situation,
    fetch_frequency,
)

# Historical daily data (2020–current)
df_flows = fetch_physical_flows()       # 2,245 rows, net exchange MWh/day
df_pc = fetch_production_consumption()  # 2,251 rows, production + consumption MWh/day

# Real-time snapshots (not cached)
df_overview = fetch_latest_overview()   # 7 Nordic countries × 8 metrics (MW)
df_sit = fetch_power_situation()        # NO_1–NO_5 situation assessment
df_freq = fetch_frequency("2026-02-22") # Grid frequency (Hz)
```

## Cached Data

All fetchers cache to `data/raw/` as Parquet files. Delete cache files to re-fetch.

| File | Size | Rows | Source |
|------|------|------|--------|
| `weather_NO_5_*.parquet` | 600 KB | 53,712 | Frost API |
| `fx_eur_nok_*.parquet` | 24 KB | 1,548 | Norges Bank |
| `commodity_yfinance_*.parquet` | 170 KB | 1,545 | yfinance |
| `reservoir_nve_all.parquet` | 339 KB | ~8,000 | NVE |
| `reservoir_nve_benchmarks.parquet` | 20 KB | 260 | NVE |
| `statnett_physical_flows.parquet` | 35 KB | 2,245 | Statnett |
| `statnett_prod_cons.parquet` | 62 KB | 2,251 | Statnett |

## Roadmap

- [x] **Phase 1** — Data fetching (all sources implemented)
- [ ] **Phase 2** — Feature engineering: lags, rolling stats, calendar features, cross-source joins
- [ ] **Phase 3** — Baseline models: naive (same-hour-last-week), linear regression, Ridge/Lasso
- [ ] **Phase 4** — Tree models: XGBoost, LightGBM, CatBoost, ensemble
- [ ] **Phase 5** — Streamlit dashboard with live forecasts and cable arbitrage tab
- [ ] **Phase 6** — Anomaly detection, cable arbitrage analysis, grid frequency signals

See `CLAUDE.md` for detailed ML strategy, feature engineering plan, and evaluation metrics.

## Key Domain Concepts

**Why these data sources matter for Norwegian electricity prices:**

- **Hydro reservoirs** (NVE) — Norway is ~95% hydro. Low reservoirs = high prices. Per-zone data from NVE is critical.
- **Wind/weather** (Frost) — Wind generation lowers prices; cold snaps raise heating demand.
- **Gas prices** (TTF) — Sets the marginal cost for thermal generation in Europe, which propagates to Nordic prices through cable flows.
- **Cross-border flows** (Statnett/ENTSO-E) — Norway is heavily interconnected. Export cables to DE, NL, GB, DK drain Norwegian supply and raise prices.
- **FX rates** (Norges Bank) — ENTSO-E prices are in EUR/MWh; Norwegian consumers pay in NOK.

## Tech Stack

- **Python 3.12** with virtual environment
- **Data:** pandas, requests, entsoe-py, yfinance
- **ML:** scikit-learn, XGBoost, LightGBM, CatBoost (planned)
- **Visualization:** Plotly, matplotlib (planned)
- **Dashboard:** Streamlit (planned)
- **Storage:** Parquet for raw data, python-dotenv for secrets

## Documentation

| File | Contents |
|------|----------|
| `CLAUDE.md` | Project context for Claude Code: architecture, conventions, ML strategy, feature plan |
| `GUIDE.md` | Getting-started guide: setup, workflow tips, learning path |
| `docs/entsoe_api_reference.md` | ENTSO-E Transparency Platform API reference |
| `docs/frost_api_docs.md` | MET Norway Frost API reference |
| `docs/commodity_price_api.md` | CommodityPriceAPI reference |
| `docs/nve_magasin_api_reference.md` | NVE Magasinstatistikk API reference |
| `docs/statnett_api_reference.md` | Statnett Driftsdata REST API reference |

## License

MIT
