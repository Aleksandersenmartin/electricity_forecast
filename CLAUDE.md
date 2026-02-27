# CLAUDE.md

## Project Overview

Nordic electricity market forecasting using ML. Predicts day-ahead prices, demand, production, reservoir levels, and trade flows for Norwegian bidding zones (NO1–NO5) using weather data (wind, temperature, precipitation), fuel prices (natural gas, oil), FX rates, reservoir levels, and grid data. Building incrementally to learn ML concepts.

MIT licensed, authored by Aleksandersenmartin.

## Purpose

ML-based forecasting system for Norwegian day-ahead electricity prices (NO1–NO5) using ENTSO-E data, weather data from MET Norway (Frost API for historical, Yr/Locationforecast for forecasts), FX data from Norges Bank, and commodity data from CommodityPriceAPI. Built to understand how the electricity market flows and to identify anomalies.

## Architecture

```
src/
    data/
        __init__.py
        fetch_electricity.py   # ENTSO-E day-ahead prices (all zones)
        fetch_nordpool.py      # Day-ahead prices via hvakosterstrommen.no (no auth, ENTSO-E data)
        fetch_metro.py         # Weather data (Frost API historical + Yr forecast)
        fetch_fx.py            # EUR/NOK exchange rates (Norges Bank)
        fetch_commodity.py     # Natural gas / oil prices (CommodityPriceAPI)
        fetch_reservoir.py     # NVE reservoir filling per zone (no auth needed)
        fetch_statnett.py      # Statnett physical flows, production/consumption, frequency (no auth)
        fetch_yr_forecast.py   # Yr Locationforecast weather (~9 days ahead, per zone)
    features/
        __init__.py
        build_features.py      # Feature engineering (lags, rolling stats, etc.)
    models/
        __init__.py
        train.py               # Training pipeline (MLPriceForecaster, train_ensemble, walk_forward_validate, forecast_with_yr)
        evaluate.py            # Metrics and evaluation (compute_metrics, comparison_table, plot_forecast, plot_residuals)
        forecasters.py         # Statistical forecasters (NaiveForecaster, ARIMAForecaster, SARIMAXForecaster, etc.)
        predict.py             # Inference / forecasting
    utils/
        __init__.py
        config.py              # Shared configuration, paths, constants

data/
    raw/                       # Untouched downloaded data
    processed/                 # Cleaned, merged datasets

artifacts/                     # Saved model artifacts (.pkl, .joblib)
app/
    streamlit_app.py           # Streamlit dashboard
notebooks/                     # Exploration and EDA (not production code)
tests/                         # Unit tests
docs/                          # Guides, data source docs, learning notes
```

## Tech Stack

- **Language:** Python 3.12
- **ML:** scikit-learn, XGBoost, CatBoost, LightGBM
- **Data:** pandas, numpy, entsoe-py, requests
- **Visualization:** Plotly, matplotlib
- **Dashboard:** Streamlit
- **Stats:** scipy, statsmodels
- **Database:** SQLite for local cache (data/cache.db)
- **Config:** python-dotenv for secrets, YAML for model params

## Key APIs & Credentials

All keys stored in `.env`, loaded with `python-dotenv`. See `.env.example` for template.

| Source | File | Auth | Env Variable |
|--------|------|------|--------------|
| ENTSO-E | `fetch_electricity.py` | API token (register + email) | `ENTSOE_API_KEY` |
| hvakosterstrommen.no (day-ahead prices) | `fetch_nordpool.py` | None (public API) | — |
| Frost API (historical weather) | `fetch_metro.py` | Client ID (register with email) | `FROST_CLIENT_ID` |
| Yr / Locationforecast (forecast) | `fetch_metro.py` | No key — requires `User-Agent` header | — |
| Norges Bank (FX rates) | `fetch_fx.py` | No key — fully open | — |
| CommodityPriceAPI | `fetch_commodity.py` | API key | `COMMODITY_API_KEY` |
| NVE Magasinstatistikk | `fetch_reservoir.py` | No key — fully open | — |
| Statnett Driftsdata | `fetch_statnett.py` | No key — fully open | — |

### Norwegian Bidding Zones (EIC codes)

- **NO1** (Øst-Norge / Oslo): `10YNO-1--------2`
- **NO2** (Sør-Norge / Kristiansand): `10YNO-2--------T`
- **NO3** (Midt-Norge / Trondheim): `10YNO-3--------J`
- **NO4** (Nord-Norge / Tromsø): `10YNO-4--------9`
- **NO5** (Vest-Norge / Bergen): `10Y1001A1001A48H`

## Environment

- Python 3.12
- Virtual environment: `.venv/` (activate with `source .venv/bin/activate`)
- Dependencies: `pip install -r requirements.txt`

## Commands

- `pytest tests/` — run tests
- `streamlit run app/streamlit_app.py` — launch dashboard
- `ruff check src/` — lint
- `python src/data/fetch_electricity.py` — fetch price data
- `python src/data/fetch_metro.py` — fetch weather data

## Code Conventions

- Type hints on all function signatures
- Docstrings (Google style) explaining parameters and return values
- snake_case for files and functions
- One data source per file in `src/data/` with a clear `fetch_*()` function
- pandas for data manipulation, scikit-learn/XGBoost/CatBoost/LightGBM for models
- Time series splits only (never random) for train/test
- API keys in `.env` — NEVER hardcode secrets, NEVER commit `.env`
- Raw data is immutable — write processed data to `data/processed/`
- All timestamps must be timezone-aware (`Europe/Oslo`)
- Logging via Python `logging` module, not print()

## Domain Rules

- ENTSO-E API has rate limits — cache downloaded data locally, don't re-fetch existing date ranges
- hvakosterstrommen.no API returns one day per zone at a time — cache aggressively per year, 0.3s delay between requests
- hvakosterstrommen.no uses zone format "NO1" (no underscore); project uses "NO_1" — fetch_nordpool.py maps both ways
- hvakosterstrommen.no prices are in EUR/kWh — multiply by 1000 for EUR/MWh (handled in fetch_nordpool.py)
- Continuous price data available from October 2021; patchy before that
- Fetch large date ranges in yearly chunks with sleep between calls
- Prices transition to 15-minute resolution in 2025 (EU market change) — design for this
- Weekend/holiday patterns differ significantly — always include calendar features
- NO2 and NO5 prices correlate with hydro reservoir levels
- EUR/NOK exchange rate affects price comparisons — ENTSO-E returns EUR/MWh
- Price unit conversion: `price_nok_mwh = price_eur_mwh × eur_nok`, `price_nok_kwh = price_nok_mwh / 1000`
- NOK lag features are computed FROM the NOK series (not EUR lags × FX), preserving the FX rate from the original timestamp
- Norges Bank FX has no weekend/holiday data — forward-fill needed
- Yr Locationforecast requires `User-Agent` header with app name + contact info
- Store raw data as Parquet (not CSV) to preserve dtypes and timezones
- Plotly `add_vline`/`add_hline` with `annotation_text` crashes on tz-aware pandas Timestamps — use `add_shape` + `add_annotation` instead

## ML Strategy

### Decision Framework — Who Decides What

| Decision | Owner | Claude Code role |
|----------|-------|-----------------|
| Target variable, forecast horizon | Martin | — |
| Train/test split periods | Martin | Implement |
| Evaluation metrics | Martin | Implement + visualize |
| Which features to include | Martin (with suggestions) | Propose + implement |
| Feature engineering code | Claude Code | Implement, Martin reviews |
| Model selection & hyperparameters | Claude Code proposes | Martin approves |
| Interpreting results | Martin | Provide analysis tools |

### Problem Definition

- **Targets:** Price (EUR/MWh), reservoir filling (%), load (MW), generation (MW), trade flows (MWh)
- **Granularity:** Hourly (transitioning to 15-min in 2025)
- **Forecast horizon:** 24 hours ahead (matching Nord Pool day-ahead auction) + ~9 days via Yr weather
- **Zones:** NO1–NO5 (separate models per zone)
- **Train period:** 2022-01-01 to 2024-12-31 (~26,280 hours)
- **Validation period:** 2025-01-01 to 2025-06-30 (~4,344 hours)
- **Test period:** 2025-07-01 to 2026-02-22
- **Walk-forward:** 6-fold expanding window, 720 hours (~1 month) per fold
- **Approach:** Fundamentals-only (no price lag features — model learns from physical drivers)
- **Validation:** Walk-forward with expanding window (never random split)

### Evaluation Metrics

| Metric | What it measures | Use for |
|--------|-----------------|---------|
| **MAE** (Mean Absolute Error) | Average EUR/MWh off | Primary metric — intuitive, robust to outliers |
| **RMSE** (Root Mean Squared Error) | Penalizes large errors more | Secondary — catches price spikes |
| **MAPE** (Mean Absolute Percentage Error) | Relative accuracy | Comparing across zones with different price levels |
| **Directional Accuracy** | % of correct up/down predictions | Trading relevance |
| **Peak Hour MAE** | MAE during hours 8–20 | Business relevance — peak hours matter most |

Always compare against the naive baseline (same hour last week).
A model that can't beat naive is not worth deploying.

### Feature Engineering Plan

#### 1. Price Features (autoregressive — strongest predictors)

**EUR/MWh (market standard — use for modeling):**
```
price_eur_mwh        — Base price in EUR/MWh
price_lag_1h         — Price 1 hour ago
price_lag_24h        — Same hour yesterday (daily pattern)
price_lag_168h       — Same hour last week (weekly pattern)
price_rolling_24h_mean   — Average price last 24 hours
price_rolling_24h_std    — Volatility last 24 hours
price_rolling_168h_mean  — Average price last week
price_diff_24h       — Price change vs 24h ago
price_diff_168h      — Price change vs 1 week ago
```

**NOK/MWh and NOK/kWh (consumer reporting — NOT for modeling):**
```
price_nok_mwh        — EUR/MWh × EUR/NOK
price_nok_kwh        — NOK/MWh / 1000
price_nok_mwh_lag_1h, price_nok_mwh_lag_24h, price_nok_mwh_lag_168h
price_nok_mwh_rolling_24h_mean, price_nok_mwh_rolling_24h_std, price_nok_mwh_rolling_168h_mean
price_nok_mwh_diff_24h, price_nok_mwh_diff_168h
price_nok_kwh_lag_1h, price_nok_kwh_lag_24h, price_nok_kwh_lag_168h
price_nok_kwh_rolling_24h_mean, price_nok_kwh_rolling_24h_std, price_nok_kwh_rolling_168h_mean
price_nok_kwh_diff_24h, price_nok_kwh_diff_168h
```

**Note:** EUR and NOK features are ~r>0.99 correlated. Use EUR/MWh features for model training. NOK features exist for consumer-facing dashboards and reporting only — including both in a model wastes feature slots with redundant information.

#### 2. Calendar Features

```
hour_of_day          — 0–23 (strong daily price pattern)
day_of_week          — 0–6 (weekend effect is huge)
month                — 1–12 (seasonal heating/cooling demand)
is_weekend           — Binary (prices drop ~20–40% on weekends)
is_holiday           — Binary (Norwegian public holidays — use 'holidays' library)
week_of_year         — 1–52 (captures seasonal patterns)
is_business_hour     — Binary (hours 8–17 weekdays)
```

#### 3. Weather Features (per zone)

```
temperature          — °C (cold = more heating = higher price)
temperature_lag_24h  — Yesterday's temperature
wind_speed           — m/s (more wind = more production = lower price)
precipitation        — mm (rain fills reservoirs)
cloud_cover          — % (affects solar, minor in Norway)
snow_depth           — cm (spring melt fills reservoirs)
temperature_forecast — Yr forecast for next 24h (if available)
```

#### 4. Supply Features (ENTSO-E + NVE)

```
# From ENTSO-E:
actual_load          — Current consumption (MW)
load_forecast        — Day-ahead consumption forecast
load_lag_24h         — Consumption same hour yesterday
load_lag_168h        — Consumption same hour last week
generation_hydro     — Hydro generation (MW) — dominates Norway
generation_wind      — Wind generation (MW)
net_import           — Total imports minus exports (MW)

# From NVE Magasinstatistikk (PRIMARY reservoir source — per zone!):
reservoir_filling    — Fyllingsgrad per zone (0–1, weekly, forward-filled to hourly)
reservoir_filling_diff — Week-over-week change (endring_fyllingsgrad from API)
reservoir_filling_twh — Absolute filling in TWh (for cross-zone comparison)
reservoir_vs_median  — Deviation from 20-year median (filling - median)
reservoir_vs_min     — Distance above historical minimum (scarcity signal)
reservoir_south      — Combined NO2+NO5 filling (55% of total capacity)
```

**Why NVE over ENTSO-E for reservoirs:** NVE provides per-zone data (NO1–NO5),
ENTSO-E only has whole-Norway. NVE has data since 1995, includes TWh, min/max/median
benchmarks, and requires no API key. See docs/nve_magasin_api_reference.md.

#### 5. Commodity Features (daily, forward-filled to hourly)

```
ttf_gas_close        — TTF Gas price (EUR/MWh) — #1 external driver
brent_oil_close      — Brent crude (USD/barrel)
coal_close           — Coal price (USD/ton)
ng_fut_close         — US natural gas futures (USD/MMBtu)
eur_nok              — Exchange rate (for NOK conversion)
ttf_gas_change_7d    — TTF 7-day price change (trend signal)
```

#### 6. Cross-Zone & Cable Arbitrage Features (Phase 4+)

```
# Foreign zone prices (all cable endpoints — fetch from ENTSO-E)
price_dk1            — Denmark West price (EUR/MWh)
price_dk2            — Denmark East price (EUR/MWh)
price_se1            — Sweden North price (EUR/MWh)
price_se2            — Sweden Mid price (EUR/MWh)
price_se3            — Sweden Stockholm price (EUR/MWh)
price_se4            — Sweden South price (EUR/MWh)
price_de_lu          — Germany/Luxembourg price (EUR/MWh)
price_nl             — Netherlands price (EUR/MWh)
price_gb             — Great Britain price (EUR/MWh)
price_fi             — Finland price (EUR/MWh)

# Physical flows on cables (MW, positive = export from Norway)
flow_no2_dk1         — Skagerrak cables (NO2 → DK1)
flow_no2_nl          — NorNed cable (NO2 → NL)
flow_no2_de          — NordLink cable (NO2 → DE)
flow_no2_gb          — North Sea Link (NO2 → GB)
flow_no1_se3         — NO1 → Sweden (largest interconnector)
flow_no3_se2         — NO3 → Sweden
flow_no4_se1         — NO4 → Sweden
flow_no4_se2         — NO4 → Sweden
flow_no4_fi          — NO4 → Finland

# Internal flows (between Norwegian zones)
flow_no1_no2         — Internal flow NO1 ↔ NO2
flow_no1_no3         — Internal flow NO1 ↔ NO3
flow_no1_no5         — Internal flow NO1 ↔ NO5
flow_no2_no5         — Internal flow NO2 ↔ NO5
flow_no3_no4         — Internal flow NO3 ↔ NO4
flow_no3_no5         — Internal flow NO3 ↔ NO5

# Price spreads (Norwegian zone minus foreign zone)
spread_no2_dk1       — NO2 price minus DK1 price
spread_no2_nl        — NO2 price minus NL price
spread_no2_de        — NO2 price minus DE price
spread_no2_gb        — NO2 price minus GB price
spread_no1_se3       — NO1 price minus SE3 price
spread_no4_fi        — NO4 price minus FI price

# Arbitrage indicators
arbitrage_no2_dk1    — spread * flow direction mismatch flag
arbitrage_no2_nl     — (see Cable Arbitrage Analysis below)
arbitrage_no2_de     — positive = potential inefficiency
```

### Cable Arbitrage Analysis

This is a dedicated analysis module to detect potential misuse or inefficiencies
in cross-border electricity trading. The concept:

**Normal market behavior:**
- Power flows from LOW price zone to HIGH price zone
- NO2 price < DK1 price → Norway exports to Denmark (flow positive)
- NO2 price > NL price → Norway imports from Netherlands (flow negative)

**Anomalous / suspicious behavior:**
- Export from Norway when Norwegian price is HIGHER than destination
  (selling cheap power abroad while domestic price is high)
- Import to Norway when Norwegian price is LOWER than source
  (buying expensive foreign power when domestic supply is cheap)
- Large flows with near-zero price spread (transaction costs exceed benefit)
- Persistent one-directional flow regardless of price signals

**Implementation approach:**

```python
# For each cable, for each hour:
spread = price_norway - price_foreign
flow = crossborder_flow  # positive = export from Norway

# Flag 1: Wrong-direction flow (export when should import, or vice versa)
wrong_direction = (spread > threshold) & (flow > 0)  # Exporting when NO is more expensive
wrong_direction |= (spread < -threshold) & (flow < 0)  # Importing when NO is cheaper

# Flag 2: Flow magnitude vs spread (large flow, small spread = suspicious)
flow_spread_ratio = abs(flow) / (abs(spread) + 0.01)  # Avoid division by zero

# Flag 3: Capacity utilization during high-spread hours
capacity_util = abs(flow) / cable_capacity
high_spread_low_util = (abs(spread) > 20) & (capacity_util < 0.5)

# Flag 4: Revenue analysis
hourly_arbitrage_eur = spread * flow  # Negative = money flowing wrong way
daily_arbitrage = hourly_arbitrage_eur.resample('D').sum()
```

**Cables to monitor (all entsoe-py calls):**

| Cable | Norwegian zone | Foreign zone | entsoe-py flow call | entsoe-py foreign price |
|-------|---------------|-------------|--------------------|-----------------------|
| Skagerrak | NO_2 | DK_1 | `query_crossborder_flows("NO_2","DK_1",...)` | `query_day_ahead_prices("DK_1",...)` |
| NorNed | NO_2 | NL | `query_crossborder_flows("NO_2","NL",...)` | `query_day_ahead_prices("NL",...)` |
| NordLink | NO_2 | DE_LU | `query_crossborder_flows("NO_2","DE_LU",...)` | `query_day_ahead_prices("DE_LU",...)` |
| North Sea Link | NO_2 | GB | `query_crossborder_flows("NO_2","GB",...)` | `query_day_ahead_prices("GB",...)` |
| NO1–SE3 | NO_1 | SE_3 | `query_crossborder_flows("NO_1","SE_3",...)` | `query_day_ahead_prices("SE_3",...)` |
| NO3–SE2 | NO_3 | SE_2 | `query_crossborder_flows("NO_3","SE_2",...)` | `query_day_ahead_prices("SE_2",...)` |
| NO4–SE1 | NO_4 | SE_1 | `query_crossborder_flows("NO_4","SE_1",...)` | `query_day_ahead_prices("SE_1",...)` |
| NO4–SE2 | NO_4 | SE_2 | `query_crossborder_flows("NO_4","SE_2",...)` | `query_day_ahead_prices("SE_2",...)` |
| NO4–FI | NO_4 | FI | `query_crossborder_flows("NO_4","FI",...)` | `query_day_ahead_prices("FI",...)` |

**Dashboard output (Streamlit Tab 5 — Cable Analysis):**
- Price comparison chart: Norwegian zone vs foreign zone (overlaid timeseries)
- Flow direction vs price spread scatter plot (should be correlated)
- Wrong-direction flow heatmap (time of day vs date)
- Daily arbitrage revenue per cable (EUR)
- Anomaly table: top wrong-direction flow events with timestamp, spread, flow, EUR impact
- Cable utilization vs price spread (are cables used when they should be?)

### Feature Selection Strategy

Apply in this order — each step filters features for the next:

**Step 1: Domain Knowledge Filter (manual)**
Remove features that don't make physical sense for electricity pricing.
Example: UV index and palm oil price are irrelevant — drop before modeling.

**Step 2: Correlation Analysis**
```python
# Remove features with >0.95 correlation to another feature (multicollinearity)
corr_matrix = df[features].corr().abs()
upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
drop_cols = [col for col in upper_tri.columns if any(upper_tri[col] > 0.95)]
```

**Step 3: XGBoost Feature Importance (built-in)**
Train a quick XGBoost model and rank features by importance.
```python
model = XGBRegressor(n_estimators=500, max_depth=6)
model.fit(X_train, y_train)
importance = pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=False)
# Keep top 20–30 features, investigate features below threshold
```

**Step 4: SHAP Values (when you want to understand WHY)**
SHAP gives per-prediction feature contributions — much more informative than
built-in importance. Use this to validate that the model makes domain sense.
```python
import shap
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test)
```

**Step 5: Recursive Feature Elimination (optional, slow)**
Iteratively removes least important features and re-trains.
Use only if you want a minimal feature set for production.

**Expected top features (based on domain knowledge):**
1. price_lag_24h, price_lag_168h (autoregressive — strongest)
2. hour_of_day, day_of_week (calendar patterns)
3. ttf_gas_close (European gas benchmark)
4. actual_load / load_forecast (demand)
5. temperature (heating demand)
6. reservoir_filling (hydro supply)
7. generation_hydro, generation_wind (supply)
8. is_weekend (demand pattern)

If the model disagrees significantly with this ranking, investigate why — it's
either a data issue or a genuine insight.

### Forecasting Methods — Build in This Order

#### Level 1: Baselines (Phase 3)

**Naive Baseline — Same Hour Last Week**
```python
y_pred = y.shift(168)  # 168 hours = 1 week
```
This is the benchmark. Every model must beat this to be useful.
Typical MAE: 8–15 EUR/MWh depending on zone and period.

**Linear Regression**
```python
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)
```
Simple, interpretable, fast. Gives you a sense of what's linearly predictable.
Probably beats naive but won't capture nonlinearities.

**Ridge/Lasso Regression**
Adds regularization to handle multicollinearity (many correlated features).
Lasso (L1) also does feature selection by shrinking unimportant features to zero.

#### Level 2: Tree-Based Models (Phase 4) — the workhorses

**XGBoost** — fast, well-documented, good defaults
```python
from xgboost import XGBRegressor
model = XGBRegressor(
    n_estimators=1000,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    early_stopping_rounds=50,
)
model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=100)
```

**LightGBM** — faster than XGBoost on large datasets, handles categoricals natively
```python
from lightgbm import LGBMRegressor
model = LGBMRegressor(
    n_estimators=1000,
    num_leaves=63,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
)
```

**CatBoost** — best with categorical features, least tuning needed
```python
from catboost import CatBoostRegressor
model = CatBoostRegressor(
    iterations=1000,
    depth=6,
    learning_rate=0.05,
    cat_features=['zone', 'day_of_week', 'month'],
)
```

**Which one to pick?** Train all three with similar hyperparameters and compare.
In practice for electricity prices, they perform similarly. LightGBM is fastest,
CatBoost needs least tuning, XGBoost has most documentation/community support.

#### Level 3: Ensemble (Phase 4)

**Simple Average Ensemble** — often beats any individual model
```python
y_pred = (pred_xgb + pred_lgbm + pred_catboost) / 3
```

**Weighted Average** — weight by validation performance
```python
# Weights inversely proportional to validation MAE
w_xgb = 1 / mae_xgb
w_lgbm = 1 / mae_lgbm
w_cat = 1 / mae_cat
total = w_xgb + w_lgbm + w_cat
y_pred = (w_xgb * pred_xgb + w_lgbm * pred_lgbm + w_cat * pred_cat) / total
```

**Stacking** — train a meta-model on base model predictions
More complex, often marginal improvement over weighted average for this use case.

#### Level 4: Advanced (Phase 6+, optional)

**Quantile Regression** — predict confidence intervals, not just point estimates
```python
# XGBoost with quantile loss
model_low = XGBRegressor(objective='reg:quantileerror', quantile_alpha=0.1)
model_mid = XGBRegressor(objective='reg:quantileerror', quantile_alpha=0.5)
model_high = XGBRegressor(objective='reg:quantileerror', quantile_alpha=0.9)
```
Gives you: "Price will be between 35 and 52 EUR/MWh with 80% confidence."

**Neural Networks (LSTM/Transformer)** — can capture complex temporal patterns
but require much more data, tuning, and compute. Not recommended until
tree-based models are fully optimized. For electricity prices, tree models
usually match or beat neural nets with much less effort.

### Hyperparameter Tuning Strategy

**Phase 3–4:** Manual tuning with sensible defaults (as shown above).
Start with defaults, then adjust if results are poor.

**When to tune more (use Optuna):**
```python
from sklearn.model_selection import TimeSeriesSplit
import optuna

def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 500, 2000),
        'max_depth': trial.suggest_int('max_depth', 4, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
    }
    model = XGBRegressor(**params)
    tscv = TimeSeriesSplit(n_splits=5)
    scores = cross_val_score(model, X_train, y_train, cv=tscv, scoring='neg_mean_absolute_error')
    return -scores.mean()

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=100)
```

Use Optuna — smarter than grid search. But only tune after features are validated.
**Feature quality matters 10x more than hyperparameter tuning.**

### Walk-Forward Validation

Standard train/test split is a minimum. For production-quality evaluation:

```
Train: 2017-01 → 2022-12  |  Val: 2023-01  (predict 1 month)
Train: 2017-01 → 2023-01  |  Val: 2023-02  (expand window)
Train: 2017-01 → 2023-02  |  Val: 2023-03  (expand window)
...
Train: 2017-01 → 2023-11  |  Val: 2023-12  (expand window)

Final test: 2024 (never touched during development)
```

This simulates real deployment — model always predicts unseen future data.
Report metrics per month to see if performance degrades seasonally.

### What Success Looks Like

| Metric | Naive baseline | Good model | Great model |
|--------|---------------|------------|-------------|
| MAE (EUR/MWh) | 10–15 | 5–8 | 3–5 |
| RMSE (EUR/MWh) | 15–25 | 8–12 | 5–8 |
| Directional accuracy | ~50% | 60–70% | >70% |

These are rough estimates for Norwegian zones. Actual numbers depend on
period (volatile 2022 energy crisis vs stable 2019) and zone (NO4 harder
due to thinner market). Price spikes are the hardest to predict — a model
with 5 EUR/MWh MAE overall might have 30+ EUR/MWh MAE during spikes.
Consider separate spike detection (anomaly detection in Phase 6).

## Working Style

- I'm learning ML — explain concepts before implementing
- Use plan mode for new features so I understand the approach
- Don't auto-commit or push without asking
- Keep code simple; avoid over-engineering
- Build incrementally: get one thing working before adding complexity

## Reference Documentation

Detailed API docs live in `docs/`. **Read the relevant doc before implementing a fetch module.**

| When working on... | Read first |
|---------------------|-----------|
| `fetch_electricity.py` (prices, load, generation, flows) | `docs/entsoe_api_reference.md` |
| `fetch_nordpool.py` (day-ahead prices, all zones) | hvakosterstrommen.no (free, no auth, ENTSO-E data) |
| `fetch_metro.py` (weather observations + forecasts) | `docs/frost_api_docs.md` |
| `fetch_commodity.py` (gas, oil, coal prices) | `docs/commodity_price_api.md` |
| `fetch_reservoir.py` (NVE reservoir filling per zone) | `docs/nve_magasin_api_reference.md` |
| `fetch_statnett.py` (physical flows, Nordic balance, frequency) | `docs/statnett_api_reference.md` |
| `fetch_fx.py` (EUR/NOK exchange rates) | Norges Bank is simple — see GUIDE.md section |
| Project setup, learning path, data strategy | `GUIDE.md` |

**When to read docs:** Always read the relevant doc before writing or modifying a fetch module.
The docs contain exact endpoint URLs, parameter names, response JSON formats, error codes,
and Norwegian-specific gotchas that are easy to get wrong without the reference.

## Current Phase

**Phase 7: Complete** — ALL PHASES IMPLEMENTED

Full pipeline from data fetching through ML forecasting to dashboard.
Data → Features → Models → Insights → Anomalies → Dashboard.

```
✅ Phase 0: Project setup
   ✅ Project structure, CLAUDE.md, GUIDE.md
   ✅ API docs written (ENTSO-E, Frost, CommodityPriceAPI, NVE Magasin, Statnett)
   ✅ Frost API client ID obtained
   ✅ CommodityPriceAPI key obtained
   ✅ ENTSO-E API key obtained and tested (all endpoints verified)

✅ Phase 1a: fetch_metro.py — Frost API weather data
   ✅ Yearly chunking + pagination (nextLink handling)
   ✅ Stations: NO_1→SN18700, NO_2→SN39040, NO_3→SN68860, NO_4→SN90450, NO_5→SN50540
   ✅ Elements: air_temperature, wind_speed, precipitation
   ✅ Tested: Bergen (NO_5) 2020–2026 → 53,712 rows, 0 missing, 600KB Parquet

✅ Phase 1b: fetch_fx.py — Norges Bank EUR/NOK
   ✅ SDMX-JSON parsing (observation index → TIME_PERIOD mapping)
   ✅ Forward-fill weekends/holidays, backfill leading NaNs
   ✅ Tested: 1,548 business days, 24KB Parquet

✅ Phase 1c: fetch_commodity.py — Gas/Oil/Coal prices
   ✅ Dual-source: CommodityPriceAPI (latest) + yfinance (historical backfill)
   ✅ Symbols: TTF-GAS (TTF=F), BRENTOIL-SPOT (BZ=F), NG-FUT (NG=F), COAL (MTF=F)
   ✅ Handles yfinance MultiIndex columns (droplevel("Ticker"))
   ✅ Tested: 1,545 rows, 16 columns (OHLC × 4), 170KB Parquet

✅ Phase 1d: fetch_nordpool.py — Day-ahead prices (primary price source)
   ✅ Day-ahead prices for all 5 zones (NO1–NO5) via hvakosterstrommen.no
   ✅ No API key required — free, public API (sources from ENTSO-E)
   ✅ Daily API calls with 0.3s rate limiting, yearly Parquet caching
   ✅ Prices in EUR/MWh + NOK/kWh + exchange rate
   ✅ Continuous data from October 2021, patchy before that
   ✅ Zone format mapping: project "NO_1" ↔ API "NO1"
   ✅ Graceful error handling: skips missing days, exponential backoff

✅ Phase 1d-alt: fetch_electricity.py — ENTSO-E prices, load, generation, flows
   ✅ Code complete: fetch_prices, fetch_load, fetch_generation,
     fetch_reservoir_filling, fetch_crossborder_flows, fetch_foreign_prices, fetch_all_entsoe
   ✅ Uses entsoe-py (v0.7.10) with yearly chunking + caching
   ✅ Graceful error when key missing (clear setup instructions)
   ✅ API key obtained and all endpoints tested (Feb 2026):
     - fetch_prices: NO_1 day-ahead prices (EUR/MWh, hourly) ✓
     - fetch_load: actual total load (MW, hourly) ✓
     - fetch_generation: per-type generation (Biomass, Fossil Gas, Hydro, Waste, Wind) ✓
     - fetch_crossborder_flows: NO_2→DK_1 flows (MW, hourly) ✓
     - fetch_foreign_prices: DK_1 prices (EUR/MWh) ✓
     - fetch_reservoir_filling: whole-Norway weekly (MWh) ✓ (NVE preferred for per-zone)
   ✅ build_features.py auto-detects ENTSOE_AVAILABLE and includes load/generation/flow features

✅ Phase 1e: fetch_reservoir.py — NVE reservoir filling per zone
   ✅ All zones (NO1–NO5) since 1995 in one API call
   ✅ Filter omrType=="EL", zone mapping (omrnr 1–5 → NO_1–NO_5)
   ✅ Benchmarks: 20-year min/max/median per week per zone
   ✅ Deviation features: filling_vs_median, filling_vs_min, filling_vs_max
   ✅ Tested: Bergen (NO_5) 2020–2026 → 320 weekly rows, 0 missing, 339KB + 20KB Parquet

✅ Phase 1f: fetch_statnett.py — Statnett physical flows & Nordic balance
   ✅ fetch_physical_flows: daily net exchange (MWh), 2,245 rows (2020–current), 35KB Parquet
   ✅ fetch_production_consumption: daily prod/cons (MWh), 2,251 rows, 62KB Parquet
   ✅ fetch_latest_overview: real-time Nordic balance (7 countries × 8 metrics)
   ✅ fetch_power_situation: per-zone assessment (NO_1–NO_5)
   ✅ fetch_frequency: grid frequency Hz (minute/second resolution)
   ✅ Note: PhysicalFlow returns aggregate net exchange only (no per-cable breakdown)
   ✅ Note: Download CSV endpoint returns empty — JSON endpoints used instead

✅ Phase 1g: fetch_yr_forecast.py — Yr Locationforecast weather
   ✅ Fetches ~9 days of weather forecasts from MET Norway (Yr/Locationforecast 2.0)
   ✅ Per-zone station coordinates (same as Frost stations)
   ✅ Variables: yr_temperature, yr_wind_speed, yr_precipitation_1h, yr_cloud_cover
   ✅ fetch_yr_forecast(zone) + fetch_all_yr_forecasts() for all 5 zones
   ✅ Caching support with TTL

✅ Phase 2: Feature engineering (build_features.py)
   ✅ Calendar, weather, commodity, FX, reservoir, Statnett features
   ✅ Nord Pool price features integrated (price_eur_mwh + lags/rolling/diff)
   ✅ EUR → NOK price conversion (price_nok_mwh, price_nok_kwh + full lag/rolling/diff)
   ✅ ENTSO-E features: load, generation (hydro/wind/total), cross-border flows, internal flows
   ✅ All-zone orchestrator with Parquet caching (~45–75 features per zone depending on cables)

✅ Phase 2.5: Statistical inference analysis (notebook 08)
   ✅ Price distribution analysis (Shapiro-Wilk, Anderson-Darling, KDE, QQ plots)
   ✅ STL seasonal decomposition (weekly cycle, seasonal strength metric)
   ✅ Kruskal-Wallis day-of-week significance tests
   ✅ Reservoir deep dive (Spearman correlation, Granger causality lags 1–8)
   ✅ Export/import pattern analysis (Mann-Whitney U, regime comparison)
   ✅ Commodity passthrough (OLS regression, rolling R², structural break detection)
   ✅ Zone decoupling (inter-zone correlation, ADF stationarity on spreads, N-S divide)
   ✅ Autocorrelation & stationarity (ADF, KPSS, ACF/PACF up to 168h lags)
   ✅ Key findings compiled with modeling recommendations for Phase 3

✅ Phase 3: ML model code + forecasting notebooks
   ✅ 3.1: src/models/forecasters.py — NaiveForecaster, ARIMAForecaster, SARIMAXForecaster, STLForecaster, ETSForecaster
   ✅ 3.2: src/models/train.py — MLPriceForecaster (XGBoost/LightGBM/CatBoost wrapper),
     prepare_ml_features(), train_ensemble(), walk_forward_validate(), forecast_with_yr()
   ✅ 3.3: src/models/evaluate.py — compute_metrics(), comparison_table(), plot_forecast(), plot_residuals()
   ✅ 3.4: notebooks/09a_price_forecasting.ipynb — NO_5 price forecasting (fundamentals-only approach)
     - Naive + SARIMA baselines, XGBoost/LightGBM/CatBoost/Ensemble
     - Walk-forward (6-fold), SHAP analysis, Yr forward forecast (daily aggregation)
     - Nord Pool patching for ENTSO-E gaps
   ✅ 3.5: notebooks/09a_all_zones_price_forecasting.ipynb — All 5 zones price comparison
     - Per-zone ML training, grand comparison table, SHAP heatmap
     - 3-month historical + forward forecast overlay for all zones
   ✅ 3.6: notebooks/09b_reservoir_forecasting.ipynb — Reservoir filling (%)
     - Target: reservoir_filling_pct, leakage prevention (drops reservoir-derived features)
     - Naive lag=52*168 (same week last year)
   ✅ 3.7: notebooks/09c_demand_forecasting.ipynb — Electricity load (MW)
     - Target: actual_load, leakage prevention (drops load_lag_* features)
     - Yr forward forecast for load prediction
   ✅ 3.8: notebooks/09d_production_forecasting.ipynb — Generation (MW)
     - Three targets: generation_total, generation_hydro, generation_wind
     - Separate leakage prevention per target
   ✅ 3.9: notebooks/09e_trade_flow_forecasting.ipynb — Net exchange (MWh)
     - Target: net_exchange_mwh, leakage prevention (drops net_balance_mwh)

   ✅ 3.10: notebooks/09f_multi_target_var.ipynb — Multi-target VAR integration
     - 5 targets: price, load, hydro gen, reservoir, net export
     - Granger causality, IRF, FEVD, comparison vs individual models

✅ Phase 4: Optuna hyperparameter tuning + advanced ensembles
   ✅ notebooks/10_optuna_tuning.ipynb — Optuna search (50 trials × 3 models)
   ✅ XGBoost/LightGBM/CatBoost objective functions with TimeSeriesSplit CV
   ✅ Default vs tuned comparison, tuned ensemble, walk-forward validation
   ✅ Per-zone tuning analysis (NO_5 vs NO_2)
   ✅ requirements.txt updated with optuna>=3.0

✅ Phase 5: ML Insights — Forstå kraftmarkedet
   ✅ notebooks/11_ml_market_insights.ipynb
   ✅ Cross-target SHAP importance heatmap (5 targets)
   ✅ Causal chain analysis (weather → reservoir → production → price)
   ✅ Zone market structure (North-South divide by feature category)
   ✅ Scenario analysis (temperature, reservoir, gas, wind per zone)
   ✅ Markov regime detection per zone
   ✅ Model reliability: error patterns, ensemble disagreement as uncertainty

✅ Phase 6: Anomaly detection + Cable Arbitrage
   ✅ src/anomaly/__init__.py, cable_arbitrage.py, detector.py
   ✅ notebooks/12_cable_arbitrage.ipynb — wrong-direction flows, capacity util, revenue
   ✅ notebooks/13_anomaly_detection.ipynb — spikes, regimes, forecast errors, multivariate
   ✅ Cable analysis: compute_cable_spreads, detect_wrong_direction_flows, build_cable_analysis
   ✅ Anomaly detection: price spikes (zscore/IQR/rolling), Isolation Forest, regime transitions

✅ Phase 7: Streamlit dashboard (7 tabs)
   ✅ app/streamlit_app.py — full 7-tab dashboard
   ✅ src/models/predict.py — inference pipeline (load_zone_model, predict_forward)
   ✅ Tab 1: Overview (current prices, 7-day history, zone comparison)
   ✅ Tab 2: Price Forecast (historical + Yr-based forward forecast)
   ✅ Tab 3: Demand/Production (load, generation mix, supply-demand balance)
   ✅ Tab 4: Reservoir (NVE filling, benchmarks, deviation analysis)
   ✅ Tab 5: Cable Arbitrage (cross-border flows, price spreads)
   ✅ Tab 6: Market Insights (feature importance, scenarios, distribution)
   ✅ Tab 7: Model Performance (availability, validation metrics, residuals)
```

## Claude Code Workflow

### How to give me tasks

**Simple task (existing module):**
```
"Implement _check_response() in fetch_metro.py"
```
CLAUDE.md gives me enough context. I'll read the file and implement.

**New module or complex task:**
```
"Read docs/entsoe_api_reference.md, then implement fetch_electricity.py
 with query_day_ahead_prices and query_load. Start with NO1, 2024 only."
```
Tell me to read the doc first so I have the API details.

**Debugging:**
```
"fetch_metro.py gives HTTP 403 when calling Frost API. Here's the error: [paste error]"
```
I'll read the code, the doc, and diagnose.

**Learning/explanation:**
```
"Explain how XGBoost feature importance works before we implement it"
```
I'll explain the concept, then we implement together.

### What I read automatically
- `CLAUDE.md` (this file) — always loaded
- Files you reference or ask me to edit

### What I don't read automatically
- `docs/*.md` — tell me to read these when relevant
- `GUIDE.md` — reference for setup, not needed for daily tasks
- `notebooks/` — exploration code, not production

### Context tips
- For a new data source: "Read docs/X first, then implement"
- For existing code: just say what to change, I can read the file
- For architecture questions: "Read CLAUDE.md and GUIDE.md, then suggest..."
- Don't paste entire docs into the prompt — just tell me which file to read

## Current Status

**Data & Features (Phase 0–2):**
- [x] Project structure, CLAUDE.md, GUIDE.md, API reference docs
- [x] fetch_metro.py — weather data (Frost API), tested with Bergen 2020–2026
- [x] fetch_fx.py — EUR/NOK exchange rates (Norges Bank), tested 2020–2026
- [x] fetch_commodity.py — gas/oil/coal (yfinance + CommodityPriceAPI), tested 2020–2026
- [x] fetch_nordpool.py — day-ahead prices via hvakosterstrommen.no (free, no auth, Oct 2021+)
- [x] fetch_electricity.py — ENTSO-E prices/load/generation/flows (fully tested, all endpoints)
- [x] fetch_reservoir.py — NVE reservoir filling per zone, tested with Bergen 2020–2026
- [x] fetch_statnett.py — physical flows, prod/cons, overview, power situation, frequency
- [x] fetch_yr_forecast.py — Yr Locationforecast weather (~9 days ahead, per zone)
- [x] build_features.py — feature engineering with all data sources (~45–75 features per zone)
- [x] Notebook 08 — Statistical inference analysis (distributions, STL, Granger, OLS, ADF/KPSS, ACF/PACF)

**ML Models & Forecasting (Phase 3):**
- [x] src/models/forecasters.py — statistical forecasters (Naive, ARIMA, SARIMA, STL, ETS)
- [x] src/models/train.py — MLPriceForecaster, train_ensemble, walk_forward_validate, forecast_with_yr
- [x] src/models/evaluate.py — compute_metrics, comparison_table, plot_forecast, plot_residuals
- [x] 09a: Price forecasting (NO_5) — XGBoost/LightGBM/CatBoost/Ensemble + SHAP + Yr forward forecast
- [x] 09a_all_zones: Price forecasting (all 5 zones) — grand comparison + forward forecast overlay
- [x] 09b: Reservoir forecasting — reservoir_filling_pct with leakage prevention
- [x] 09c: Demand forecasting — actual_load (MW) with Yr forward forecast
- [x] 09d: Production forecasting — generation_total/hydro/wind (3 sub-targets)
- [x] 09e: Trade flow forecasting — net_exchange_mwh
- [x] 09f: Multi-target VAR integration — VAR model, Granger causality, IRF, FEVD

**Tuning & Insights (Phases 4-5):**
- [x] 10: Optuna hyperparameter tuning — 50 trials × 3 models, default vs tuned comparison
- [x] 11: ML market insights — cross-target SHAP, causal chains, scenarios, regime detection

**Anomaly Detection (Phase 6):**
- [x] src/anomaly/cable_arbitrage.py — wrong-direction flows, arbitrage revenue
- [x] src/anomaly/detector.py — price spikes, forecast anomalies, Isolation Forest, regime transitions
- [x] 12: Cable arbitrage notebook — per-cable analysis, capacity utilization, revenue
- [x] 13: Anomaly detection notebook — multi-method spike detection, cross-zone coincidence

**Dashboard (Phase 7):**
- [x] app/streamlit_app.py — 7-tab Streamlit dashboard
- [x] src/models/predict.py — inference pipeline (load models, forward forecast, caching)

## Reference

See `GUIDE.md` for detailed setup instructions, data source documentation, and learning path.