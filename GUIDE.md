# Getting Started Guide — Electricity Price Forecasting

A hands-on reference for building a Nordic electricity price forecasting project with Python, ML, and Streamlit — using Claude Code as your learning partner.

---

## Table of Contents

1. [Project Structure](#1-project-structure)
2. [CLAUDE.md Best Practices](#2-claudemd-best-practices)
3. [MCP Server Setup](#3-mcp-server-setup)
4. [Claude Code Workflow Tips](#4-claude-code-workflow-tips)
5. [Settings & Permissions](#5-settings--permissions)
6. [Data Sources](#6-data-sources)
7. [ML Approach](#7-ml-approach)
8. [Streamlit App & Deployment](#8-streamlit-app--deployment)
9. [Suggested Learning Path](#9-suggested-learning-path)

---

## 1. Project Structure

A clean structure separates concerns and makes it easy for Claude Code to understand your codebase.

```
electricity_forecast/
├── CLAUDE.md                # Context for Claude Code
├── GUIDE.md                 # This file (your reference)
├── README.md                # Project readme (create when ready)
├── requirements.txt         # pip dependencies
├── .env                     # API keys (never commit — add to .gitignore)
├── .gitignore
│
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── fetch_nordpool.py    # Nord Pool price data
│   │   ├── fetch_entsoe.py      # ENTSO-E transparency platform
│   │   ├── fetch_weather.py     # Weather data (wind, sun, temp, rain)
│   │   └── fetch_gas.py         # Natural gas / oil prices
│   │
│   ├── features/
│   │   ├── __init__.py
│   │   └── build_features.py    # Feature engineering (lags, rolling stats, etc.)
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── train.py             # Training pipeline
│   │   ├── evaluate.py          # Metrics and evaluation
│   │   └── predict.py           # Inference / forecasting
│   │
│   └── utils/
│       ├── __init__.py
│       └── config.py            # Shared configuration, paths, constants
│
├── notebooks/
│   └── exploration.ipynb        # EDA and experimentation
│
├── data/
│   ├── raw/                     # Untouched downloaded data
│   └── processed/               # Cleaned, merged datasets
│
├── models/                      # Saved model artifacts (.pkl, .joblib)
│
├── app/
│   └── streamlit_app.py         # Streamlit dashboard
│
└── tests/
    └── test_features.py         # Unit tests
```

**Key principles:**
- `src/data/` — one file per data source, each with a clear `fetch_*()` function
- `src/features/` — transform raw data into model-ready features
- `src/models/` — training, evaluation, prediction as separate concerns
- `data/raw/` vs `data/processed/` — never overwrite raw data
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
| **Current status** | What's built, what's in progress — update as you go |
| **Dependencies** | Key libraries and their roles |

### Tips for effectiveness

- **Keep it under 150 lines.** Claude reads this every time — bloat slows things down.
- **Update it regularly.** As your project grows, the CLAUDE.md should reflect reality.
- **Be specific.** "Use pandas for data, scikit-learn for models" is better than "use standard libraries."
- **Include commands.** If Claude knows how to run tests (`pytest tests/`), it can verify its own work.
- **State what NOT to do.** "Don't auto-commit" or "Don't add type hints unless I ask" prevents unwanted changes.

### Example structure

```markdown
# CLAUDE.md

## Project Overview
Nordic electricity price forecasting using ML. Predicts day-ahead
prices for NO1-NO5 bidding zones using weather, fuel prices, and
historical price data.

## Architecture
- `src/data/` — data fetching (one file per source)
- `src/features/` — feature engineering
- `src/models/` — training and prediction
- `app/` — Streamlit dashboard

## Commands
- Install: `pip install -r requirements.txt`
- Run tests: `pytest tests/`
- Run app: `streamlit run app/streamlit_app.py`
- Lint: `ruff check src/`

## Conventions
- Python 3.12, use f-strings
- pandas for data manipulation
- scikit-learn and XGBoost for models
- Functions should have docstrings explaining parameters
- Keep functions small and single-purpose

## Current Status
- [x] Project structure created
- [ ] Nord Pool data fetching
- [ ] Weather data integration
- [ ] Feature engineering
- [ ] Model training
- [ ] Streamlit app

## Key Dependencies
- pandas, numpy — data handling
- requests — API calls
- scikit-learn, xgboost — modeling
- streamlit — dashboard
- python-dotenv — environment variables
```

---

## 3. MCP Server Setup

MCP (Model Context Protocol) servers give Claude Code access to external tools and data sources. Here are the most useful ones for this project.

### Recommended MCP servers

#### 1. Filesystem MCP Server
Already built-in to Claude Code — no setup needed. Claude can read/write files directly.

#### 2. Fetch MCP Server (web requests)
Lets Claude fetch URLs and API responses directly.

```json
// In ~/.claude/settings.json under "mcpServers":
{
  "fetch": {
    "command": "uvx",
    "args": ["mcp-server-fetch"]
  }
}
```

Install: `pip install mcp-server-fetch` (or use `uvx` which runs it without install).

**Use case:** Claude can fetch Nord Pool or ENTSO-E API responses to help you debug data pipelines.

#### 3. SQLite MCP Server (if you store data in SQLite)
```json
{
  "sqlite": {
    "command": "uvx",
    "args": ["mcp-server-sqlite", "--db-path", "data/electricity.db"]
  }
}
```

**Use case:** If you store processed data in SQLite, Claude can query it directly.

#### 4. GitHub MCP Server
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

**Use case:** Claude can create issues, PRs, and manage your repo without leaving the conversation.

### Where to configure

MCP servers go in `~/.claude/settings.json` (global) or `.claude/settings.json` (project-level). Project-level is better for project-specific servers like SQLite.

### Don't overdo it

Start with just the **Fetch** server. Add others only when you have a concrete need. Each MCP server adds startup time and context.

---

## 4. Claude Code Workflow Tips

### Use Plan Mode for learning

Plan mode (`/plan`) is your best friend for learning. Instead of Claude writing code immediately, it:
1. Explores your codebase
2. Proposes an approach
3. Waits for your approval

**How to use it for learning:**
```
You: "I want to fetch Nord Pool day-ahead prices. Don't write the code —
      explain the API, show me the data format, and outline the steps.
      I'll implement it myself."
```

### Effective prompting patterns

| Pattern | Example |
|---|---|
| **Explain, don't implement** | "Explain how I would connect to the ENTSO-E API" |
| **Review my code** | "Review this function — what could be improved?" |
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

This way you learn the material while Claude handles the parts that aren't educational (boilerplate, test setup, debugging obscure API errors).

---

## 5. Settings & Permissions

### Project settings (`.claude/settings.json`)

Create this file to configure Claude Code for your project:

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

**What this does:**
- `allow` — Claude can run these without asking every time
- `deny` — Claude can never run these (safety net)

### Recommended `.gitignore` additions

```
# Environment
.venv/
.env
__pycache__/

# Data (too large for git)
data/raw/
data/processed/
models/*.pkl
models/*.joblib

# IDE
.vscode/
.idea/

# Claude Code
.claude/
```

### Environment variables (`.env`)

Store API keys here, never in code:

```
ENTSOE_API_KEY=your-key-here
OPENWEATHER_API_KEY=your-key-here
```

Load them with `python-dotenv`:
```python
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("ENTSOE_API_KEY")
```

---

## 6. Data Sources

### Nord Pool (electricity prices)

| Detail | Info |
|---|---|
| **What** | Day-ahead electricity prices for Nordic bidding zones (NO1–NO5, SE1–SE4, DK1–DK2, FI) |
| **API** | No official free API. Options: ENTSO-E Transparency Platform, or `nordpool` Python package |
| **Python package** | `pip install nordpool` — community package, scrapes Nord Pool website |
| **Granularity** | Hourly prices, published day-ahead around 12:42 CET |
| **Format** | EUR/MWh (convert to NOK/kWh if needed) |

### ENTSO-E Transparency Platform (recommended for prices + generation)

| Detail | Info |
|---|---|
| **What** | Official EU transparency data: prices, generation, load, cross-border flows |
| **API** | REST API — free, requires registration |
| **Sign up** | https://transparency.entsoe.eu/ — create account, request API key |
| **Python package** | `pip install entsoe-py` — well-maintained wrapper |
| **Key endpoints** | Day-ahead prices, actual generation per type, load forecast |
| **Bidding zones** | NO-1 through NO-5 (use ENTSO-E area codes) |

```python
# Example using entsoe-py
from entsoe import EntsoePandasClient
client = EntsoePandasClient(api_key="YOUR_KEY")

# Fetch day-ahead prices for NO1
import pandas as pd
start = pd.Timestamp("2024-01-01", tz="Europe/Oslo")
end = pd.Timestamp("2024-01-31", tz="Europe/Oslo")
prices = client.query_day_ahead_prices("NO_1", start=start, end=end)
```

### Weather data

| Source | What you get | Free tier |
|---|---|---|
| **Open-Meteo** (open-meteo.com) | Temperature, wind speed, solar radiation, precipitation — historical + forecast | Free, no API key needed |
| **MET Norway (Frost API)** | Norwegian weather stations — very detailed | Free, requires registration |
| **OpenWeather** | Global weather data | Free tier: 1000 calls/day |

**Open-Meteo is the best starting point** — no API key, good historical data, and forecast data.

```python
# Example: Open-Meteo historical weather
import requests

params = {
    "latitude": 59.91,   # Oslo
    "longitude": 10.75,
    "start_date": "2024-01-01",
    "end_date": "2024-01-31",
    "hourly": "temperature_2m,wind_speed_10m,shortwave_radiation,precipitation"
}
resp = requests.get("https://archive-api.open-meteo.com/v1/archive", params=params)
data = resp.json()
```

### Fuel prices

| Source | What | Access |
|---|---|---|
| **ECB/Fred** | EUR/USD exchange rates | Free APIs |
| **EIA** (U.S. Energy Information Administration) | Natural gas (Henry Hub), oil (Brent/WTI) | Free API key |
| **Yahoo Finance** | Gas futures, oil futures | `yfinance` Python package (free) |

```python
# Example: Natural gas prices via yfinance
import yfinance as yf
gas = yf.download("NG=F", start="2024-01-01", end="2024-12-31")  # Natural gas futures
oil = yf.download("BZ=F", start="2024-01-01", end="2024-12-31")  # Brent crude futures
```

---

## 7. ML Approach

### Recommended progression (simple to complex)

#### Step 1: Baseline — Linear Regression
Start here. Predict tomorrow's price using:
- Yesterday's price (lag-1)
- Day of week
- Month

This gives you a baseline to beat and teaches you the pipeline.

#### Step 2: Feature-rich — XGBoost / LightGBM
Gradient boosted trees work very well for tabular data with mixed features:
- Price lags (1h, 24h, 48h, 168h/1 week)
- Rolling statistics (mean, std of last 24h, 7 days)
- Weather features (temperature, wind speed, solar radiation)
- Fuel prices (gas, oil — use lagged values)
- Calendar features (hour of day, day of week, month, holidays)
- Cross-border flow data (from ENTSO-E)

```python
# Example feature set
features = [
    "price_lag_1h", "price_lag_24h", "price_lag_168h",
    "price_rolling_mean_24h", "price_rolling_std_24h",
    "temperature", "wind_speed", "solar_radiation", "precipitation",
    "gas_price_lag_1d", "oil_price_lag_1d",
    "hour_of_day", "day_of_week", "month", "is_holiday",
]
```

#### Step 3: Time series models (optional)
Once comfortable:
- **Prophet** — good for seasonality, easy to use
- **LSTM / Transformer** — neural networks for sequences (more complex)

### Key ML concepts to learn along the way

| Concept | Why it matters |
|---|---|
| **Train/test split for time series** | Always split by time, never randomly. Train on past, test on future. |
| **Feature engineering** | Lags, rolling stats, and calendar features matter more than model choice. |
| **Cross-validation** | Use `TimeSeriesSplit` from scikit-learn, not regular k-fold. |
| **Evaluation metrics** | MAE (mean absolute error) and RMSE are standard for price forecasting. |
| **Overfitting** | If train error is much lower than test error, your model memorized noise. |
| **Feature importance** | XGBoost shows which features matter — use this to iterate. |

### Libraries to install

```
pip install pandas numpy scikit-learn xgboost lightgbm matplotlib seaborn
pip install entsoe-py yfinance requests python-dotenv
pip install streamlit plotly
```

---

## 8. Streamlit App & Deployment

### Start simple

Your first Streamlit app should just show a chart:

```python
# app/streamlit_app.py
import streamlit as st
import pandas as pd

st.title("Electricity Price Forecast — Nordic")

# Load your predictions
df = pd.read_csv("data/processed/predictions.csv", parse_dates=["timestamp"])

st.line_chart(df.set_index("timestamp")["predicted_price"])
```

Run with: `streamlit run app/streamlit_app.py`

### Build up incrementally

1. **v1:** Show historical prices as a chart
2. **v2:** Add weather data overlay
3. **v3:** Show model predictions vs actual
4. **v4:** Add zone selector (NO1–NO5)
5. **v5:** Add feature importance plot
6. **v6:** Add date range picker and refresh button

### Deployment options

| Platform | Cost | Notes |
|---|---|---|
| **Streamlit Community Cloud** | Free | Connect to GitHub, auto-deploys. Best for sharing. |
| **Railway / Render** | Free tier | More control, can run scheduled jobs too. |
| **Local only** | Free | Just run `streamlit run` on your machine. |

For Streamlit Community Cloud:
1. Push code to GitHub
2. Go to share.streamlit.io
3. Connect your repo
4. Set `app/streamlit_app.py` as the entry point
5. Add secrets (API keys) in the Streamlit dashboard

---

## 9. Suggested Learning Path

A step-by-step order to build this project. Each step is a natural stopping point.

### Phase 1: Foundation (week 1–2)
- [ ] Set up project structure (folders, venv, requirements.txt)
- [ ] Register for ENTSO-E API key
- [ ] Write `src/data/fetch_entsoe.py` — fetch day-ahead prices for one zone
- [ ] Save raw data to `data/raw/`
- [ ] Explore data in a notebook (`notebooks/exploration.ipynb`)

### Phase 2: More Data (week 2–3)
- [ ] Write `src/data/fetch_weather.py` — fetch from Open-Meteo
- [ ] Write `src/data/fetch_gas.py` — fetch gas/oil prices via yfinance
- [ ] Merge all data sources by timestamp
- [ ] Save merged data to `data/processed/`

### Phase 3: Feature Engineering (week 3–4)
- [ ] Create lag features (price_lag_1h, price_lag_24h, etc.)
- [ ] Create rolling statistics (mean, std over windows)
- [ ] Add calendar features (hour, day of week, month, holiday)
- [ ] Handle missing values
- [ ] Write `src/features/build_features.py`

### Phase 4: Modeling (week 4–6)
- [ ] Implement train/test split by time
- [ ] Train a linear regression baseline
- [ ] Evaluate with MAE and RMSE
- [ ] Train XGBoost model
- [ ] Compare models, analyze feature importance
- [ ] Write `src/models/train.py` and `src/models/evaluate.py`

### Phase 5: Streamlit App (week 6–7)
- [ ] Build basic Streamlit app showing price chart
- [ ] Add prediction overlay
- [ ] Add zone selector and date range
- [ ] Deploy to Streamlit Community Cloud (optional)

### Phase 6: Iterate & Improve
- [ ] Try LightGBM, compare with XGBoost
- [ ] Add more weather stations / zones
- [ ] Experiment with cross-border flow features
- [ ] Set up automated daily data refresh
- [ ] Add confidence intervals to predictions

---

## Quick Reference — Asking Claude Code for Help

```
# Learning a concept
"Explain time series cross-validation. Why can't I use random splits?"

# Getting started on a task
"I want to write the ENTSO-E data fetcher. Enter plan mode and help me
design it, but I'll write the code myself."

# Reviewing your code
"Review src/data/fetch_entsoe.py — is my error handling correct?
What edge cases am I missing?"

# Debugging
"This function returns NaN for some hours. Help me debug — don't fix it,
just help me understand why."

# Comparing approaches
"Should I use XGBoost or LightGBM for hourly price prediction?
What are the trade-offs?"
```

---

*This guide is your roadmap. Work through it at your own pace, use Claude Code to learn as you go, and update CLAUDE.md as your project evolves.*
