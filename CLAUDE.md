# CLAUDE.md

## Project Overview

Nordic electricity price forecasting using ML. Predicts day-ahead prices for Norwegian bidding zones (NO1–NO5) using weather data (wind, sun, temperature, rain), fuel prices (natural gas, oil), and historical price patterns. Early stage — building incrementally to learn ML concepts.

MIT licensed, authored by Aleksandersenmartin.

## Architecture

```
src/data/       — data fetching (one file per source: ENTSO-E, weather, gas/oil)
src/features/   — feature engineering (lags, rolling stats, calendar features)
src/models/     — training, evaluation, prediction
src/utils/      — shared config and helpers
app/            — Streamlit dashboard
notebooks/      — exploration and EDA (not production code)
data/raw/       — untouched downloaded data
data/processed/ — cleaned, merged datasets
tests/          — unit tests
```

## Environment

- Python 3.12
- Virtual environment: `.venv/` (activate with `source .venv/bin/activate`)
- Dependencies: `pip install -r requirements.txt`

## Commands

- Run tests: `pytest tests/`
- Run app: `streamlit run app/streamlit_app.py`
- Lint: `ruff check src/`

## Conventions

- pandas for data manipulation, scikit-learn and XGBoost for models
- One data source per file in `src/data/` with a clear `fetch_*()` function
- Time series splits only (never random) for train/test
- API keys in `.env`, loaded with `python-dotenv` — never hardcode secrets
- Functions should have docstrings explaining parameters and return values

## Working Style

- I'm learning — explain concepts before implementing
- Use plan mode for new features so I understand the approach
- Don't auto-commit or push without asking
- Keep code simple; avoid over-engineering

## Current Status

- [x] Project structure created
- [ ] Data fetching (ENTSO-E, weather, fuel prices)
- [ ] Feature engineering
- [ ] Model training (baseline → XGBoost)
- [ ] Streamlit dashboard

## Reference

See `GUIDE.md` for detailed setup instructions, data source documentation, and learning path.
