# Claude Code — Master Prompt

Kopier denne prompten inn i Claude Code for å starte prosjektet.
Tilpass etter behov — du kan kjøre den i én omgang eller dele opp i faser.

---

## Prompt (kopier alt under denne linjen)

```
Les CLAUDE.md, GUIDE.md, og alle filer i docs/ mappen først. Dette er et
Norwegian electricity price forecasting prosjekt. Du har all kontekst du trenger
i disse filene — API-dokumentasjon, feature engineering plan, ML-strategi,
og prosjektstruktur.

Jeg vil at du bygger hele prosjektet fra start til slutt. Slutresultatet skal gi meg:

1. PRICE FORECASTING — Prediker day-ahead strømpris (EUR/MWh) for NO1–NO5
2. ML INSIGHTS — Forstå hva som driver strømprisen (feature importance, SHAP)
3. ANOMALY DETECTION — Identifiser unormale priser og hva som forårsaket dem

## Viktige forutsetninger

- Frost API client ID er klar i .env (FROST_CLIENT_ID)
- ENTSO-E API key er IKKE klar ennå — lag all kode klar med mock/demo data
  slik at jeg bare trenger å legge inn ENTSOE_API_KEY i .env og kjøre på nytt
- CommodityPriceAPI key er IKKE klar ennå — samme approach, lag klar med mock
- Norges Bank API trenger ingen nøkkel — kan kjøres direkte
- Les docs/entsoe_api_reference.md for ENTSO-E endepunkter og parametere
- Les docs/frost_api_notes.md for Frost API detaljer
- Les docs/commodity_price_api.md for commodity API detaljer

## Bygg i denne rekkefølgen

### Steg 1: Prosjektoppsett
- Lag requirements.txt med alle nødvendige pakker
- Lag src/utils/config.py med felles konfigurasjon (paths, zone mappings, date ranges)
- Sørg for at mappestrukturen fra CLAUDE.md eksisterer
- Lag __init__.py filer der det trengs

### Steg 2: Data Fetching (alle 5 kilder)

**fetch_electricity.py** (ENTSO-E — les docs/entsoe_api_reference.md)
- Bruk entsoe-py (EntsoePandasClient)
- Funksjoner for: day-ahead prices, actual load, load forecast, generation per type,
  cross-border flows
- Hent for alle 5 norske soner (NO1–NO5) med riktige EIC-koder
- Hent også utenlandske sonepriser for kabelanalyse: DK1, DK2, SE1–SE4, DE_LU, NL, GB, FI
- Hent cross-border flows for alle kabler (se Cable Arbitrage i CLAUDE.md)
- Yearly chunks (2017–2025) med caching til data/raw/
- Parquet format, timezone-aware (Europe/Oslo)
- Må fungere med bare ENTSOE_API_KEY i .env — generer demo data hvis nøkkel mangler

**fetch_reservoir.py** (NVE Magasinstatistikk — les docs/nve_magasin_api_reference.md)
- INGEN API-nøkkel nødvendig! Kan kjøres direkte.
- Base URL: https://biapi.nve.no/magasinstatistikk
- HentOffentligData: ALL historisk data siden 1995 (~13k rader) i én GET-request
- HentOffentligDataMinMaxMedian: Min/max/median benchmarks (siste 20 år)
- HentOffentligDataSisteUke: Siste ukes data (for live dashboard)
- Filter omrType=="EL" for elspot-soner (omrnr: 1=NO1, 2=NO2, 3=NO3, 4=NO4, 5=NO5)
- Beregn ekstra features: reservoir_vs_median, reservoir_vs_min, reservoir_south
- Forward-fill weekly data til hourly for merge med priser
- Cache som data/raw/nve_reservoir_all.parquet og nve_reservoir_benchmarks.parquet
- ERSTATTER ENTSO-E reservoir filling (A72) som kun har hele Norge

**fetch_metro.py** (Frost API — les docs/frost_api_notes.md)
- Implementer alle 4 funksjoner fra skjelettet som allerede finnes
- Stasjoner: SN18700 (Oslo), SN39040 (Kristiansand), SN68860 (Trondheim),
  SN90450 (Tromsø), SN50540 (Bergen)
- Elements: air_temperature, wind_speed, precipitation_amount,
  surface_snow_thickness, cloud_area_fraction
- Caching, pagination, error handling
- Frost API key er klar — denne kan kjøres direkte

**fetch_fx.py** (Norges Bank — ingen nøkkel nødvendig)
- Hent EUR/NOK daglig kurs
- Norges Bank API er åpent — ingen autentisering
- Forward-fill for helger/helligdager
- Parquet til data/raw/

**fetch_commodity.py** (CommodityPriceAPI — les docs/commodity_price_api.md)
- Symboler: TTF-GAS, BRENTOIL-SPOT, NG-FUT, COAL
- Timeseries endpoint for backfill (yearly chunks, max 365 dager per request)
- Demo data hvis API key mangler
- Parquet til data/raw/

**Viktig for alle fetch-moduler:**
- Sjekk om data allerede finnes i data/raw/ før API-kall (caching)
- Lag en generate_demo_data() funksjon som lager realistisk syntetisk data
  for testing når API-nøkler mangler (ikke nødvendig for NVE/Norges Bank)
- Lag en fetch_all.py som kjører alle fetch-moduler i riktig rekkefølge
- Prioriter: fetch_reservoir.py og fetch_fx.py først (ingen auth!)

### Steg 3: Data Merging & Feature Engineering

**build_features.py** (les ML Strategy seksjonen i CLAUDE.md)
- Merge alle datakilder til ett dataset per sone
- Håndter ulike frekvenser (hourly prices, daily commodities, weekly reservoir)
- Forward-fill for missing values der det gir mening

Features å lage (fra CLAUDE.md):
- Price lags: 1h, 24h, 168h (1 uke)
- Rolling stats: mean/std over 24h og 168h vinduer
- Price diffs: endring vs 24h og 168h siden
- Kalender: hour, day_of_week, month, is_weekend, is_holiday, week_of_year
- Vær: temperature, wind_speed, precipitation (per sone)
- Supply: actual_load, load_forecast, generation_hydro, generation_wind
- Reservoir: filling %, endring vs forrige uke
- Commodity: ttf_gas_close, brent_oil_close, coal_close, eur_nok
- Commodity trends: ttf_gas 7-dagers endring

Lagre ferdig feature-sett som data/processed/features_{zone}.parquet

### Steg 4: Modelltrening

**train.py**
- Train/test split: 2017–2023 train, 2024 test (ALDRI random split)
- Walk-forward validation med expanding window over 2023

Modeller å trene (i denne rekkefølgen):
1. Naive baseline (same hour last week) — dette er benchmark
2. Linear Regression (Ridge) — enkel baseline
3. XGBoost — med sensible defaults fra CLAUDE.md
4. LightGBM — sammenlign med XGBoost
5. CatBoost — med categorical features (zone, day_of_week, month)
6. Ensemble — vektet gjennomsnitt av XGBoost + LightGBM + CatBoost

Lagre trente modeller til artifacts/

**evaluate.py**
- Beregn per modell: MAE, RMSE, MAPE, directional accuracy
- Peak hour MAE (timer 8–20)
- Sammenligningstabeller mellom alle modeller
- Per-måned og per-sone breakdown
- Residual-analyse (er feilene tilfeldige eller systematiske?)

### Steg 5: ML Insights — Forstå strømmarkedet

**Feature Importance Analysis**
- XGBoost built-in feature importance (gain, cover, weight)
- SHAP values for den beste modellen — summary plot, dependence plots
- Per-sone analyse: er driverne forskjellige for NO1 vs NO4?
- Temporal analysis: endrer feature importance seg over sesonger?

Spørsmål modellen skal svare på:
- Hva påvirker strømprisen mest? (ranked feature list)
- Hvordan påvirker temperatur prisen? (SHAP dependence plot)
- Er TTF-gasspris viktigere enn magasinfylling? (sammenlign)
- Hva er forskjellen mellom soner? (NO1 industri vs NO4 tynt marked)
- Påvirker helg prisen mer enn time-of-day? (interaction effects)

### Steg 6: Anomaly Detection

Implementer i src/models/ eller en egen src/anomaly/ modul:

**Price Anomaly Detection**
- XmR control charts (individuals chart) på priser og residualer
- Identifiser timer der faktisk pris avviker > 2 sigma fra predikert
- Flagg negative priser og pristopper (>95. persentil)
- Seasonal decomposition: trend + sesong + residual

### Steg 6b: Cable Arbitrage Analysis (les Cable Arbitrage Analysis i CLAUDE.md)

**Hent priser for ALLE utenlandske soner kablene går til:**
- DK1, DK2 (Danmark)
- SE1, SE2, SE3, SE4 (Sverige)
- DE_LU (Tyskland/Luxembourg)
- NL (Nederland)
- GB (Storbritannia)
- FI (Finland)

Bruk same entsoe-py client — bare query_day_ahead_prices med annen sonekode.
Hent i yearly chunks til data/raw/prices_{zone}_{year}.parquet.

**Implementer src/anomaly/cable_arbitrage.py:**

For hver kabel (NO2→DK1, NO2→NL, NO2→DE, NO2→GB, NO1→SE3, etc.):
1. Hent norsk sonepris OG utenlandsk sonepris for same timeperiode
2. Hent fysisk flyt (crossborder_flows) for kabelen
3. Beregn price spread = norsk_pris - utenlandsk_pris
4. Analyser:
   - Wrong-direction flow: eksport når norsk pris > utenlandsk pris
     (Norge selger til lavere pris = tap for norske forbrukere)
   - Import når norsk pris < utenlandsk pris
     (Norge kjøper dyrere strøm enn egen = unødvendig kostnad)
   - Kapasitetsutnyttelse: brukes kablene fullt når price spread er stort?
   - Daglig/månedlig arbitrasje-inntekt per kabel i EUR
   - Hvem tjener/taper på flyten?

**Spørsmål analysen skal svare på:**
- Flyter strømmen i "riktig" retning (fra billig til dyr sone)?
- Hvor ofte og når flyter strøm i "feil" retning?
- Hvor mye penger "lekker" ut via feil-retning flyt (EUR per dag/måned)?
- Er noen kabler verre enn andre?
- Er det tidspunkter (natt, helg, sommer) der det skjer oftere?
- Har mønsteret endret seg etter nye kabler ble åpnet (North Sea Link 2021)?
- Påvirker kabelflyt norske priser mer enn de burde?

**Root Cause Analysis for Anomalies**
- Når en anomali detekteres, vis hvilke features avvek mest fra normalt
- Eksempel: "Pris-spike 15. jan 2024 kl 18: temperatur -15°C (normalt -2°C),
  load 25,000 MW (normalt 18,000 MW), TTF-gas +12% siste 7 dager"
- SHAP force plots for anomale timer — vis hvorfor modellen bommet

**Consumption Anomaly Detection**
- XmR charts på actual_load vs load_forecast
- Identifiser unormalt høyt/lavt forbruk
- Korreler med vær, helligdager, industriell aktivitet

### Steg 7: Streamlit Dashboard

**app/streamlit_app.py**

Lag et dashboard med disse tabs:

Tab 1 — Price Forecast:
- Dropdown for å velge sone (NO1–NO5)
- Tidsserie: faktisk pris vs predikert pris
- Confidence interval (hvis quantile regression implementert)
- MAE/RMSE metrics synlig
- Date range picker

Tab 2 — Market Insights:
- Feature importance bar chart (topp 15 features)
- SHAP summary plot
- SHAP dependence plots for utvalgte features
- Sone-sammenligning

Tab 3 — Anomaly Detection:
- XmR chart med kontrollgrenser
- Tabell over detekterte anomalier med dato, sone, avvik
- Root cause breakdown for valgt anomali
- Heatmap: anomalier over tid (x: dato, y: time, farge: avvik)

Tab 4 — Data Explorer:
- Rådata-plot for alle datakilder
- Korrelasjonsheatmap mellom features
- Distribusjon av priser per sone
- Sesongmønstre (gjennomsnittspris per time/dag/måned)

Tab 5 — Cable Arbitrage Analysis:
- Prissammenligning: norsk sone vs utenlandsk sone (overlaid timeseries per kabel)
- Scatter plot: price spread (x) vs flow direction (y) — skal være korrelert
- Wrong-direction flow heatmap (x: dato, y: time, farge: avvik i EUR)
- Daglig arbitrasje-inntekt/tap per kabel (bar chart, EUR)
- Tabell: topp wrong-direction events med timestamp, spread, flow, EUR-impact
- Kapasitetsutnyttelse vs price spread (brukes kablene når de bør brukes?)
- Sammenligning før/etter North Sea Link (2021) — endret dette dynamikken?

### Steg 8: Entry Points og Dokumentasjon

**Lag disse kjørbare skriptene:**
- `scripts/fetch_all.py` — hent all data (med demo-fallback)
- `scripts/train_all.py` — tren alle modeller
- `scripts/evaluate_all.py` — kjør evaluering og generer rapport
- `scripts/run_pipeline.py` — kjør alt fra start til slutt

**README.md oppdatering:**
- Kort prosjektbeskrivelse
- Quick start (3 steg: install, fetch data, run dashboard)
- Resultater og screenshots

## Generelle regler

- Følg alle konvensjoner fra CLAUDE.md (type hints, docstrings, snake_case)
- Bruk logging, ikke print()
- All tidshåndtering med timezone-aware timestamps (Europe/Oslo)
- Parquet for all datalagring
- Forklar viktige ML-konsepter i kodekommentarer (jeg lærer)
- Lag demo/mock data for alle kilder som krever API-nøkler jeg ikke har ennå
- Test med en liten subset først (1 sone, 1 år) før full kjøring
- Bruk plan mode og forklar hvert steg før du implementerer

Start med å lage en detaljert plan, vis den til meg, og begynn deretter
med implementering steg for steg.
```

---

## Tips for å bruke denne prompten

### Alternativ 1: Kjør alt i én omgang
Kopier hele prompten over. Claude Code vil lage en plan og bygge steg for steg.
Kan ta lang tid og du mister noe kontroll.

### Alternativ 2: Del opp i faser (anbefalt)
Kopier intro + ett steg om gangen:

**Runde 1:**
"Les CLAUDE.md, GUIDE.md, og docs/. Gjør Steg 1 og Steg 2."

**Runde 2:**
"Les CLAUDE.md. Gjør Steg 3 (feature engineering). Bruk demo-data hvis API-data mangler."

**Runde 3:**
"Les CLAUDE.md (ML Strategy). Gjør Steg 4 og 5 (trening + insights)."

**Runde 4:**
"Les CLAUDE.md. Gjør Steg 6 (anomaly detection)."

**Runde 5:**
"Les CLAUDE.md. Gjør Steg 7 (Streamlit dashboard)."

### Alternativ 3: Minimal start
"Les CLAUDE.md og docs/frost_api_notes.md. Implementer fetch_metro.py med alle
4 funksjoner. Test med Bergen-stasjonen for januar 2024."

Bygg videre derfra.