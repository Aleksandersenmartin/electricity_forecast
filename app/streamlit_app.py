"""Streamlit dashboard for Nordic electricity market forecasting.

7-tab dashboard: Overview, Price Forecast, Demand/Production,
Reservoir, Cable Arbitrage, Market Insights, Model Performance.

Launch: streamlit run app/streamlit_app.py
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

# Project imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Design System
# ---------------------------------------------------------------------------

ZONES = ["NO_1", "NO_2", "NO_3", "NO_4", "NO_5"]
ZONE_NAMES = {
    "NO_1": "Oslo (Øst-Norge)",
    "NO_2": "Kristiansand (Sør-Norge)",
    "NO_3": "Trondheim (Midt-Norge)",
    "NO_4": "Tromsø (Nord-Norge)",
    "NO_5": "Bergen (Vest-Norge)",
}
ZONE_SHORT = {
    "NO_1": "Oslo", "NO_2": "Kristiansand", "NO_3": "Trondheim",
    "NO_4": "Tromsø", "NO_5": "Bergen",
}
ZONE_COLORS = {
    "NO_1": "#4A9EFF",  # Electric blue
    "NO_2": "#FF6B6B",  # Coral red
    "NO_3": "#51CF66",  # Fresh green
    "NO_4": "#FFD43B",  # Arctic gold
    "NO_5": "#CC5DE8",  # Bergen purple
}
ZONE_EMOJI = {
    "NO_1": "🏙️", "NO_2": "🌊", "NO_3": "⚓", "NO_4": "🌌", "NO_5": "🌧️",
}

ZONE_COORDS = {
    "NO_1": {"lat": 59.94, "lon": 10.72},
    "NO_2": {"lat": 58.20, "lon": 8.08},
    "NO_3": {"lat": 63.41, "lon": 10.45},
    "NO_4": {"lat": 69.65, "lon": 18.96},
    "NO_5": {"lat": 60.38, "lon": 5.33},
}
FOREIGN_COORDS = {
    "dk1":  {"lat": 56.0, "lon": 9.5, "name": "Denmark (DK1)"},
    "nl":   {"lat": 52.3, "lon": 4.9, "name": "Netherlands"},
    "delu": {"lat": 53.5, "lon": 10.0, "name": "Germany"},
    "gb":   {"lat": 54.5, "lon": -1.5, "name": "Great Britain"},
    "se1":  {"lat": 67.0, "lon": 20.5, "name": "Sweden (SE1)"},
    "se2":  {"lat": 63.0, "lon": 17.0, "name": "Sweden (SE2)"},
    "se3":  {"lat": 59.3, "lon": 18.0, "name": "Sweden (SE3)"},
    "fi":   {"lat": 64.0, "lon": 26.0, "name": "Finland"},
}
# Cross-border cable columns per zone: (col_suffix, foreign_key)
CROSS_BORDER_CABLES = {
    "NO_1": [("se3", "se3")],
    "NO_2": [("dk1", "dk1"), ("nl", "nl"), ("delu", "delu"), ("gb", "gb")],
    "NO_3": [("se2", "se2")],
    "NO_4": [("se1", "se1"), ("se2", "se2"), ("fi", "fi")],
    "NO_5": [],
}

COLOR_POSITIVE = "#51CF66"
COLOR_NEGATIVE = "#FF6B6B"
COLOR_ACCENT = "#4A9EFF"
COLOR_CARD_BG = "#1E2130"
COLOR_BORDER = "rgba(255,255,255,0.05)"
COLOR_GRID = "rgba(255,255,255,0.06)"

CHART_HEIGHT = 380
CHART_HEIGHT_COMPACT = 280
CHART_MARGINS = dict(l=20, r=20, t=50, b=20)
CHART_FONT = dict(family="-apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif")

DATA_DIR = PROJECT_ROOT / "data" / "processed"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"


# ---------------------------------------------------------------------------
# Page config + CSS
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Nordic Power Forecast",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
/* ---- Global ---- */
.block-container { padding-top: 1.5rem; }

/* ---- KPI metric cards (inside st.columns) ---- */
.kpi-card {
    background: #1E2130;
    border-radius: 10px;
    padding: 1rem 1rem 0.8rem;
    border-top: 3px solid var(--zone-color, #4A9EFF);
    box-shadow: 0 2px 12px rgba(0,0,0,0.3);
    margin-bottom: 0.5rem;
}
.kpi-card .kpi-zone {
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: rgba(255,255,255,0.45);
    margin-bottom: 0.4rem;
}
.kpi-card .kpi-zone .zone-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    display: inline-block;
    margin-right: 0.3rem;
    vertical-align: middle;
}
.kpi-card .kpi-value {
    font-size: 1.4rem;
    font-weight: 700;
    color: #FAFAFA;
    line-height: 1.15;
}
.kpi-card .kpi-sub {
    font-size: 0.78rem;
    color: rgba(255,255,255,0.35);
    margin-top: 0.15rem;
}
.kpi-card .kpi-delta {
    font-size: 0.72rem;
    font-weight: 600;
    margin-top: 0.35rem;
    padding: 0.1rem 0.45rem;
    border-radius: 4px;
    display: inline-block;
}
.kpi-delta.positive {
    color: #51CF66;
    background: rgba(81, 207, 102, 0.1);
}
.kpi-delta.negative {
    color: #FF6B6B;
    background: rgba(255, 107, 107, 0.1);
}

/* ---- Section spacers ---- */
.section-spacer {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.06), transparent);
    margin: 1.5rem 0;
}

/* ---- Sidebar styling ---- */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #141824 0%, #0E1117 100%);
}
[data-testid="stSidebar"] .stSelectbox label {
    font-weight: 600;
    letter-spacing: 0.03em;
}

/* ---- Sidebar title ---- */
.sidebar-title {
    font-size: 1.45rem;
    font-weight: 700;
    color: #FAFAFA;
    margin-bottom: 0.2rem;
    line-height: 1.3;
}
.sidebar-subtitle {
    font-size: 0.78rem;
    color: rgba(255,255,255,0.4);
    letter-spacing: 0.04em;
    text-transform: uppercase;
    margin-bottom: 1rem;
}

/* ---- Sidebar info card ---- */
.sidebar-card {
    background: rgba(255,255,255,0.04);
    border-radius: 8px;
    padding: 0.8rem 1rem;
    margin: 0.5rem 0;
    border: 1px solid rgba(255,255,255,0.06);
}
.sidebar-card .card-label {
    font-size: 0.7rem;
    color: rgba(255,255,255,0.4);
    letter-spacing: 0.05em;
    text-transform: uppercase;
    margin-bottom: 0.15rem;
}
.sidebar-card .card-value {
    font-size: 1.1rem;
    font-weight: 600;
    color: #FAFAFA;
}
.sidebar-card .card-delta {
    font-size: 0.75rem;
    color: rgba(255,255,255,0.35);
}

/* ---- Tabs ---- */
.stTabs [data-baseweb="tab-list"] {
    gap: 0.3rem;
}
.stTabs [data-baseweb="tab"] {
    padding: 0.6rem 1rem;
    border-radius: 8px 8px 0 0;
}

/* ---- Model status badges ---- */
.badge-ok {
    display: inline-block;
    background: rgba(81, 207, 102, 0.15);
    color: #51CF66;
    padding: 0.15rem 0.5rem;
    border-radius: 4px;
    font-size: 0.8rem;
    font-weight: 600;
}
.badge-missing {
    display: inline-block;
    background: rgba(255, 107, 107, 0.15);
    color: #FF6B6B;
    padding: 0.15rem 0.5rem;
    border-radius: 4px;
    font-size: 0.8rem;
    font-weight: 600;
}

/* ---- Footer ---- */
.dashboard-footer {
    text-align: center;
    color: rgba(255,255,255,0.25);
    font-size: 0.72rem;
    padding: 2rem 0 1rem;
    letter-spacing: 0.03em;
}
.dashboard-footer a { color: rgba(255,255,255,0.35); text-decoration: none; }
.dashboard-footer a:hover { color: #4A9EFF; }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Plotly chart factory
# ---------------------------------------------------------------------------

def create_chart(
    title: str = "",
    height: int = CHART_HEIGHT,
    yaxis_title: str = "",
    xaxis_title: str = "",
    show_legend: bool = True,
) -> go.Figure:
    """Create a pre-styled Plotly figure with dark theme."""
    fig = go.Figure()
    fig.update_layout(
        template="plotly_dark",
        title=dict(text=title, font=dict(size=15, color="rgba(255,255,255,0.85)")),
        font=CHART_FONT,
        height=height,
        margin=CHART_MARGINS,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(
            title=xaxis_title,
            gridcolor=COLOR_GRID,
            zerolinecolor=COLOR_GRID,
            showgrid=True,
        ),
        yaxis=dict(
            title=yaxis_title,
            gridcolor=COLOR_GRID,
            zerolinecolor=COLOR_GRID,
            showgrid=True,
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
            font=dict(size=11),
            bgcolor="rgba(0,0,0,0)",
        ) if show_legend else dict(visible=False),
        hoverlabel=dict(
            bgcolor="#1E2130",
            bordercolor="rgba(255,255,255,0.1)",
            font=dict(size=12, color="#FAFAFA"),
        ),
    )
    return fig


# ---------------------------------------------------------------------------
# HTML helpers
# ---------------------------------------------------------------------------

def kpi_card_html(
    zone: str,
    value: str,
    subtitle: str,
    delta: str = "",
    delta_positive: bool = True,
) -> str:
    """Render a KPI card as HTML for use inside st.columns."""
    color = ZONE_COLORS.get(zone, COLOR_ACCENT)
    delta_html = ""
    if delta:
        cls = "positive" if delta_positive else "negative"
        arrow = "▲" if delta_positive else "▼"
        delta_html = f'<div class="kpi-delta {cls}">{arrow} {delta}</div>'

    return (
        f'<div class="kpi-card" style="--zone-color: {color};">'
        f'<div class="kpi-zone">'
        f'<span class="zone-dot" style="background: {color};"></span>'
        f'{zone} {ZONE_SHORT.get(zone, "")}</div>'
        f'<div class="kpi-value">{value}</div>'
        f'<div class="kpi-sub">{subtitle}</div>'
        f'{delta_html}'
        f'</div>'
    )


def align_features(X: pd.DataFrame, model) -> pd.DataFrame:
    """Align feature columns to match what the model was trained on."""
    if hasattr(model, "feature_names_"):
        expected = model.feature_names_
        # Keep only columns the model knows, add missing as 0
        missing = [c for c in expected if c not in X.columns]
        extra = [c for c in X.columns if c not in expected]
        if missing:
            for c in missing:
                X[c] = 0
        if extra:
            X = X.drop(columns=extra)
        return X[expected]
    return X


def section_spacer() -> None:
    """Render a styled section divider."""
    st.markdown('<div class="section-spacer"></div>', unsafe_allow_html=True)


def build_flow_map(
    zone_data: dict[str, pd.DataFrame],
    selected_zone: str,
) -> go.Figure | None:
    """Build interactive Scattergeo map of Norwegian power grid flows.

    Shows zone nodes sized by load, cross-border flow lines colored by
    direction (green=export, red=import), and internal flow lines.
    Uses the latest hourly snapshot from each zone's feature data.

    Args:
        zone_data: Dict mapping zone name to its feature DataFrame.
        selected_zone: Currently selected zone (highlighted on map).

    Returns:
        Plotly Figure with the flow map, or None if insufficient data.
    """
    if not zone_data:
        return None

    fig = go.Figure()

    # --- Collect latest cross-border flow snapshot ---
    cb_flows: list[dict] = []
    for zone, cables in CROSS_BORDER_CABLES.items():
        if zone not in zone_data:
            continue
        df = zone_data[zone]
        for col_suffix, foreign_key in cables:
            col_name = f"flow_{zone.lower()}_{col_suffix}"
            if col_name not in df.columns:
                continue
            val = df[col_name].dropna()
            if val.empty:
                continue
            cb_flows.append({
                "zone": zone,
                "foreign_key": foreign_key,
                "flow_mw": float(val.iloc[-1]),
            })

    # --- Collect latest internal flows (deduplicated) ---
    internal_flows: list[tuple[str, str, float]] = []
    seen_pairs: set[tuple[str, str]] = set()
    for zone in ZONES:
        if zone not in zone_data:
            continue
        df = zone_data[zone]
        for col in df.columns:
            if not col.startswith("flow_from_"):
                continue
            neighbor = col.replace("flow_from_", "").upper()
            pair = tuple(sorted([zone, neighbor]))
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            val = df[col].dropna()
            if val.empty:
                continue
            # flow_from_X in zone Y: positive = X→Y
            internal_flows.append((neighbor, zone, float(val.iloc[-1])))

    # --- Legend-only invisible traces ---
    fig.add_trace(go.Scattergeo(
        lat=[None], lon=[None], mode="markers",
        marker=dict(size=8, color=COLOR_POSITIVE),
        name="Export (from Norway)",
    ))
    fig.add_trace(go.Scattergeo(
        lat=[None], lon=[None], mode="markers",
        marker=dict(size=8, color=COLOR_NEGATIVE),
        name="Import (to Norway)",
    ))
    fig.add_trace(go.Scattergeo(
        lat=[None], lon=[None], mode="markers",
        marker=dict(size=8, color="rgba(255,255,255,0.35)"),
        name="Internal flow",
    ))

    # --- Plot internal flow lines (dimmer, behind everything) ---
    for from_z, to_z, mw in internal_flows:
        c_from = ZONE_COORDS.get(from_z)
        c_to = ZONE_COORDS.get(to_z)
        if not c_from or not c_to:
            continue
        width = max(0.8, min(4, abs(mw) / 400))
        if mw >= 0:
            label = f"{from_z} → {to_z}: {abs(mw):,.0f} MW"
        else:
            label = f"{to_z} → {from_z}: {abs(mw):,.0f} MW"
        fig.add_trace(go.Scattergeo(
            lat=[c_from["lat"], c_to["lat"]],
            lon=[c_from["lon"], c_to["lon"]],
            mode="lines",
            line=dict(width=width, color="rgba(255,255,255,0.20)"),
            hoverinfo="text",
            text=[label, label],
            showlegend=False,
            opacity=0.6,
        ))

    # --- Plot cross-border flow lines ---
    for flow in cb_flows:
        zone = flow["zone"]
        fk = flow["foreign_key"]
        mw = flow["flow_mw"]
        if fk not in FOREIGN_COORDS:
            continue
        zc = ZONE_COORDS[zone]
        fc = FOREIGN_COORDS[fk]
        is_export = mw > 0
        color = COLOR_POSITIVE if is_export else COLOR_NEGATIVE
        direction = "Export" if is_export else "Import"
        width = max(1, min(5, abs(mw) / 400))
        label = f"{zone} ↔ {FOREIGN_COORDS[fk]['name']}<br>{direction}: {abs(mw):,.0f} MW"
        fig.add_trace(go.Scattergeo(
            lat=[zc["lat"], fc["lat"]],
            lon=[zc["lon"], fc["lon"]],
            mode="lines",
            line=dict(width=width, color=color),
            hoverinfo="text",
            text=[label, label],
            showlegend=False,
            opacity=0.75,
        ))

    # --- Plot foreign zone nodes (small diamonds) ---
    plotted_foreign: set[str] = set()
    for flow in cb_flows:
        fk = flow["foreign_key"]
        if fk in plotted_foreign or fk not in FOREIGN_COORDS:
            continue
        plotted_foreign.add(fk)
        fc = FOREIGN_COORDS[fk]
        fig.add_trace(go.Scattergeo(
            lat=[fc["lat"]], lon=[fc["lon"]],
            text=[fc["name"]],
            hoverinfo="text",
            mode="markers+text",
            marker=dict(size=9, color="rgba(255,255,255,0.45)", symbol="diamond"),
            textfont=dict(size=9, color="rgba(255,255,255,0.5)"),
            textposition="top center",
            showlegend=False,
        ))

    # --- Plot Norwegian zone nodes ---
    for zone in ZONES:
        coords = ZONE_COORDS[zone]
        load_mw = None
        gen_mw = None
        if zone in zone_data:
            df = zone_data[zone]
            if "actual_load" in df.columns:
                s = df["actual_load"].dropna()
                if not s.empty:
                    load_mw = float(s.iloc[-1])
            if "generation_total" in df.columns:
                s = df["generation_total"].dropna()
                if not s.empty:
                    gen_mw = float(s.iloc[-1])

        marker_size = 20
        if load_mw is not None:
            marker_size = max(16, min(38, 16 + load_mw / 400))

        hover_parts = [f"<b>{zone} ({ZONE_SHORT[zone]})</b>"]
        if load_mw is not None:
            hover_parts.append(f"Load: {load_mw:,.0f} MW")
        if gen_mw is not None:
            hover_parts.append(f"Generation: {gen_mw:,.0f} MW")
        is_selected = zone == selected_zone
        fig.add_trace(go.Scattergeo(
            lat=[coords["lat"]], lon=[coords["lon"]],
            text=["<br>".join(hover_parts)],
            hoverinfo="text",
            marker=dict(
                size=marker_size,
                color=ZONE_COLORS[zone],
                line=dict(
                    width=3 if is_selected else 1,
                    color="white" if is_selected else "rgba(255,255,255,0.3)",
                ),
                opacity=1.0 if is_selected else 0.85,
            ),
            showlegend=False,
        ))
        # Zone label
        fig.add_trace(go.Scattergeo(
            lat=[coords["lat"] + 0.6], lon=[coords["lon"]],
            text=[zone], mode="text",
            textfont=dict(
                size=11 if is_selected else 10,
                color=ZONE_COLORS[zone],
            ),
            showlegend=False,
            hoverinfo="skip",
        ))

    # --- Map layout ---
    fig.update_layout(
        geo=dict(
            scope="europe",
            projection_type="natural earth",
            showland=True,
            landcolor="rgb(30, 33, 48)",
            showocean=True,
            oceancolor="rgb(14, 17, 23)",
            showcountries=True,
            countrycolor="rgba(255,255,255,0.08)",
            showlakes=True,
            lakecolor="rgb(14, 17, 23)",
            center=dict(lat=62, lon=12),
            lonaxis=dict(range=[-5, 32]),
            lataxis=dict(range=[50, 72]),
        ),
        template="plotly_dark",
        font=CHART_FONT,
        height=550,
        margin=dict(l=0, r=0, t=30, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.0,
            xanchor="center",
            x=0.5,
            font=dict(size=11),
            bgcolor="rgba(0,0,0,0)",
        ),
    )

    return fig


def sidebar_card(label: str, value: str, delta: str = "") -> str:
    """Render a sidebar info card."""
    delta_html = f'<div class="card-delta">{delta}</div>' if delta else ""
    return f"""
    <div class="sidebar-card">
        <div class="card-label">{label}</div>
        <div class="card-value">{value}</div>
        {delta_html}
    </div>
    """


# ---------------------------------------------------------------------------
# Data loading (cached)
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600)
def load_features(zone: str) -> pd.DataFrame | None:
    """Load feature matrix for a zone, picking the file with the latest end date."""
    yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    # Find all feature files for this zone, sort by filename descending
    # (filenames contain date ranges, so alphabetical = chronological)
    matches = sorted(
        Path(DATA_DIR).glob(f"features_{zone}_*.parquet"),
        key=lambda p: p.name,
        reverse=True,
    )
    if matches:
        df = pd.read_parquet(matches[0])
        return df[df.index <= yesterday]
    return None


@st.cache_data(ttl=3600)
def load_all_features() -> dict[str, pd.DataFrame]:
    """Load features for all zones."""
    result = {}
    for zone in ZONES:
        df = load_features(zone)
        if df is not None:
            result[zone] = df
    return result


@st.cache_data(ttl=1800)
def load_reservoir_data():
    """Load latest reservoir data."""
    try:
        from src.data.fetch_reservoir import fetch_reservoir_latest
        return fetch_reservoir_latest()
    except Exception:
        return None


@st.cache_data(ttl=1800)
def load_reservoir_zone(zone: str):
    """Load historical reservoir data for a zone."""
    try:
        from src.data.fetch_reservoir import fetch_zone_reservoir_with_benchmarks
        return fetch_zone_reservoir_with_benchmarks(zone, "2020-01-01", "2026-12-31")
    except Exception:
        return None


@st.cache_data(ttl=900)
def load_yr_forecast(zone: str):
    """Load Yr weather forecast for a zone."""
    try:
        from src.data.fetch_yr_forecast import fetch_yr_forecast
        return fetch_yr_forecast(zone, cache=True)
    except Exception:
        return None


@st.cache_data(ttl=1800)
def load_fx_rate() -> float:
    """Load latest EUR/NOK rate."""
    try:
        from src.data.fetch_fx import fetch_eur_nok_daily_filled
        fx = fetch_eur_nok_daily_filled("2025-01-01", "2026-12-31")
        if fx is not None and not fx.empty:
            return float(fx.iloc[-1])
    except Exception:
        pass
    return 11.5  # fallback


@st.cache_resource
def load_zone_model(zone: str):
    """Load saved model artifacts."""
    try:
        from src.models.predict import load_zone_model as _load
        return _load(zone, model_type="ensemble")
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown(
        '<div class="sidebar-title">⚡ Nordic Power</div>'
        '<div class="sidebar-subtitle">Electricity Market Forecast</div>',
        unsafe_allow_html=True,
    )

    section_spacer()

    selected_zone = st.selectbox(
        "Bidding Zone",
        ZONES,
        format_func=lambda z: f"{ZONE_EMOJI[z]}  {z} — {ZONE_SHORT[z]}",
        index=4,  # Default NO_5
    )

    zone_color = ZONE_COLORS[selected_zone]
    st.markdown(
        f'<div style="font-size: 0.85rem; color: rgba(255,255,255,0.6); margin-top: 0.3rem;">'
        f'<span style="color: {zone_color}; font-size: 1.1rem;">●</span> '
        f'{ZONE_NAMES[selected_zone]}</div>',
        unsafe_allow_html=True,
    )

    section_spacer()

    eur_nok = load_fx_rate()
    st.markdown(sidebar_card("EUR/NOK Exchange Rate", f"{eur_nok:.2f}"), unsafe_allow_html=True)

    # Data freshness — show actual latest date from features
    yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    st.markdown(
        sidebar_card(
            "Data Through",
            yesterday,
            f"Dashboard loaded {datetime.now().strftime('%H:%M')}",
        ),
        unsafe_allow_html=True,
    )

    section_spacer()

    with st.expander("Data Sources", expanded=False):
        st.markdown(
            "**Prices** — hvakosterstrommen.no\n\n"
            "**Weather** — MET Norway (Frost + Yr)\n\n"
            "**Reservoir** — NVE Magasinstatistikk\n\n"
            "**Grid** — ENTSO-E / Statnett\n\n"
            "**FX** — Norges Bank\n\n"
            "**Commodities** — yfinance"
        )


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "Overview",
    "Price Forecast",
    "Demand / Production",
    "Reservoir",
    "Cable Arbitrage",
    "Market Insights",
    "Model Performance",
])


# ===== TAB 1: OVERVIEW =====
with tab1:
    st.markdown("### Market Overview")

    zone_data = load_all_features()

    # --- KPI cards (one per column) ---
    kpi_cols = st.columns(5)
    for i, zone in enumerate(ZONES):
        with kpi_cols[i]:
            if zone in zone_data:
                df = zone_data[zone]
                prices = df["price_eur_mwh"].dropna() if "price_eur_mwh" in df.columns else pd.Series(dtype=float)
                if not prices.empty:
                    latest = prices.iloc[-1]
                    nok_kwh = latest * eur_nok / 1000

                    # 24h change
                    if len(prices) > 24:
                        prev = prices.iloc[-25]
                        change_pct = ((latest - prev) / prev * 100) if prev != 0 else 0
                        delta_str = f"{abs(change_pct):.1f}% (24h)"
                        delta_pos = change_pct >= 0
                    else:
                        delta_str = ""
                        delta_pos = True

                    st.markdown(
                        kpi_card_html(zone, f"{nok_kwh:.2f} kr/kWh", f"{latest:.1f} EUR/MWh", delta_str, delta_pos),
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(kpi_card_html(zone, "N/A", "No price data"), unsafe_allow_html=True)
            else:
                st.markdown(kpi_card_html(zone, "—", "No data loaded"), unsafe_allow_html=True)

    section_spacer()

    # --- 7-day price chart (area) ---
    st.markdown("#### Recent Price History (7 Days)")
    fig = create_chart(yaxis_title="EUR/MWh")
    for zone in ZONES:
        if zone not in zone_data:
            continue
        df = zone_data[zone]
        if "price_eur_mwh" not in df.columns:
            continue
        recent = df["price_eur_mwh"].dropna().last("7D")
        if not recent.empty:
            fig.add_trace(go.Scatter(
                x=recent.index, y=recent.values,
                name=f"{zone} ({ZONE_SHORT[zone]})",
                line=dict(color=ZONE_COLORS[zone], width=2),
                fill="tozeroy",
                fillcolor=f"rgba({int(ZONE_COLORS[zone][1:3], 16)},{int(ZONE_COLORS[zone][3:5], 16)},{int(ZONE_COLORS[zone][5:7], 16)},0.05)",
                hovertemplate="%{y:.1f} EUR/MWh<extra>%{fullData.name}</extra>",
            ))
    # Only fill the first trace to avoid messy overlaps
    for i, trace in enumerate(fig.data):
        if i > 0:
            trace.fill = None
            trace.fillcolor = None

    st.plotly_chart(fig, use_container_width=True)

    section_spacer()

    # --- Zone comparison table ---
    st.markdown("#### Zone Comparison (30 Days)")
    comp_rows = []
    for zone in ZONES:
        if zone not in zone_data:
            continue
        df = zone_data[zone]
        if "price_eur_mwh" not in df.columns:
            continue
        prices = df["price_eur_mwh"].dropna()
        last_30d = prices.last("30D")
        comp_rows.append({
            "Zone": f"{zone} ({ZONE_SHORT[zone]})",
            "Latest (EUR/MWh)": round(prices.iloc[-1], 2) if not prices.empty else None,
            "Latest (kr/kWh)": round(prices.iloc[-1] * eur_nok / 1000, 3) if not prices.empty else None,
            "30d Mean": round(last_30d.mean(), 2) if not last_30d.empty else None,
            "30d Min": round(last_30d.min(), 2) if not last_30d.empty else None,
            "30d Max": round(last_30d.max(), 2) if not last_30d.empty else None,
        })
    if comp_rows:
        st.dataframe(pd.DataFrame(comp_rows), hide_index=True, use_container_width=True)


# ===== TAB 2: PRICE FORECAST =====
with tab2:
    st.markdown("### Price Forecast")

    zone_data_all = load_all_features()

    # --- Historical overlay (all zones, area) ---
    st.markdown("#### 3-Month Price History (All Zones)")

    fig_forecast = create_chart(yaxis_title="EUR/MWh (daily mean)", height=420)
    for zone in ZONES:
        if zone not in zone_data_all:
            continue
        df = zone_data_all[zone]
        if "price_eur_mwh" not in df.columns:
            continue
        hist = df["price_eur_mwh"].dropna().last("90D")
        if not hist.empty:
            daily_hist = hist.resample("D").mean()
            fig_forecast.add_trace(go.Scatter(
                x=daily_hist.index, y=daily_hist.values,
                name=f"{zone} ({ZONE_SHORT[zone]})",
                line=dict(color=ZONE_COLORS[zone], width=2),
                legendgroup=zone,
                hovertemplate="%{y:.1f} EUR/MWh<extra>%{fullData.name}</extra>",
            ))

    st.plotly_chart(fig_forecast, use_container_width=True)

    section_spacer()

    # --- Forward forecast (selected zone) ---
    st.markdown(f"#### Yr-Based Forward Forecast — {selected_zone} ({ZONE_SHORT[selected_zone]})")

    yr_df = load_yr_forecast(selected_zone)
    model_info = load_zone_model(selected_zone)

    if yr_df is not None and model_info is not None and model_info["models"]:
        try:
            from src.models.train import forecast_with_yr
            first_model = next(iter(model_info["models"].values()))
            features_df = zone_data_all.get(selected_zone)
            if features_df is not None:
                last_features = features_df.iloc[-200:]
                fwd = forecast_with_yr(first_model, last_features, yr_df, eur_nok)
                if not fwd.empty:
                    daily_fwd = fwd.resample("D").agg({
                        "price_eur_mwh": "mean",
                        "price_nok_kwh": "mean",
                    })

                    # Summary metrics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.markdown(
                            sidebar_card(
                                "Mean Forecast",
                                f"{daily_fwd['price_eur_mwh'].mean():.1f} EUR/MWh",
                            ),
                            unsafe_allow_html=True,
                        )
                    with col2:
                        st.markdown(
                            sidebar_card(
                                "Consumer Price",
                                f"{daily_fwd['price_nok_kwh'].mean():.3f} kr/kWh",
                            ),
                            unsafe_allow_html=True,
                        )
                    with col3:
                        n_models = len(model_info["models"])
                        st.markdown(
                            sidebar_card(
                                "Ensemble",
                                f"{n_models} models",
                                "Weighted average",
                            ),
                            unsafe_allow_html=True,
                        )

                    # Forecast chart
                    fig_fwd = create_chart(
                        title=f"Forward Forecast — {selected_zone}",
                        yaxis_title="EUR/MWh",
                        height=CHART_HEIGHT_COMPACT,
                    )
                    fig_fwd.add_trace(go.Scatter(
                        x=daily_fwd.index,
                        y=daily_fwd["price_eur_mwh"],
                        name="Forecast",
                        line=dict(color=ZONE_COLORS[selected_zone], width=2.5),
                        fill="tozeroy",
                        fillcolor=f"rgba({int(ZONE_COLORS[selected_zone][1:3], 16)},{int(ZONE_COLORS[selected_zone][3:5], 16)},{int(ZONE_COLORS[selected_zone][5:7], 16)},0.1)",
                        hovertemplate="%{y:.1f} EUR/MWh<extra>Forecast</extra>",
                    ))
                    st.plotly_chart(fig_fwd, use_container_width=True)

                    # Forecast table
                    display_df = daily_fwd.reset_index().rename(columns={"index": "Date"})
                    display_df["price_eur_mwh"] = display_df["price_eur_mwh"].round(2)
                    display_df["price_nok_kwh"] = display_df["price_nok_kwh"].round(3)
                    display_df.columns = ["Date", "EUR/MWh", "NOK/kWh"]
                    st.dataframe(display_df, hide_index=True, use_container_width=True)
                else:
                    st.info("Forward forecast returned empty — check Yr data availability.")
            else:
                st.info("No feature data for forecast context.")
        except Exception as e:
            st.warning(f"Could not generate forecast: {e}")
    else:
        missing = []
        if yr_df is None:
            missing.append("Yr weather forecast")
        if model_info is None or not model_info.get("models"):
            missing.append("saved model artifacts")
        st.info(
            f"Forward forecast requires: {', '.join(missing)}. "
            "Run the forecasting notebooks (09a) first to save model artifacts."
        )


# ===== TAB 3: DEMAND / PRODUCTION =====
with tab3:
    st.markdown("### Demand & Production")

    df_zone = load_features(selected_zone)
    if df_zone is not None:
        # --- Supply-demand summary ---
        balance_cols = ["actual_load", "generation_total"]
        if all(c in df_zone.columns for c in balance_cols):
            latest_load = df_zone["actual_load"].dropna()
            latest_gen = df_zone["generation_total"].dropna()
            if not latest_load.empty and not latest_gen.empty:
                load_val = latest_load.iloc[-1]
                gen_val = latest_gen.iloc[-1]
                surplus = gen_val - load_val
                surplus_pct = (surplus / load_val * 100) if load_val != 0 else 0

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown(
                        sidebar_card("Current Load", f"{load_val:,.0f} MW"),
                        unsafe_allow_html=True,
                    )
                with col2:
                    st.markdown(
                        sidebar_card("Current Generation", f"{gen_val:,.0f} MW"),
                        unsafe_allow_html=True,
                    )
                with col3:
                    label = "Surplus" if surplus >= 0 else "Deficit"
                    color = COLOR_POSITIVE if surplus >= 0 else COLOR_NEGATIVE
                    st.markdown(
                        f'<div class="sidebar-card">'
                        f'<div class="card-label">{label}</div>'
                        f'<div class="card-value" style="color: {color};">{surplus:+,.0f} MW</div>'
                        f'<div class="card-delta">{surplus_pct:+.1f}% of load</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

        section_spacer()

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Electricity Load")
            if "actual_load" in df_zone.columns:
                load_data = df_zone["actual_load"].dropna().last("30D")
                if not load_data.empty:
                    fig_load = create_chart(
                        title=f"Load (30 days) — {selected_zone}",
                        yaxis_title="MW",
                        height=CHART_HEIGHT_COMPACT,
                        show_legend=False,
                    )
                    fig_load.add_trace(go.Scatter(
                        x=load_data.index, y=load_data.values,
                        line=dict(color=COLOR_ACCENT, width=1.5),
                        fill="tozeroy",
                        fillcolor="rgba(74, 158, 255, 0.08)",
                        hovertemplate="%{y:,.0f} MW<extra>Load</extra>",
                    ))
                    st.plotly_chart(fig_load, use_container_width=True)
                else:
                    st.info("No recent load data.")
            else:
                st.info("Load data not available for this zone.")

        with col2:
            st.markdown("#### Generation Mix")
            gen_cols = {
                "generation_hydro": "Hydro",
                "generation_wind": "Wind",
                "generation_total": "Total",
            }
            gen_colors = {"Hydro": "#51CF66", "Wind": "#4A9EFF", "Total": "rgba(255,255,255,0.3)"}
            available_gen = {k: v for k, v in gen_cols.items() if k in df_zone.columns}

            if available_gen:
                fig_gen = create_chart(
                    title=f"Generation (30 days) — {selected_zone}",
                    yaxis_title="MW",
                    height=CHART_HEIGHT_COMPACT,
                )
                for col, name in available_gen.items():
                    gen_data = df_zone[col].dropna().last("30D")
                    if not gen_data.empty:
                        is_total = name == "Total"
                        fig_gen.add_trace(go.Scatter(
                            x=gen_data.index, y=gen_data.values,
                            name=name,
                            line=dict(
                                color=gen_colors.get(name, "#666"),
                                width=1 if is_total else 1.5,
                                dash="dot" if is_total else "solid",
                            ),
                            hovertemplate="%{y:,.0f} MW<extra>" + name + "</extra>",
                        ))
                st.plotly_chart(fig_gen, use_container_width=True)
            else:
                st.info("Generation data not available for this zone.")

        # --- Supply-demand balance bar chart ---
        if all(c in df_zone.columns for c in balance_cols):
            section_spacer()
            st.markdown("#### Daily Generation Surplus / Deficit")
            balance = df_zone[balance_cols].dropna().last("30D")
            if not balance.empty:
                daily_balance = balance.resample("D").mean()
                daily_balance["surplus"] = daily_balance["generation_total"] - daily_balance["actual_load"]

                fig_bal = create_chart(
                    yaxis_title="MW (positive = surplus)",
                    height=CHART_HEIGHT_COMPACT,
                    show_legend=False,
                )
                colors = [COLOR_POSITIVE if v >= 0 else COLOR_NEGATIVE for v in daily_balance["surplus"]]
                fig_bal.add_trace(go.Bar(
                    x=daily_balance.index, y=daily_balance["surplus"],
                    marker_color=colors,
                    hovertemplate="%{y:+,.0f} MW<extra>Surplus/Deficit</extra>",
                ))
                st.plotly_chart(fig_bal, use_container_width=True)
    else:
        st.warning(f"No feature data for {selected_zone}")


# ===== TAB 4: RESERVOIR =====
with tab4:
    st.markdown("### Reservoir Levels")

    # --- Latest overview ---
    st.markdown("#### Current Reservoir Status")
    latest_res = load_reservoir_data()
    if latest_res is not None and not latest_res.empty:
        res_display = latest_res[latest_res["zone"].isin(ZONES)] if "zone" in latest_res.columns else latest_res
        if not res_display.empty:
            st.dataframe(res_display, hide_index=True, use_container_width=True)
    else:
        st.info("Could not fetch latest reservoir data from NVE.")

    section_spacer()

    # --- Filling gauge + history ---
    st.markdown(f"#### Reservoir Filling — {selected_zone} ({ZONE_SHORT[selected_zone]})")
    df_zone = load_features(selected_zone)
    if df_zone is not None and "reservoir_filling_pct" in df_zone.columns:
        res_data = df_zone["reservoir_filling_pct"].dropna()

        # Gauge indicator
        if not res_data.empty:
            current_fill = res_data.iloc[-1] * 100
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=current_fill,
                number=dict(suffix="%", font=dict(size=36, color="#FAFAFA")),
                gauge=dict(
                    axis=dict(range=[0, 100], tickcolor="rgba(255,255,255,0.3)"),
                    bar=dict(color=ZONE_COLORS[selected_zone]),
                    bgcolor="rgba(255,255,255,0.05)",
                    borderwidth=0,
                    steps=[
                        dict(range=[0, 30], color="rgba(255, 107, 107, 0.15)"),
                        dict(range=[30, 70], color="rgba(255, 212, 59, 0.1)"),
                        dict(range=[70, 100], color="rgba(81, 207, 102, 0.1)"),
                    ],
                ),
                title=dict(text="Current Filling", font=dict(size=14, color="rgba(255,255,255,0.5)")),
            ))
            fig_gauge.update_layout(
                height=220,
                margin=dict(l=30, r=30, t=50, b=10),
                paper_bgcolor="rgba(0,0,0,0)",
                font=CHART_FONT,
            )
            st.plotly_chart(fig_gauge, use_container_width=True)

        # Historical filling chart
        fig_res = create_chart(yaxis_title="Filling %")
        fig_res.add_trace(go.Scatter(
            x=res_data.index, y=res_data.values * 100,
            name="Actual Filling",
            line=dict(color=ZONE_COLORS[selected_zone], width=2),
            fill="tozeroy",
            fillcolor=f"rgba({int(ZONE_COLORS[selected_zone][1:3], 16)},{int(ZONE_COLORS[selected_zone][3:5], 16)},{int(ZONE_COLORS[selected_zone][5:7], 16)},0.08)",
            hovertemplate="%{y:.1f}%<extra>Filling</extra>",
        ))

        if "reservoir_vs_median" in df_zone.columns:
            median_val = (res_data - df_zone["reservoir_vs_median"].reindex(res_data.index)) * 100
            median_clean = median_val.dropna()
            if not median_clean.empty:
                fig_res.add_trace(go.Scatter(
                    x=median_clean.index, y=median_clean.values,
                    name="20-Year Median",
                    line=dict(color="rgba(255,255,255,0.3)", width=1.5, dash="dash"),
                    hovertemplate="%{y:.1f}%<extra>Median</extra>",
                ))

        st.plotly_chart(fig_res, use_container_width=True)

        # Deviation chart
        if "reservoir_vs_median" in df_zone.columns:
            dev = df_zone["reservoir_vs_median"].dropna().last("365D")
            if not dev.empty:
                section_spacer()
                st.markdown("#### Deviation from 20-Year Median (12 Months)")
                fig_dev = create_chart(
                    yaxis_title="Deviation (%)",
                    height=CHART_HEIGHT_COMPACT,
                    show_legend=False,
                )
                colors = [COLOR_POSITIVE if v >= 0 else COLOR_NEGATIVE for v in dev.values]
                fig_dev.add_trace(go.Bar(
                    x=dev.index, y=dev.values * 100,
                    marker_color=colors,
                    hovertemplate="%{y:+.1f}%<extra>vs Median</extra>",
                ))
                st.plotly_chart(fig_dev, use_container_width=True)
    else:
        st.info(f"No reservoir data for {selected_zone}")


# ===== TAB 5: CABLE ARBITRAGE =====
with tab5:
    st.markdown("### Cable Arbitrage Analysis")
    st.markdown(
        '<div style="color: rgba(255,255,255,0.5); font-size: 0.85rem; margin-bottom: 1rem;">'
        'Cross-border flow analysis: detecting wrong-direction flows '
        '(power flowing from expensive to cheap zone) and economic inefficiencies.'
        '</div>',
        unsafe_allow_html=True,
    )

    # --- Interactive flow map (all zones) ---
    zone_data_map = load_all_features()
    if zone_data_map:
        fig_map = build_flow_map(zone_data_map, selected_zone)
        if fig_map is not None:
            # Find timestamp of the snapshot
            snap_ts = None
            if selected_zone in zone_data_map:
                snap_ts = zone_data_map[selected_zone].index[-1]
            snap_label = snap_ts.strftime("%Y-%m-%d %H:%M") if snap_ts is not None else "latest"
            st.markdown(
                f"#### Power Flow Map"
                f'<span style="color: rgba(255,255,255,0.35); font-size: 0.8rem; margin-left: 0.7rem;">'
                f"Snapshot: {snap_label}</span>",
                unsafe_allow_html=True,
            )

            # Summary cards: Total Export, Total Import, Net Position
            total_export = 0.0
            total_import = 0.0
            for zone_key, cables in CROSS_BORDER_CABLES.items():
                if zone_key not in zone_data_map:
                    continue
                df_z = zone_data_map[zone_key]
                for col_suffix, _ in cables:
                    col_name = f"flow_{zone_key.lower()}_{col_suffix}"
                    if col_name in df_z.columns:
                        val = df_z[col_name].dropna()
                        if not val.empty:
                            mw = float(val.iloc[-1])
                            if mw > 0:
                                total_export += mw
                            else:
                                total_import += abs(mw)

            card_cols = st.columns(3)
            with card_cols[0]:
                st.markdown(
                    f'<div class="sidebar-card" style="border-left: 3px solid {COLOR_POSITIVE};">'
                    f'<div class="card-label">Total Cross-Border Export</div>'
                    f'<div class="card-value" style="color: {COLOR_POSITIVE};">'
                    f'{total_export:,.0f} MW</div></div>',
                    unsafe_allow_html=True,
                )
            with card_cols[1]:
                st.markdown(
                    f'<div class="sidebar-card" style="border-left: 3px solid {COLOR_NEGATIVE};">'
                    f'<div class="card-label">Total Cross-Border Import</div>'
                    f'<div class="card-value" style="color: {COLOR_NEGATIVE};">'
                    f'{total_import:,.0f} MW</div></div>',
                    unsafe_allow_html=True,
                )
            with card_cols[2]:
                net = total_export - total_import
                net_color = COLOR_POSITIVE if net >= 0 else COLOR_NEGATIVE
                net_label = "Net Exporter" if net >= 0 else "Net Importer"
                st.markdown(
                    f'<div class="sidebar-card" style="border-left: 3px solid {net_color};">'
                    f'<div class="card-label">Norway Net Position</div>'
                    f'<div class="card-value" style="color: {net_color};">'
                    f'{net_label}</div>'
                    f'<div class="card-delta">{net:+,.0f} MW</div></div>',
                    unsafe_allow_html=True,
                )

            st.plotly_chart(fig_map, use_container_width=True)

        section_spacer()

    df_zone = load_features(selected_zone)
    if df_zone is not None:
        flow_cols = [c for c in df_zone.columns if c.startswith("flow_")]
        price_cols = [c for c in df_zone.columns if c.startswith("price_") and "eur_mwh" in c and c != "price_eur_mwh"]

        if flow_cols:
            # --- Net position card ---
            if "total_net_export" in df_zone.columns:
                net_exp = df_zone["total_net_export"].dropna().last("30D")
                if not net_exp.empty:
                    mean_export = net_exp.mean()
                    is_exporter = mean_export > 0
                    st.markdown(
                        f'<div class="sidebar-card" style="border-left: 3px solid '
                        f'{COLOR_POSITIVE if is_exporter else COLOR_NEGATIVE};">'
                        f'<div class="card-label">30-Day Net Position</div>'
                        f'<div class="card-value" style="color: '
                        f'{COLOR_POSITIVE if is_exporter else COLOR_NEGATIVE};">'
                        f'{"Net Exporter" if is_exporter else "Net Importer"}</div>'
                        f'<div class="card-delta">{mean_export:+,.0f} MWh avg</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

            section_spacer()

            # --- Flow chart ---
            st.markdown(f"#### Cross-Border Flows — {selected_zone}")
            fig_flows = create_chart(yaxis_title="MW (positive = export)", height=400)

            cable_colors = ["#4A9EFF", "#FF6B6B", "#51CF66", "#FFD43B", "#CC5DE8", "#FF922B"]
            for idx, col in enumerate(flow_cols[:6]):
                flow_data = df_zone[col].dropna().last("90D")
                if not flow_data.empty:
                    daily_flow = flow_data.resample("D").mean()
                    cable_name = col.replace("flow_", "").upper().replace("_", " → ")
                    fig_flows.add_trace(go.Scatter(
                        x=daily_flow.index, y=daily_flow.values,
                        name=cable_name,
                        line=dict(width=1.5, color=cable_colors[idx % len(cable_colors)]),
                        hovertemplate="%{y:+,.0f} MW<extra>" + cable_name + "</extra>",
                    ))

            # Zero reference line
            fig_flows.add_shape(
                type="line", y0=0, y1=0,
                x0=0, x1=1, xref="paper",
                line=dict(color="rgba(255,255,255,0.15)", width=1, dash="dash"),
            )
            st.plotly_chart(fig_flows, use_container_width=True)

        else:
            st.info(
                f"No cross-border flow data for {selected_zone}. "
                "Cable analysis requires ENTSO-E data (run fetch_electricity.py)."
            )

        # --- Price spread analysis ---
        if price_cols:
            section_spacer()
            st.markdown("#### Price Spreads vs Foreign Zones")
            no_price = df_zone["price_eur_mwh"].dropna().last("90D")

            spread_cols = st.columns(min(len(price_cols[:4]), 4))
            for idx, pcol in enumerate(price_cols[:4]):
                foreign_price = df_zone[pcol].dropna().last("90D")
                if not foreign_price.empty and not no_price.empty:
                    spread = (no_price - foreign_price.reindex(no_price.index)).dropna()
                    if not spread.empty:
                        daily_spread = spread.resample("D").mean()
                        mean_spread = daily_spread.mean()
                        zone_label = pcol.replace("price_", "").replace("_eur_mwh", "").upper()
                        is_cheaper = mean_spread < 0
                        with spread_cols[idx]:
                            st.markdown(
                                f'<div class="sidebar-card">'
                                f'<div class="card-label">vs {zone_label}</div>'
                                f'<div class="card-value" style="color: '
                                f'{COLOR_POSITIVE if is_cheaper else COLOR_NEGATIVE};">'
                                f'{mean_spread:+.1f}</div>'
                                f'<div class="card-delta">EUR/MWh (90d avg)</div>'
                                f'</div>',
                                unsafe_allow_html=True,
                            )
    else:
        st.warning(f"No data for {selected_zone}")


# ===== TAB 6: MARKET INSIGHTS =====
with tab6:
    st.markdown("### Market Insights")

    df_zone = load_features(selected_zone)
    if df_zone is None:
        st.warning(f"No data for {selected_zone}")
    else:
        # --- Feature importance ---
        st.markdown(f"#### Feature Importance — {selected_zone}")

        model_info = load_zone_model(selected_zone)
        if model_info and model_info["models"]:
            first_model = next(iter(model_info["models"].values()))
            try:
                importance = first_model.feature_importance()
                top_20 = importance.head(20)

                # Gradient colors by rank
                n = len(top_20)
                bar_colors = [
                    f"rgba(74, 158, 255, {0.4 + 0.6 * (n - i) / n})"
                    for i in range(n)
                ]

                fig_imp = create_chart(
                    title=f"Top 20 Features — {selected_zone}",
                    xaxis_title="Importance (gain)",
                    height=500,
                    show_legend=False,
                )
                fig_imp.add_trace(go.Bar(
                    x=top_20.values[::-1],
                    y=top_20.index[::-1],
                    orientation="h",
                    marker_color=bar_colors[::-1],
                    hovertemplate="%{x:.4f}<extra>%{y}</extra>",
                ))
                fig_imp.update_layout(margin=dict(l=180, r=20, t=50, b=20))
                st.plotly_chart(fig_imp, use_container_width=True)
            except Exception as e:
                st.info(f"Could not compute feature importance: {e}")
        else:
            st.info("Load model artifacts first (run notebooks 09a).")

        section_spacer()

        # --- Scenario analysis ---
        st.markdown("#### What-If Scenario Analysis")
        st.markdown(
            '<div style="color: rgba(255,255,255,0.5); font-size: 0.85rem; margin-bottom: 1rem;">'
            'Adjust conditions to see estimated price impact on the validation period.'
            '</div>',
            unsafe_allow_html=True,
        )

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            temp_change = st.slider("Temperature (°C)", -5.0, 5.0, 0.0, 0.5)
        with col2:
            reservoir_mult = st.slider("Reservoir (%)", -20, 20, 0, 5)
        with col3:
            gas_change = st.slider("TTF Gas (%)", -50, 100, 0, 10)
        with col4:
            wind_change = st.slider("Wind Speed (%)", -30, 50, 0, 10)

        if any([temp_change != 0, reservoir_mult != 0, gas_change != 0, wind_change != 0]):
            if model_info and model_info["models"]:
                try:
                    from src.models.train import prepare_ml_features
                    df_val = df_zone.loc["2024-12-31":"2025-06-30"].iloc[1:]
                    X_val, y_val = prepare_ml_features(df_val)
                    first_model = next(iter(model_info["models"].values()))
                    X_val = align_features(X_val, first_model)

                    base_pred = first_model.predict(X_val).mean()
                    X_scenario = X_val.copy()

                    if temp_change != 0 and "temperature" in X_scenario.columns:
                        X_scenario["temperature"] += temp_change
                    if reservoir_mult != 0 and "reservoir_filling_pct" in X_scenario.columns:
                        X_scenario["reservoir_filling_pct"] *= (1 + reservoir_mult / 100)
                    if gas_change != 0 and "ttf_gas_close" in X_scenario.columns:
                        X_scenario["ttf_gas_close"] *= (1 + gas_change / 100)
                    if wind_change != 0 and "wind_speed" in X_scenario.columns:
                        X_scenario["wind_speed"] *= (1 + wind_change / 100)

                    scenario_pred = first_model.predict(X_scenario).mean()
                    delta = scenario_pred - base_pred

                    impact_color = COLOR_NEGATIVE if delta > 0 else COLOR_POSITIVE
                    st.markdown(
                        f'<div class="sidebar-card" style="border-left: 3px solid {impact_color};">'
                        f'<div class="card-label">Estimated Price Impact</div>'
                        f'<div class="card-value" style="color: {impact_color};">'
                        f'{delta:+.2f} EUR/MWh</div>'
                        f'<div class="card-delta">Base: {base_pred:.1f} → '
                        f'Scenario: {scenario_pred:.1f} EUR/MWh</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                except Exception as e:
                    st.info(f"Scenario failed: {e}")
            else:
                st.info("Need saved model for scenario analysis.")

        section_spacer()

        # --- Price distribution ---
        st.markdown("#### Price Distribution")
        if "price_eur_mwh" in df_zone.columns:
            prices = df_zone["price_eur_mwh"].dropna()

            fig_dist = create_chart(
                xaxis_title="EUR/MWh",
                yaxis_title="Frequency",
                height=CHART_HEIGHT_COMPACT,
                show_legend=False,
            )
            fig_dist.add_trace(go.Histogram(
                x=prices.values, nbinsx=100,
                marker_color=ZONE_COLORS[selected_zone],
                opacity=0.75,
                hovertemplate="%{x:.0f} EUR/MWh: %{y} hours<extra></extra>",
            ))
            st.plotly_chart(fig_dist, use_container_width=True)


# ===== TAB 7: MODEL PERFORMANCE =====
with tab7:
    st.markdown("### Model Performance")
    st.markdown(
        '<div style="color: rgba(255,255,255,0.5); font-size: 0.85rem; margin-bottom: 1rem;">'
        'Fundamentals-only ML models (no autoregressive price lags). '
        'Must beat naive baseline (same hour last week).'
        '</div>',
        unsafe_allow_html=True,
    )

    # --- Model availability (styled badges) ---
    st.markdown("#### Model Availability")
    badge_rows = []
    for zone in ZONES:
        row_html = f'<span style="color: {ZONE_COLORS[zone]}; font-weight: 600;">{zone}</span> '
        for mt in ["xgboost", "lightgbm", "catboost"]:
            path = ARTIFACTS_DIR / f"model_{zone}_{mt}.joblib"
            if path.exists():
                row_html += f'<span class="badge-ok">{mt}</span> '
            else:
                row_html += f'<span class="badge-missing">{mt}</span> '
        weights_path = ARTIFACTS_DIR / f"weights_{zone}.joblib"
        if weights_path.exists():
            row_html += '<span class="badge-ok">weights</span>'
        else:
            row_html += '<span class="badge-missing">weights</span>'
        badge_rows.append(row_html)

    st.markdown(
        '<div style="line-height: 2.2;">' + '<br>'.join(badge_rows) + '</div>',
        unsafe_allow_html=True,
    )

    section_spacer()

    # --- Validation results ---
    st.markdown(f"#### Validation Metrics — {selected_zone}")
    model_info = load_zone_model(selected_zone)
    df_zone = load_features(selected_zone)

    if model_info and model_info["models"] and df_zone is not None:
        try:
            from src.models.train import prepare_ml_features
            from src.models.evaluate import compute_metrics

            df_val = df_zone.loc["2024-12-31":"2025-06-30"].iloc[1:]
            X_val, y_val = prepare_ml_features(df_val)

            results = []
            for mt, model in model_info["models"].items():
                X_aligned = align_features(X_val.copy(), model)
                pred = model.predict(X_aligned)
                metrics = compute_metrics(y_val, pred)
                results.append({"Model": mt, **metrics})

            if results:
                results_df = pd.DataFrame(results).sort_values("mae")
                st.dataframe(results_df, hide_index=True, use_container_width=True)

                # MAE bar chart with naive baseline
                model_colors = {
                    "xgboost": "#4A9EFF",
                    "lightgbm": "#51CF66",
                    "catboost": "#FFD43B",
                }
                fig_mae = create_chart(
                    yaxis_title="MAE (EUR/MWh)",
                    height=320,
                )
                fig_mae.add_trace(go.Bar(
                    x=results_df["Model"],
                    y=results_df["mae"],
                    marker_color=[model_colors.get(m, COLOR_ACCENT) for m in results_df["Model"]],
                    hovertemplate="%{y:.2f} EUR/MWh<extra>%{x}</extra>",
                    name="Validation MAE",
                ))

                # Naive baseline reference (typical ~10-15 EUR/MWh)
                fig_mae.add_shape(
                    type="line", y0=12, y1=12,
                    x0=-0.5, x1=len(results_df) - 0.5,
                    line=dict(color=COLOR_NEGATIVE, width=1.5, dash="dash"),
                )
                fig_mae.add_annotation(
                    x=len(results_df) - 1, y=12,
                    text="Naive baseline (~12)",
                    showarrow=False,
                    font=dict(size=11, color=COLOR_NEGATIVE),
                    yshift=12,
                )
                st.plotly_chart(fig_mae, use_container_width=True)

                section_spacer()

                # Residual distribution
                st.markdown(f"#### Residual Distribution — Best Model")
                best_model = results_df.iloc[0]["Model"]
                best_pred = model_info["models"][best_model].predict(
                    align_features(X_val.copy(), model_info["models"][best_model])
                )
                residuals = y_val - best_pred

                fig_resid = create_chart(
                    xaxis_title="Residual (EUR/MWh)",
                    yaxis_title="Frequency",
                    height=CHART_HEIGHT_COMPACT,
                    show_legend=False,
                )
                fig_resid.add_trace(go.Histogram(
                    x=residuals.values, nbinsx=80,
                    marker_color=model_colors.get(best_model, COLOR_ACCENT),
                    opacity=0.75,
                    hovertemplate="%{x:.1f} EUR/MWh: %{y}<extra></extra>",
                ))
                # Mean residual line
                mean_resid = residuals.mean()
                fig_resid.add_shape(
                    type="line",
                    x0=mean_resid, x1=mean_resid,
                    y0=0, y1=1, yref="paper",
                    line=dict(color="rgba(255,255,255,0.4)", width=1, dash="dash"),
                )
                st.plotly_chart(fig_resid, use_container_width=True)

        except Exception as e:
            st.warning(f"Validation failed: {e}")
    else:
        st.info(
            "No saved model artifacts found. Run the forecasting notebooks "
            "(09a_all_zones_price_forecasting.ipynb) and save models to artifacts/ first."
        )

    section_spacer()

    # Methodology
    with st.expander("Methodology"):
        st.markdown("""
        **Approach:** Fundamentals-only ML forecasting (no autoregressive price lags)

        **Models:** XGBoost, LightGBM, CatBoost with inverse-MAE weighted ensemble

        **Validation:** Walk-forward with expanding window (6 folds x 720 hours)

        **Features (~45-75 per zone):**
        - Calendar: hour, day-of-week, month, holidays
        - Weather: temperature, wind speed, precipitation
        - Commodity: TTF gas, Brent oil, natural gas futures
        - Reservoir: NVE filling per zone with benchmarks
        - Grid: ENTSO-E load, generation (hydro/wind), cross-border flows
        - FX: EUR/NOK exchange rate

        **Key metric:** MAE (EUR/MWh) — must beat naive baseline (same hour last week)
        """)


# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------

st.markdown(
    '<div class="dashboard-footer">'
    'Nordic Power Forecast&ensp;|&ensp;'
    'ENTSO-E&ensp;&middot;&ensp;MET Norway&ensp;&middot;&ensp;NVE&ensp;&middot;&ensp;Norges Bank'
    '&ensp;|&ensp;MIT License'
    '</div>',
    unsafe_allow_html=True,
)
