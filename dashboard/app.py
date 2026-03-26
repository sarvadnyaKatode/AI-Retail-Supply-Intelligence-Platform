import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="AI Retail Supply Intelligence",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .stApp { background-color: #0f1117; color: #e0e0e0; }
    
    /* Metric cards */
    [data-testid="metric-container"] {
        background: linear-gradient(135deg, #1e2130, #252a3d);
        border: 1px solid #2e3250;
        border-radius: 12px;
        padding: 16px;
    }
    [data-testid="metric-container"] label { color: #8892b0 !important; font-size: 0.8rem !important; }
    [data-testid="metric-container"] [data-testid="stMetricValue"] { color: #64ffda !important; font-size: 1.6rem !important; }
    [data-testid="metric-container"] [data-testid="stMetricDelta"] { font-size: 0.75rem !important; }

    /* Sidebar */
    [data-testid="stSidebar"] { background-color: #161b27; border-right: 1px solid #2e3250; }
    [data-testid="stSidebar"] .stSelectbox label { color: #8892b0 !important; font-size: 0.85rem; }
    
    /* Section headers */
    .section-header {
        font-size: 1.2rem;
        font-weight: 700;
        color: #64ffda;
        margin: 1.5rem 0 0.75rem 0;
        padding-bottom: 4px;
        border-bottom: 2px solid #2e3250;
    }
    
    /* Alert boxes */
    .alert-danger {
        background: linear-gradient(135deg, #3d1a1a, #2d1010);
        border-left: 4px solid #ff6b6b;
        border-radius: 8px;
        padding: 12px 16px;
        margin: 8px 0;
        color: #ff6b6b;
    }
    .alert-warning {
        background: linear-gradient(135deg, #3d2e1a, #2d200f);
        border-left: 4px solid #ffd93d;
        border-radius: 8px;
        padding: 12px 16px;
        margin: 8px 0;
        color: #ffd93d;
    }
    .alert-success {
        background: linear-gradient(135deg, #1a3d2a, #0f2d1a);
        border-left: 4px solid #6bffb8;
        border-radius: 8px;
        padding: 12px 16px;
        margin: 8px 0;
        color: #6bffb8;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] { background-color: #161b27; border-radius: 8px; gap: 4px; }
    .stTabs [data-baseweb="tab"] { background-color: transparent; color: #8892b0; border-radius: 6px; }
    .stTabs [data-baseweb="tab"][aria-selected="true"] { background-color: #1e2130; color: #64ffda; }

    /* Title */
    .main-title {
        font-size: 2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #64ffda, #7c83fd);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    .main-subtitle { color: #8892b0; font-size: 0.95rem; margin-top: 4px; }
    
    /* General text */
    h1, h2, h3, h4 { color: #ccd6f6 !important; }
    p, li { color: #a8b2d8; }
    
    div[data-testid="stDataFrame"] { border-radius: 8px; overflow: hidden; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# DATA LOADING (self-contained, no API needed)
# ─────────────────────────────────────────────
# Streamlit Cloud runs from the project root, so we check if 'data' is in current dir
if os.path.exists(os.path.join(os.getcwd(), "data")):
    BASE_DIR = os.getcwd()
else:
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

@st.cache_data(ttl=3600)
def load_data():
    df = pd.read_csv(os.path.join(BASE_DIR, "data", "retail_demand_dataset.csv"), parse_dates=["date"])
    forecast = pd.read_csv(os.path.join(BASE_DIR, "data", "demand_forecasts.csv"), parse_dates=["date"])
    restock = pd.read_csv(os.path.join(BASE_DIR, "data", "restock_recommendations.csv"))
    return df, forecast, restock

try:
    df, forecast_df, restock_df = load_data()
    DATA_OK = True
except Exception as e:
    DATA_OK = False
    st.error(f"❌ Data load error: {e}")
    st.stop()

ALL_STORES = sorted(df["store"].unique().tolist())
ALL_PRODUCTS = sorted(df["product"].unique().tolist())

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🛒 Retail Intelligence")
    st.markdown('<p style="color:#8892b0;font-size:0.8rem;">AI-powered demand & inventory platform</p>', unsafe_allow_html=True)
    st.divider()
    
    st.markdown("#### 🏪 Filters")
    selected_store = st.selectbox("Store", ALL_STORES, index=0)
    selected_product = st.selectbox("Product", ALL_PRODUCTS, index=0)
    
    st.divider()
    st.markdown("#### 📅 Date Range")
    date_min = df["date"].min().date()
    date_max = df["date"].max().date()
    date_start = st.date_input("From", value=date_max - timedelta(days=180), min_value=date_min, max_value=date_max)
    date_end = st.date_input("To", value=date_max, min_value=date_min, max_value=date_max)
    
    st.divider()
    st.markdown('<p style="color:#8892b0;font-size:0.75rem;">📊 Dataset: 10 stores × 10 products × 3 years<br>🤖 Model: Facebook Prophet<br>🐳 Docker-ready deployment</p>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown('<div class="main-title">🛒 AI Retail Supply Intelligence</div>', unsafe_allow_html=True)
st.markdown('<div class="main-subtitle">Demand Forecasting · Inventory Optimization · Business Intelligence · Maharashtra Retail Network</div>', unsafe_allow_html=True)
st.markdown("---")

# ─────────────────────────────────────────────
# SMART ALERTS (top of page)
# ─────────────────────────────────────────────
def generate_alerts(restock_df, forecast_df, df):
    alerts = []
    # Critical restock alerts (restock > 300)
    critical = restock_df[restock_df["recommended_restock"] > 300].sort_values("recommended_restock", ascending=False)
    for _, row in critical.head(3).iterrows():
        alerts.append(("danger", f"🚨 **CRITICAL:** {row['store']} — {row['product'].title()} needs **{int(row['recommended_restock'])} units** restocked immediately"))
    
    # Demand spike: products where forecast > 1.3x historical average
    for _, row in restock_df.iterrows():
        hist_avg = df[(df["store"]==row["store"]) & (df["product"]==row["product"])]["sales"].mean() * 7
        if row["forecast_sales"] > hist_avg * 1.35:
            alerts.append(("warning", f"📈 **SPIKE DETECTED:** {row['store']} — {row['product'].title()} forecast 35%+ above average next 7 days"))
            if len(alerts) >= 5:
                break

    # Healthy stores
    healthy_count = len(restock_df[restock_df["recommended_restock"] == 0])
    if healthy_count > 0:
        alerts.append(("success", f"✅ **{healthy_count} store-product combinations** have healthy inventory levels"))
    return alerts

with st.expander("🔔 **Smart Alerts** — Live System Status", expanded=True):
    alerts = generate_alerts(restock_df, forecast_df, df)
    for alert_type, msg in alerts[:5]:
        st.markdown(f'<div class="alert-{alert_type}">{msg}</div>', unsafe_allow_html=True)

st.markdown("")

# ─────────────────────────────────────────────
# PLATFORM KPIs
# ─────────────────────────────────────────────
st.markdown('<div class="section-header">📊 Platform Business Metrics</div>', unsafe_allow_html=True)

# Compute business metrics from data
total_forecast_demand = restock_df["forecast_sales"].sum()
total_restock_units = restock_df["recommended_restock"].sum()
critical_items = len(restock_df[restock_df["recommended_restock"] > 200])

# Stockout risk reduction: items where we're recommending restock (proactively preventing stockout)
stockout_historical_rate = df["stockout"].mean() * 100
items_covered = len(restock_df[restock_df["recommended_restock"] > 0])

# Revenue impact estimate (avg unit price × prevented stockout units)
avg_price = df.set_index("product")["price"].groupby("product").mean()
revenue_at_risk = 0
for _, row in restock_df.iterrows():
    p = row["product"]
    if p in avg_price.index:
        revenue_at_risk += row["recommended_restock"] * avg_price[p]

cost_savings_est = revenue_at_risk * 0.15  # 15% margin improvement from optimized stock

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("🏪 Stores Monitored", f"{len(ALL_STORES)}", "Maharashtra network")
col2.metric("📦 7-Day Forecast Demand", f"{int(total_forecast_demand):,} units", "Across all SKUs")
col3.metric("🚨 Critical Restock Items", f"{critical_items}", "Need attention now")
col4.metric("💰 Revenue at Risk (Prevented)", f"₹{int(revenue_at_risk/1000):,}K", "Via proactive restock")
col5.metric("📉 Historical Stockout Rate", f"{stockout_historical_rate:.1f}%", "Baseline")

st.markdown("")

# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Demand Forecast",
    "📦 Restock Intelligence",
    "🗺️ Store Network Map",
    "📊 Analytics Deep Dive",
    "💰 Business Impact"
])

# ========================
# TAB 1: DEMAND FORECAST
# ========================
with tab1:
    st.markdown(f'<div class="section-header">📈 7-Day Demand Forecast — {selected_store} / {selected_product.title()}</div>', unsafe_allow_html=True)

    store_product_forecast = forecast_df[
        (forecast_df["store"] == selected_store) &
        (forecast_df["product"] == selected_product)
    ].copy()

    # Historical sales
    hist = df[
        (df["store"] == selected_store) &
        (df["product"] == selected_product) &
        (df["date"] >= pd.Timestamp(date_start)) &
        (df["date"] <= pd.Timestamp(date_end))
    ].groupby("date")["sales"].sum().reset_index()

    col1, col2 = st.columns([2, 1])
    with col1:
        if not store_product_forecast.empty and not hist.empty:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=hist["date"], y=hist["sales"],
                name="Historical Sales", line=dict(color="#7c83fd", width=1.5),
                fill="tozeroy", fillcolor="rgba(124,131,253,0.08)"
            ))
            fig.add_trace(go.Scatter(
                x=store_product_forecast["date"], y=store_product_forecast["forecast_sales"],
                name="7-Day Forecast", line=dict(color="#64ffda", width=2.5, dash="dot"),
                mode="lines+markers", marker=dict(size=8, color="#64ffda")
            ))
            fig.update_layout(
                paper_bgcolor="#0f1117", plot_bgcolor="#161b27",
                font=dict(color="#8892b0"),
                legend=dict(bgcolor="#1e2130", bordercolor="#2e3250"),
                xaxis=dict(gridcolor="#1e2130", zeroline=False),
                yaxis=dict(gridcolor="#1e2130", zeroline=False, title="Units Sold"),
                height=350, margin=dict(l=0, r=0, t=10, b=0)
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No forecast data available for this selection.")

    with col2:
        if not store_product_forecast.empty:
            st.markdown("#### 📋 7-Day Forecast")
            disp = store_product_forecast[["date", "forecast_sales"]].copy()
            disp["date"] = disp["date"].dt.strftime("%d %b")
            disp["forecast_sales"] = disp["forecast_sales"].round(1)
            disp.columns = ["Date", "Units"]
            st.dataframe(disp, hide_index=True, use_container_width=True)
            
            avg_forecast = store_product_forecast["forecast_sales"].mean()
            hist_avg = hist["sales"].mean() if not hist.empty else 0
            delta_pct = ((avg_forecast - hist_avg) / hist_avg * 100) if hist_avg > 0 else 0
            st.metric("Avg Daily Forecast", f"{avg_forecast:.0f} units", f"{delta_pct:+.1f}% vs historical avg")

    # Weekly trend across all stores for this product
    st.markdown('<div class="section-header">🏪 This Product Across All Stores</div>', unsafe_allow_html=True)
    all_store_forecast = forecast_df[forecast_df["product"] == selected_product].groupby("store")["forecast_sales"].sum().reset_index().sort_values("forecast_sales", ascending=True)
    fig2 = px.bar(all_store_forecast, x="forecast_sales", y="store", orientation="h",
                  color="forecast_sales", color_continuous_scale=["#1e2749","#7c83fd","#64ffda"],
                  labels={"forecast_sales": "7-Day Forecast (units)", "store": "Store"},
                  title=f"7-Day Total Forecast: {selected_product.title()} across all stores")
    fig2.update_layout(paper_bgcolor="#0f1117", plot_bgcolor="#161b27",
                       font=dict(color="#8892b0"), coloraxis_showscale=False,
                       height=320, margin=dict(l=0, r=0, t=40, b=0))
    st.plotly_chart(fig2, use_container_width=True)

# ========================
# TAB 2: RESTOCK INTELLIGENCE
# ========================
with tab2:
    st.markdown(f'<div class="section-header">📦 Restock Intelligence — {selected_store} / {selected_product.title()}</div>', unsafe_allow_html=True)

    item_restock = restock_df[
        (restock_df["store"] == selected_store) &
        (restock_df["product"] == selected_product)
    ]

    if not item_restock.empty:
        r = item_restock.iloc[0]
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("📊 Forecast Demand (7d)", f"{int(r['forecast_sales'])} units")
        col2.metric("🏭 Current Inventory", f"{int(r['inventory'])} units")
        col3.metric("🛡️ Safety Stock (20%)", f"{int(r['safety_stock'])} units")
        col4.metric("🚚 Recommended Restock", f"{int(r['recommended_restock'])} units",
                    "⚠️ Urgent" if r['recommended_restock'] > 200 else ("✅ Ok" if r['recommended_restock'] == 0 else "📦 Needed"))
        
        # Visual gauge
        inventory_pct = min(r['inventory'] / max(r['forecast_sales'], 1) * 100, 150)
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=inventory_pct,
            title={"text": "Inventory Coverage vs Forecast (%)", "font": {"color": "#8892b0"}},
            delta={"reference": 100, "suffix": "%"},
            gauge={
                "axis": {"range": [0, 150], "tickcolor": "#8892b0"},
                "bar": {"color": "#64ffda"},
                "bgcolor": "#1e2130",
                "steps": [
                    {"range": [0, 50], "color": "#3d1a1a"},
                    {"range": [50, 100], "color": "#2d2010"},
                    {"range": [100, 150], "color": "#1a3d2a"},
                ],
                "threshold": {"line": {"color": "#ff6b6b", "width": 3}, "thickness": 0.75, "value": 100}
            }
        ))
        fig_gauge.update_layout(paper_bgcolor="#0f1117", font=dict(color="#ccd6f6"), height=280, margin=dict(l=30, r=30, t=40, b=10))
        st.plotly_chart(fig_gauge, use_container_width=True)
    else:
        st.info("No restock data available for this selection.")

    # Full restock heatmap
    st.markdown('<div class="section-header">🔥 Restock Urgency — Full Network</div>', unsafe_allow_html=True)
    pivot = restock_df.pivot_table(index="store", columns="product", values="recommended_restock", aggfunc="sum").fillna(0)
    fig_heat = px.imshow(pivot, color_continuous_scale=["#1a3d2a","#ffd93d","#ff6b6b"],
                         labels=dict(color="Restock Units"), aspect="auto",
                         title="Restock Urgency Heatmap (units needed)")
    fig_heat.update_layout(paper_bgcolor="#0f1117", plot_bgcolor="#0f1117",
                           font=dict(color="#8892b0"), height=340,
                           margin=dict(l=0, r=0, t=40, b=0))
    st.plotly_chart(fig_heat, use_container_width=True)

    # Top 10 most urgent
    st.markdown('<div class="section-header">🚨 Top 10 Most Urgent Restock Items</div>', unsafe_allow_html=True)
    top10 = restock_df.sort_values("recommended_restock", ascending=False).head(10)[
        ["store", "product", "forecast_sales", "inventory", "recommended_restock"]
    ].round(0)
    top10.columns = ["Store", "Product", "Forecast Demand (7d)", "Current Inventory", "Restock Needed"]
    st.dataframe(top10, hide_index=True, use_container_width=True)

# ========================
# TAB 3: GEO-MAP
# ========================
with tab3:
    st.markdown('<div class="section-header">🗺️ Maharashtra Store Network — Restock Urgency Map</div>', unsafe_allow_html=True)

    store_coords = {
        "Mumbai":    {"lat": 19.0760, "lon": 72.8777},
        "Pune":      {"lat": 18.5204, "lon": 73.8567},
        "Nagpur":    {"lat": 21.1458, "lon": 79.0882},
        "Nashik":    {"lat": 20.0059, "lon": 73.7898},
        "Aurangabad":{"lat": 19.8762, "lon": 75.3433},
        "Thane":     {"lat": 19.2183, "lon": 72.9781},
        "Kolhapur":  {"lat": 16.7050, "lon": 74.2433},
        "Solapur":   {"lat": 17.6868, "lon": 75.9064},
        "Amravati":  {"lat": 20.9320, "lon": 77.7523},
        "Akola":     {"lat": 20.7002, "lon": 77.0082},
    }

    store_restock_total = restock_df.groupby("store")["recommended_restock"].sum().reset_index()
    store_restock_total.columns = ["store", "total_restock"]
    store_restock_count = restock_df[restock_df["recommended_restock"] > 0].groupby("store").size().reset_index(name="items_needing_restock")

    map_data = pd.DataFrame([
        {
            "store": s,
            "lat": coords["lat"],
            "lon": coords["lon"],
            "total_restock": store_restock_total[store_restock_total["store"]==s]["total_restock"].values[0] if s in store_restock_total["store"].values else 0,
        }
        for s, coords in store_coords.items()
    ])
    map_data = map_data.merge(store_restock_count, on="store", how="left").fillna(0)
    map_data["urgency"] = pd.cut(map_data["total_restock"], bins=3, labels=["🟢 Low", "🟡 Medium", "🔴 High"])
    map_data["size"] = map_data["total_restock"].apply(lambda x: max(x / 50, 10))

    fig_map = px.scatter_mapbox(
        map_data,
        lat="lat", lon="lon",
        size="size",
        color="total_restock",
        color_continuous_scale=["#1a3d2a", "#ffd93d", "#ff6b6b"],
        hover_name="store",
        hover_data={"total_restock": True, "items_needing_restock": True, "lat": False, "lon": False, "size": False},
        labels={"total_restock": "Total Restock Needed", "items_needing_restock": "Items Needing Restock"},
        mapbox_style="carto-darkmatter",
        zoom=6, center={"lat": 19.5, "lon": 76.0},
        title="Store Restock Urgency — Maharashtra Network",
        height=500,
    )
    fig_map.update_layout(paper_bgcolor="#0f1117", font=dict(color="#8892b0"),
                          coloraxis_colorbar=dict(title="Restock Units", tickfont=dict(color="#8892b0")),
                          margin=dict(l=0, r=0, t=40, b=0))
    st.plotly_chart(fig_map, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        fig_bar = px.bar(
            map_data.sort_values("total_restock", ascending=False),
            x="store", y="total_restock",
            color="total_restock", color_continuous_scale=["#1a3d2a","#ffd93d","#ff6b6b"],
            title="Total Restock by Store", labels={"total_restock": "Units", "store": ""}
        )
        fig_bar.update_layout(paper_bgcolor="#0f1117", plot_bgcolor="#161b27",
                               font=dict(color="#8892b0"), coloraxis_showscale=False,
                               height=280, showlegend=False, margin=dict(l=0,r=0,t=40,b=0))
        st.plotly_chart(fig_bar, use_container_width=True)
    with col2:
        st.dataframe(
            map_data[["store","total_restock","items_needing_restock"]].sort_values("total_restock", ascending=False).rename(
                columns={"store":"Store","total_restock":"Total Restock","items_needing_restock":"Items Pending"}
            ).reset_index(drop=True),
            hide_index=True, use_container_width=True
        )

# ========================
# TAB 4: ANALYTICS DEEP DIVE
# ========================
with tab4:
    st.markdown('<div class="section-header">📊 Sales Analytics — Full Dataset</div>', unsafe_allow_html=True)

    df_filtered = df[(df["date"] >= pd.Timestamp(date_start)) & (df["date"] <= pd.Timestamp(date_end))]
    
    col1, col2 = st.columns(2)
    with col1:
        # Weekend vs Weekday
        df_filtered = df_filtered.copy()
        df_filtered["day_type"] = df_filtered["is_weekend"].map({1: "Weekend", 0: "Weekday"})
        wk = df_filtered.groupby("day_type")["sales"].mean().reset_index()
        fig_wk = px.bar(wk, x="day_type", y="sales", color="day_type",
                        color_discrete_map={"Weekend":"#64ffda","Weekday":"#7c83fd"},
                        title="Avg Sales: Weekend vs Weekday",
                        labels={"sales":"Avg Units Sold","day_type":""})
        fig_wk.update_layout(paper_bgcolor="#0f1117", plot_bgcolor="#161b27",
                              font=dict(color="#8892b0"), height=280, showlegend=False,
                              margin=dict(l=0,r=0,t=40,b=0))
        st.plotly_chart(fig_wk, use_container_width=True)

    with col2:
        # Promotion impact
        df_filtered["promo_label"] = df_filtered["promotion"].map({1:"On Promotion",0:"Regular"})
        promo = df_filtered.groupby("promo_label")["sales"].mean().reset_index()
        fig_promo = px.bar(promo, x="promo_label", y="sales", color="promo_label",
                           color_discrete_map={"On Promotion":"#ffd93d","Regular":"#7c83fd"},
                           title="Avg Sales: Promotion Impact",
                           labels={"sales":"Avg Units Sold","promo_label":""})
        fig_promo.update_layout(paper_bgcolor="#0f1117", plot_bgcolor="#161b27",
                                font=dict(color="#8892b0"), height=280, showlegend=False,
                                margin=dict(l=0,r=0,t=40,b=0))
        st.plotly_chart(fig_promo, use_container_width=True)

    # Demand over time
    daily_sales = df_filtered.groupby("date")["sales"].sum().reset_index()
    fig_time = px.line(daily_sales, x="date", y="sales", title="Total Retail Demand Over Time",
                       labels={"sales":"Total Units Sold","date":""})
    fig_time.update_traces(line_color="#7c83fd", line_width=1.5,
                            fill="tozeroy", fillcolor="rgba(124,131,253,0.08)")
    fig_time.update_layout(paper_bgcolor="#0f1117", plot_bgcolor="#161b27",
                           font=dict(color="#8892b0"), height=280,
                           margin=dict(l=0,r=0,t=40,b=0))
    st.plotly_chart(fig_time, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        # Store demand heatmap
        pivot = df_filtered.pivot_table(index="store", columns="product", values="sales", aggfunc="mean").round(1)
        fig_heat = px.imshow(pivot, color_continuous_scale=["#161b27","#7c83fd","#64ffda"],
                             title="Avg Daily Sales Heatmap",
                             labels=dict(color="Avg Sales"))
        fig_heat.update_layout(paper_bgcolor="#0f1117", font=dict(color="#8892b0"),
                               height=320, margin=dict(l=0,r=0,t=40,b=0))
        st.plotly_chart(fig_heat, use_container_width=True)

    with col2:
        # Top products
        top_prod = df_filtered.groupby("product")["sales"].sum().sort_values(ascending=True).reset_index()
        fig_tp = px.bar(top_prod, x="sales", y="product", orientation="h",
                        color="sales", color_continuous_scale=["#1e2749","#7c83fd","#64ffda"],
                        title="Total Sales by Product",
                        labels={"sales":"Total Units","product":""})
        fig_tp.update_layout(paper_bgcolor="#0f1117", plot_bgcolor="#161b27",
                             font=dict(color="#8892b0"), coloraxis_showscale=False,
                             height=320, margin=dict(l=0,r=0,t=40,b=0))
        st.plotly_chart(fig_tp, use_container_width=True)

    # Demand spike detection
    st.markdown('<div class="section-header">⚡ Demand Spike Detection</div>', unsafe_allow_html=True)
    mean_s = df_filtered["sales"].mean()
    std_s = df_filtered["sales"].std()
    spikes = df_filtered[df_filtered["sales"] > mean_s + 2.5 * std_s][
        ["date","store","product","sales","promotion","festival"]
    ].sort_values("sales", ascending=False).head(15)
    if not spikes.empty:
        st.markdown('<div class="alert-warning">⚠️ Demand anomalies detected (>2.5σ above mean)</div>', unsafe_allow_html=True)
        st.dataframe(spikes.reset_index(drop=True), hide_index=True, use_container_width=True)
    else:
        st.markdown('<div class="alert-success">✅ No abnormal demand spikes in selected period</div>', unsafe_allow_html=True)

# ========================
# TAB 5: BUSINESS IMPACT
# ========================
with tab5:
    st.markdown('<div class="section-header">💰 Business Impact & ROI Analysis</div>', unsafe_allow_html=True)

    # Per-product price data
    product_prices = df.groupby("product")["price"].mean()
    
    # Business metrics per store
    store_metrics = []
    for store in ALL_STORES:
        store_data = restock_df[restock_df["store"] == store]
        store_hist = df[df["store"] == store]
        
        total_restock_units = store_data["recommended_restock"].sum()
        total_forecast = store_data["forecast_sales"].sum()
        
        # Revenue impact: restock units × avg price per product
        rev_impact = sum(
            row["recommended_restock"] * product_prices.get(row["product"], 50)
            for _, row in store_data.iterrows()
        )
        
        # Historical stockouts for this store
        stockout_rate = store_hist["stockout"].mean() * 100
        items_critical = len(store_data[store_data["recommended_restock"] > 200])
        
        store_metrics.append({
            "Store": store,
            "Forecast Demand (7d)": int(total_forecast),
            "Restock Units Needed": int(total_restock_units),
            "Revenue at Risk": f"₹{int(rev_impact/1000)}K",
            "Stockout Rate": f"{stockout_rate:.1f}%",
            "Critical Items": items_critical,
        })

    metrics_df = pd.DataFrame(store_metrics)
    st.dataframe(metrics_df, hide_index=True, use_container_width=True)

    st.markdown("")
    col1, col2 = st.columns(2)

    with col1:
        # Revenue at risk by store
        rar = []
        for store in ALL_STORES:
            store_data = restock_df[restock_df["store"] == store]
            rev = sum(
                row["recommended_restock"] * product_prices.get(row["product"], 50)
                for _, row in store_data.iterrows()
            )
            rar.append({"store": store, "revenue": rev / 1000})
        rar_df = pd.DataFrame(rar).sort_values("revenue", ascending=True)
        fig_rev = px.bar(rar_df, x="revenue", y="store", orientation="h",
                         color="revenue", color_continuous_scale=["#1e2749","#7c83fd","#ff6b6b"],
                         title="Revenue at Risk by Store (₹K)",
                         labels={"revenue": "₹K", "store": ""})
        fig_rev.update_layout(paper_bgcolor="#0f1117", plot_bgcolor="#161b27",
                              font=dict(color="#8892b0"), coloraxis_showscale=False,
                              height=320, margin=dict(l=0,r=0,t=40,b=0))
        st.plotly_chart(fig_rev, use_container_width=True)

    with col2:
        # Product-level impact
        prod_impact = []
        for prod in ALL_PRODUCTS:
            prod_data = restock_df[restock_df["product"] == prod]
            total_units = prod_data["recommended_restock"].sum()
            rev = total_units * product_prices.get(prod, 50)
            prod_impact.append({"product": prod.title(), "units": total_units, "revenue": rev / 1000})
        prod_df = pd.DataFrame(prod_impact).sort_values("revenue", ascending=True)
        fig_prod = px.bar(prod_df, x="revenue", y="product", orientation="h",
                          color="revenue", color_continuous_scale=["#1e2749","#64ffda","#ffd93d"],
                          title="Revenue at Risk by Product (₹K)",
                          labels={"revenue": "₹K", "product": ""})
        fig_prod.update_layout(paper_bgcolor="#0f1117", plot_bgcolor="#161b27",
                               font=dict(color="#8892b0"), coloraxis_showscale=False,
                               height=320, margin=dict(l=0,r=0,t=40,b=0))
        st.plotly_chart(fig_prod, use_container_width=True)

    # Key business insight boxes
    st.markdown('<div class="section-header">🎯 Key Business Insights</div>', unsafe_allow_html=True)
    
    total_rev_risk = sum(
        row["recommended_restock"] * product_prices.get(row["product"], 50)
        for _, row in restock_df.iterrows()
    )
    
    promo_lift = df[df["promotion"]==1]["sales"].mean() / df[df["promotion"]==0]["sales"].mean()
    weekend_lift = df[df["is_weekend"]==1]["sales"].mean() / df[df["is_weekend"]==0]["sales"].mean()
    
    insights = [
        ("💡", "Total Revenue Protected", f"₹{int(total_rev_risk/100000)}.{int((total_rev_risk%100000)/10000)}L by proactive restocking across all {len(ALL_STORES)} stores"),
        ("📈", "Promotion Effectiveness", f"Promotions drive {(promo_lift-1)*100:.1f}% higher sales — optimize timing for festivals (Diwali, Holi)"),
        ("📅", "Weekend Demand Spike", f"Weekend sales are {(weekend_lift-1)*100:.1f}% higher — pre-position stock by Friday"),
        ("🏪", "Highest Risk Store", f"{metrics_df.sort_values('Restock Units Needed', ascending=False).iloc[0]['Store']} has the most units needing restock this week"),
        ("📦", "Most Critical Product", f"{prod_df.sort_values('revenue', ascending=False).iloc[0]['product']} has the highest revenue impact if not restocked"),
    ]
    for icon, title, text in insights:
        st.markdown(f'<div class="alert-warning"><strong>{icon} {title}:</strong> {text}</div>', unsafe_allow_html=True)

    st.markdown("")
    st.markdown("""
    > 💡 **How metrics are calculated:**
    > - **Revenue at Risk** = Recommended restock units × Avg product price
    > - **Safety Stock** = 20% buffer above 7-day forecast demand
    > - **Restock Needed** = Forecast Demand + Safety Stock − Current Inventory (clipped at 0)
    > - **Stockout Rate** = Historical rate from synthetic dataset (simulated real-world conditions)
    """)

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center;color:#4a5568;font-size:0.8rem;">
    🛒 <strong style="color:#64ffda">AI Retail Supply Intelligence Platform</strong> &nbsp;|&nbsp;
    Built with Prophet · FastAPI · Streamlit · Plotly &nbsp;|&nbsp;
    10 Maharashtra Stores · 10 Products · 3 Years of Data &nbsp;|&nbsp;
    🐳 Docker Ready
</div>
""", unsafe_allow_html=True)