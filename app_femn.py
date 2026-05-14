"""
LC/MC FeMn Alloy Forecasting Dashboard  ·  v2.0
=================================================
Includes strict hover isolation and Carbon Penalty VIU Math.
"""

from __future__ import annotations
import os
import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG & THEME
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="FeMn Price Forecasting", page_icon="📈", layout="wide")

C_ACTUAL   = "#FF9800"
C_HYBRID   = "#2196F3"
C_FUTURE   = "#4CAF50"
C_MARKET   = "#9C27B0"
C_GRID     = "#EEEEEE"
C_TEXT     = "#333333"
C_CI       = "rgba(76, 175, 80, 0.12)"

HORIZON_OPTIONS = {"4 weeks (1 month)": 4, "12 weeks (3 months)": 12, "26 weeks (6 months)": 26, "52 weeks (1 year)": 52, "104 weeks (2 years)": 104, "156 weeks (3 years)": 156}

def _layout(title: str, y_title: str = "Price", height: int = 460) -> dict:
    return dict(template="plotly_white", paper_bgcolor="white", plot_bgcolor="#FAFAFA", font=dict(family="sans-serif", size=12, color=C_TEXT), title=dict(text=title, font=dict(size=16, color="#111"), x=0.01), legend=dict(bgcolor="rgba(255,255,255,0.8)", bordercolor="#CCC", borderwidth=1), xaxis=dict(showgrid=True, gridcolor=C_GRID, zeroline=False), yaxis=dict(showgrid=True, gridcolor=C_GRID, zeroline=False, title=y_title), hovermode="x unified", height=height, margin=dict(l=55, r=20, t=55, b=40))

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR & DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚙️ Alloy Selection")
    alloy_choice = st.radio("Select Target Alloy:", ["LC FeMn (80%)", "MC FeMn (75%)"])
    alloy_code = "LC" if "LC" in alloy_choice else "MC"
    output_dir = f"./outputs_{alloy_code}"
    
    st.divider()
    display_window = st.slider("History to show (weeks)", 52, 520, 260, step=26)
    horizon_label  = st.selectbox("Forecast horizon", list(HORIZON_OPTIONS.keys()), index=2)
    show_market = st.checkbox("Show Market Price", value=True)
    show_dual = st.checkbox("Show Dual-Axis Chart", value=False)

try:
    hist = pd.read_csv(f"{output_dir}/historical_predictions.csv", index_col=0, parse_dates=True)
    future = pd.read_csv(f"{output_dir}/future_forecast.csv", index_col=0, parse_dates=True).iloc[:HORIZON_OPTIONS[horizon_label]]
    fi = pd.read_csv(f"{output_dir}/feature_importance.csv").sort_values("importance", ascending=False).head(15)
    with open(f"{output_dir}/model_metadata.json") as f: meta = json.load(f)
except FileNotFoundError:
    st.error(f"⛔ Data for {alloy_code} not found. Please run `python pipeline_femn.py --alloy {alloy_code}` first.")
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# MAIN DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────

st.markdown(f"## 📈 {alloy_choice} Price Forecasting Engine")

c1, c2, c3, c4, c5 = st.columns(5)
last_idx = float(hist["actual"].iloc[-1])
last_real = float(hist["real_price"].iloc[-1])
nxt_real = float(future["real_price"].iloc[0])
end_real = float(future["real_price"].iloc[-1])

c1.metric("Last Index", f"{last_idx:.2f}")
c2.metric("Last Model Price", f"₹{last_real:.2f}/Kg")
c3.metric("Next-Wk Forecast", f"₹{nxt_real:.2f}", delta=f"{((nxt_real-last_real)/last_real)*100:+.1f}%")
c4.metric("End Forecast", f"₹{end_real:.2f}", delta=f"{len(future)} wks ahead", delta_color="off")
if "market_price" in hist.columns and not hist["market_price"].dropna().empty:
    c5.metric("Last Market Price", f"₹{hist['market_price'].dropna().iloc[-1]:.2f}/Kg")
else: c5.metric("Scaling Factor", f"{meta.get('scaling_factor', 0):.4f}")

tab1, tab2, tab3 = st.tabs(["📉 Price Forecast", "⚙️ Drivers & Importance", "🏭 Steel VIU Calculator"])

# ══ TAB 1: STRICT ISOLATION FORECAST CHARTS ══════════════════════════════════
with tab1:
    h = hist[hist.index >= hist.index[-1] - pd.DateOffset(weeks=display_window)].copy()
    
    # 1. Main Price Chart
    fig1 = go.Figure()
    full_x = list(h.index) + list(future.index)
    N, M = len(h), len(future)
    fmt = "₹%{y:.2f}"

    # Historical (Padded with None to disappear in future)
    fig1.add_trace(go.Scatter(
        x=full_x, y=list(h["real_price"]) + [None]*M, name="Model Price", line=dict(color=C_HYBRID, width=3),
        hovertemplate="<b>%{x|%B %Y}</b><br>%{x|%Y-%m-%d}<br>Model Price: " + fmt + "<extra></extra>"
    ))
    if show_market and "market_price" in h.columns:
        fig1.add_trace(go.Scatter(
            x=full_x, y=list(h["market_price"]) + [None]*M, name="Market Price", line=dict(color=C_MARKET, width=2.5),
            hovertemplate="<b>%{x|%B %Y}</b><br>%{x|%Y-%m-%d}<br>Market Price: " + fmt + "<extra></extra>"
        ))

    # Future (Padded with None to disappear in past)
    conn_val = h["real_price"].iloc[-1]
    fut_y = [None]*(N-1) + [conn_val] + list(future["real_price"])
    
    idx_arr = np.arange(1, M + 1)
    sigma = np.std(future["real_price"]) * 0.015 * idx_arr
    ci_up = list(future["real_price"] + 1.96 * sigma)
    ci_dn = list(future["real_price"] - 1.96 * sigma)
    
    ci_upper_fill = [conn_val] + ci_up
    ci_lower_fill = [conn_val] + ci_dn
    
    # Draw CI Band visually 
    ci_x = [h.index[-1]] + list(future.index)
    fig1.add_trace(go.Scatter(
        x=ci_x + ci_x[::-1], y=ci_upper_fill + ci_lower_fill[::-1], 
        fill="toself", fillcolor=C_CI, line=dict(width=0), name="95% CI", showlegend=True, hoverinfo="skip"
    ))

    # Future Line with Max/Min Tooltips
    upper_arr = [None]*(N-1) + [conn_val] + ci_up
    lower_arr = [None]*(N-1) + [conn_val] + ci_dn
    custom_data = np.stack((upper_arr, lower_arr), axis=-1)
    
    htemplate = ("<b>%{x|%B %Y}</b><br>%{x|%Y-%m-%d}<br>Future Forecast Price: " + fmt + 
                 "<br>Max (95% CI): " + fmt.replace("%{y", "%{customdata[0]") + 
                 "<br>Min (95% CI): " + fmt.replace("%{y", "%{customdata[1]") + "<extra></extra>")

    fig1.add_trace(go.Scatter(x=full_x, y=fut_y, name="Future Forecast Price", line=dict(color=C_FUTURE, width=3), customdata=custom_data, hovertemplate=htemplate))

    fig1.update_layout(**_layout(f"{alloy_choice} Price Trajectory", "Price (Rs/Kg)", 500))
    st.plotly_chart(fig1, use_container_width=True)

    # 2. Dual Axis Chart
    if show_dual:
        fig2 = make_subplots(specs=[[{"secondary_y": True}]])
        
        hist_idx_y = list(h["hybrid_prediction"]) + [None]*M
        fig2.add_trace(go.Scatter(x=full_x, y=hist_idx_y, name="Index", line=dict(color=C_HYBRID, width=3), hovertemplate="<b>%{x|%B %Y}</b><br>%{x|%Y-%m-%d}<br>Index: %{y:.2f}<extra></extra>"), secondary_y=False)
        
        hist_rp_y = list(h["real_price"]) + [None]*M
        fig2.add_trace(go.Scatter(x=full_x, y=hist_rp_y, name="Model Price", line=dict(color=C_ACTUAL, width=2.5, dash="dash"), hovertemplate="<b>%{x|%B %Y}</b><br>%{x|%Y-%m-%d}<br>Model Price: ₹%{y:.2f}<extra></extra>"), secondary_y=True)

        fut_idx_y = [None]*(N-1) + [h["hybrid_prediction"].iloc[-1]] + list(future["predicted_index"])
        fig2.add_trace(go.Scatter(x=full_x, y=fut_idx_y, name="Future Index", line=dict(color="#64B5F6", width=3), hovertemplate="<b>%{x|%B %Y}</b><br>%{x|%Y-%m-%d}<br>Future Index: %{y:.2f}<extra></extra>"), secondary_y=False)
        
        fut_rp_y2 = [None]*(N-1) + [h["real_price"].iloc[-1]] + list(future["real_price"])
        fig2.add_trace(go.Scatter(x=full_x, y=fut_rp_y2, name="Future Price", line=dict(color="#FF5722", width=2.5, dash="dash"), hovertemplate="<b>%{x|%B %Y}</b><br>%{x|%Y-%m-%d}<br>Future Price: ₹%{y:.2f}<extra></extra>"), secondary_y=True)

        fig2.update_layout(**_layout("Index vs Price — Dual Axis Comparison", "Price Index", 400))
        fig2.update_yaxes(title_text="Price Index (Solid Lines)", secondary_y=False)
        fig2.update_yaxes(title_text="Price Rs/Kg (Dashed Lines)", secondary_y=True, showgrid=False)
        st.plotly_chart(fig2, use_container_width=True)

# ══ TAB 2: DRIVERS ═══════════════════════════════════════════════════════════
with tab2:
    fc1, fc2 = st.columns(2)
    fig_bar = go.Figure(go.Bar(x=fi["importance"][::-1], y=fi["feature"][::-1], orientation="h", marker=dict(color="#3F51B5", opacity=0.85), hovertemplate="Feature: %{y}<br>Score: %{x}<extra></extra>"))
    fig_bar.update_layout(**_layout("Top Driver Importance", "Score", 400))
    fig_bar.update_layout(yaxis=dict(tickfont=dict(size=10), showgrid=False), margin=dict(l=150, r=20, t=55, b=40))
    fc1.plotly_chart(fig_bar, use_container_width=True)

    top_d = fi.head(7).copy()
    other_val = fi.iloc[7:]["importance"].sum()
    if other_val > 0: top_d = pd.concat([top_d, pd.DataFrame([{"feature": "Other Variables", "importance": other_val}])])
    top_d["percent"] = (top_d["importance"] / top_d["importance"].sum()) * 100
    fig_pie = go.Figure(data=[go.Pie(labels=top_d["feature"], values=top_d["percent"], hole=0.5, textinfo='label+percent', marker=dict(colors=["#1E88E5", "#43A047", "#E53935", "#FB8C00", "#8E24AA", "#00ACC1", "#7CB342", "#9E9E9E"]), hovertemplate="%{label}<br>Dependence: %{percent:.1f}%<extra></extra>")])
    fig_pie.update_layout(title=dict(text="% Dependence of FeMn Price on Drivers", font=dict(size=16, color="#111"), x=0.5), template="plotly_white", paper_bgcolor="white", height=400, showlegend=False)
    fc2.plotly_chart(fig_pie, use_container_width=True)

# ══ TAB 3: STEEL CALCULATOR WITH CARBON PENALTY ══════════════════════════════
with tab3:
    st.markdown("### 🏭 Value-in-Use (VIU) Calculator with Carbon Penalties")
    st.caption("LC/MC FeMn introduces carbon into the steel bath. This calculator adds a financial penalty for the extra oxygen and time required to burn off that carbon.")
    
    default_mn = 80.0 if alloy_code == "LC" else 75.0
    default_c  = 0.5  if alloy_code == "LC" else 1.5
    
    cc1, cc2, cc3 = st.columns(3)
    mn_pct = cc1.number_input("Manganese Content (%)", value=default_mn, step=0.5)
    c_pct  = cc2.number_input("Carbon Content (%)", value=default_c, step=0.1)
    c_pen  = cc3.number_input("Carbon Penalty (₹ per 1% C)", value=15.0, step=1.0)
    
    eff_mn_frac = mn_pct / 100.0
    
    h_viu = h.copy()
    f_viu = future.copy()
    h_viu["viu"] = (h_viu["real_price"] / eff_mn_frac) + (c_pct * c_pen)
    f_viu["viu"] = (f_viu["real_price"] / eff_mn_frac) + (c_pct * c_pen)
    
    fig3 = go.Figure()
    
    # Historical Trace (Strict Isolation)
    hist_rp_viu = list(h_viu["real_price"]) + [None]*M
    hist_viu_viu = list(h_viu["viu"]) + [None]*M
    fig3.add_trace(go.Scatter(x=full_x, y=hist_rp_viu, name="Alloy Price (Raw)", line=dict(color=C_HYBRID, width=3), hovertemplate="<b>%{x|%B %Y}</b><br>%{x|%Y-%m-%d}<br>Raw Alloy Price: ₹%{y:.2f}<extra></extra>"))
    fig3.add_trace(go.Scatter(x=full_x, y=hist_viu_viu, name="True VIU Cost", line=dict(color=C_ACTUAL, width=2.5, dash="dash"), yaxis="y2", hovertemplate="<b>%{x|%B %Y}</b><br>%{x|%Y-%m-%d}<br>True VIU Cost: ₹%{y:.2f}<extra></extra>"))
    
    # Future Trace (Strict Isolation)
    fut_rp_viu = [None]*(N-1) + [h_viu["real_price"].iloc[-1]] + list(f_viu["real_price"])
    fut_viu_viu = [None]*(N-1) + [h_viu["viu"].iloc[-1]] + list(f_viu["viu"])
    fig3.add_trace(go.Scatter(x=full_x, y=fut_rp_viu, name="Future Alloy Price", line=dict(color=C_FUTURE, width=3), hovertemplate="<b>%{x|%B %Y}</b><br>%{x|%Y-%m-%d}<br>Future Alloy Price: ₹%{y:.2f}<extra></extra>"))
    fig3.add_trace(go.Scatter(x=full_x, y=fut_viu_viu, name="Future VIU Cost", line=dict(color="#FF5722", width=2.5, dash="dash"), yaxis="y2", hovertemplate="<b>%{x|%B %Y}</b><br>%{x|%Y-%m-%d}<br>Future VIU Cost: ₹%{y:.2f}<extra></extra>"))

    fig3.update_layout(**_layout(f"Raw Price vs. True Metallurgical Cost (Mn={mn_pct}%, C={c_pct}%)", "Raw Price (Rs/Kg)", 500))
    fig3.update_layout(yaxis2=dict(title="True Cost (Rs/Kg Mn)", overlaying="y", side="right", showgrid=False, matches="y"))
    st.plotly_chart(fig3, use_container_width=True)