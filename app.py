"""
Urja AI v2.0 — Streamlit Demo App
Zone Risk Map + Inspector Priority Queue + DT Drill-down
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import pickle
import plotly.graph_objects as go
import plotly.express as px
from scipy.stats import linregress
import warnings
warnings.filterwarnings("ignore")

# ─── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Urja AI v2.0 — BESCOM Smart Grid Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CUSTOM CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .metric-card {
        background: #1e2530;
        border-radius: 10px;
        padding: 16px 20px;
        margin: 4px 0;
        border-left: 4px solid;
    }
    .critical { border-color: #ff4b4b; }
    .high     { border-color: #ffa500; }
    .medium   { border-color: #ffdd00; }
    .clean    { border-color: #21c55d; }
    .stDataFrame { font-size: 13px; }
    div[data-testid="stMetricValue"] { font-size: 2rem; font-weight: 700; }
</style>
""", unsafe_allow_html=True)

# ─── DATA LOADING ─────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    balance_df  = pd.read_parquet("data/dt_balance.parquet")
    meter_df    = pd.read_parquet("data/meter_readings.parquet")
    fp_df       = pd.read_parquet("data/fingerprint_results.parquet")
    queue_df    = pd.read_parquet("data/inspection_queue.parquet")
    fc_df       = pd.read_parquet("data/forecast_results.parquet")
    
    with open("data/dt_registry.json") as f:
        dt_reg = json.load(f)
    with open("models/shap_cache.json") as f:
        shap_cache = json.load(f)
    with open("data/anomalies_ground_truth.json") as f:
        gt = json.load(f)
    
    balance_df["timestamp"] = pd.to_datetime(balance_df["timestamp"])
    fp_df["monthly_rev_loss_low"]  = fp_df["monthly_rev_loss_low"].fillna(0)
    fp_df["monthly_rev_loss_high"] = fp_df["monthly_rev_loss_high"].fillna(0)
    fp_df["avg_loss_pct"]          = fp_df["avg_loss_pct"].fillna(0)
    fp_df["r_squared"]             = fp_df["r_squared"].fillna(0)
    fp_df["intercept"]             = fp_df["intercept"].fillna(0)
    
    return balance_df, meter_df, fp_df, queue_df, fc_df, dt_reg, shap_cache, gt

balance_df, meter_df, fp_df, queue_df, fc_df, DT_REG, shap_cache, GT = load_data()

# Color maps
PRIORITY_COLOR = {
    "CRITICAL":   "#ff4b4b",
    "HIGH":       "#ff8c00",
    "MEDIUM":     "#ffd700",
    "MAINTENANCE":"#00bfff",
    "LOW":        "#21c55d",
    "DEFERRED":   "#888888",
}
PRIORITY_LABEL = {
    "CRITICAL":   "CRITICAL — Bypass/Urgent",
    "HIGH":       "HIGH — Tamper/Unbilled",
    "MEDIUM":     "MEDIUM — Investigate",
    "MAINTENANCE":"MAINTENANCE — Aging Transformer",
    "LOW":        "LOW — Technical Clean",
    "DEFERRED":   "DEFERRED — Insufficient Load Variance (industrial/heavy load DT, fingerprint needs wider load swing)",
}
PATTERN_ICON = {
    "BYPASS_SIGNATURE":    "🔴",
    "NEW_TAMPER_EVENT":    "🟠",
    "COMMERCIAL_DOMINANT": "🟡",
    "AGING_TRANSFORMER":   "🔵",
    "TECHNICAL_CLEAN":     "🟢",
}

# ─── SIDEBAR ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/electricity.png", width=60)
    st.title("Urja AI v2.0")
    st.caption("Network-Topology-Aware Energy Intelligence")
    st.divider()
    
    page = st.radio("Navigate", [
        "📍 Zone Risk Map",
        "🚨 Inspector Queue",
        "🔬 DT Drill-Down",
        "📊 Loss Scatter",
    ])
    
    st.divider()
    st.markdown("**System Status**")
    n_critical    = len(fp_df[fp_df["priority"] == "CRITICAL"])
    n_high        = len(fp_df[fp_df["priority"] == "HIGH"])
    n_maintenance = len(fp_df[fp_df["priority"] == "MAINTENANCE"])
    st.success(f"✅ {len(fp_df)} DTs monitored")
    if n_critical > 0:
        st.error(f"🔴 {n_critical} CRITICAL alert{'s' if n_critical > 1 else ''}")
    if n_high > 0:
        st.warning(f"🟠 {n_high} HIGH alert{'s' if n_high > 1 else ''}")
    if n_maintenance > 0:
        st.info(f"🔵 {n_maintenance} MAINTENANCE alert{'s' if n_maintenance > 1 else ''}")
    st.info("🔵 SHAP cache: loaded")
    
    st.divider()
    st.caption("PAN IIT · AI for Bharat 2026 · Theme 8 · BESCOM")

# ─── PAGE 1: ZONE RISK MAP ────────────────────────────────────────────────────
if page == "📍 Zone Risk Map":
    st.title("⚡ Urja AI — Zone Risk Map")
    st.caption("Bengaluru South Distribution Zone · 15 DTs · 45-day analysis window")
    
    # KPI row — all values computed live from fingerprinting results
    n_critical    = len(fp_df[fp_df["priority"] == "CRITICAL"])
    n_high        = len(fp_df[fp_df["priority"] == "HIGH"])
    n_maintenance = len(fp_df[fp_df["priority"] == "MAINTENANCE"])
    total_rev_low  = int(queue_df["rev_low"].sum())
    total_rev_high = int(queue_df["rev_high"].sum())

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total DTs", len(fp_df))
    c2.metric("🔴 CRITICAL", n_critical, delta="Bypass detected" if n_critical > 0 else "None")
    c3.metric("🟠 HIGH", n_high, delta=f"₹{total_rev_low/1000:.1f}K–₹{total_rev_high/1000:.1f}K/mo")
    c4.metric("Est. Monthly Loss", f"₹{total_rev_low/1000:.1f}K–₹{total_rev_high/1000:.1f}K")
    c5.metric("False Positives", "Low", delta="4-condition gate")
    
    st.divider()
    
    # Build map dataframe
    map_data = []
    for _, row in fp_df.iterrows():
        dt_info = DT_REG.get(row["dt_id"], {})
        map_data.append({
            "dt_id":    row["dt_id"],
            "lat":      row["lat"],
            "lon":      row["lon"],
            "priority": row["priority"],
            "pattern":  row["pattern"],
            "avg_loss": row["avg_loss_pct"],
            "rev_est":  f"₹{int(row['monthly_rev_loss_low']):,}–₹{int(row['monthly_rev_loss_high']):,}",
            "color":    PRIORITY_COLOR.get(row["priority"], "#888"),
            "size":     {"CRITICAL":20,"HIGH":16,"MEDIUM":12,"LOW":8,"MAINTENANCE":10,"DEFERRED":6}.get(row["priority"],8),
        })
    map_df = pd.DataFrame(map_data)
    
    fig_map = go.Figure()
    
    # Add DT markers
    for priority, color in PRIORITY_COLOR.items():
        subset = map_df[map_df["priority"] == priority]
        if len(subset) == 0:
            continue
        fig_map.add_trace(go.Scattermap(
            lat=subset["lat"],
            lon=subset["lon"],
            mode="markers+text",
            marker=dict(size=subset["size"], color=color, opacity=0.9),
            text=subset["dt_id"].str.replace("-DT-","<br>DT-"),
            textposition="top center",
            textfont=dict(size=9, color="white"),
            customdata=np.stack([
                subset["dt_id"],
                subset["pattern"],
                subset["avg_loss"].astype(str),
                subset["rev_est"],
                subset["priority"],
                subset["priority"].map(PRIORITY_LABEL).fillna(""),
            ], axis=-1),
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "Priority: <b>%{customdata[4]}</b><br>"
                "Pattern: %{customdata[1]}<br>"
                "Avg Loss: %{customdata[2]}%<br>"
                "Est. Loss: %{customdata[3]}/mo<br>"
                "<i style='color:#aaa'>%{customdata[5]}</i><br>"
                "<extra></extra>"
            ),
            name=priority,
        ))
    
    fig_map.update_layout(
        map=dict(
            style="carto-darkmatter",
            center=dict(lat=12.928, lon=77.610),
            zoom=13,
        ),
        height=520,
        margin=dict(l=0, r=0, t=0, b=0),
        legend=dict(
            bgcolor="rgba(30,37,48,0.9)",
            bordercolor="gray",
            borderwidth=1,
            font=dict(color="white"),
        ),
        paper_bgcolor="#0e1117",
        plot_bgcolor="#0e1117",
    )
    
    st.plotly_chart(fig_map, use_container_width=True)
    
    # Legend
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.markdown("🔴 **CRITICAL** — Bypass/Urgent")
    col2.markdown("🟠 **HIGH** — Tamper/Unbilled")
    col3.markdown("🟡 **MEDIUM** — Investigate")
    col4.markdown("🟢 **LOW** — Technical Clean")
    col5.markdown("⚫ **DEFERRED** — Heavy load DT, needs wider load swing for fingerprint")
    
    # Summary table
    st.divider()
    st.subheader("DT Status Summary")
    display_fp = fp_df[["dt_id","feeder","consumer_type","pattern","priority",
                         "avg_loss_pct","days_persistent",
                         "monthly_rev_loss_low","monthly_rev_loss_high"]].copy()
    display_fp.columns = ["DT ID","Feeder","Type","Pattern","Priority",
                           "Avg Loss%","Days","Rev Loss Low","Rev Loss High"]
    display_fp = display_fp.sort_values("Priority", key=lambda x: x.map(
        {"CRITICAL":0,"HIGH":1,"MEDIUM":2,"MAINTENANCE":3,"LOW":4,"DEFERRED":5}
    ).fillna(99))
    
    def color_priority(val):
        c = {"CRITICAL":   "background-color:#ff4b4b;color:black",
             "HIGH":       "background-color:#ff8c00;color:black",
             "MEDIUM":     "background-color:#ffd700;color:black",
             "MAINTENANCE":"background-color:#00bfff;color:black",
             "LOW":        "background-color:#21c55d;color:black"}.get(val,"")
        return c
    
    st.dataframe(
        display_fp.style.map(color_priority, subset=["Priority"]),
        use_container_width=True, height=400
    )

# ─── PAGE 2: INSPECTOR QUEUE ──────────────────────────────────────────────────
elif page == "🚨 Inspector Queue":
    st.title("🚨 Inspector Priority Queue")
    st.caption(f"Revenue-ranked dispatch · Generated {pd.Timestamp.now().strftime('%d %b %Y, %H:%M IST')}")
    
    c1, c2, c3 = st.columns(3)
    insp_only  = queue_df[queue_df["queue_type"] == "INSPECTION"]
    total_low  = insp_only["rev_low"].sum()
    total_high = insp_only["rev_high"].sum()
    n_aging    = len(fp_df[fp_df["pattern"] == "AGING_TRANSFORMER"])
    c1.metric("Inspection Queue", len(insp_only), delta=f"+{n_aging} maintenance")
    c2.metric("Est. Recoverable Revenue", f"₹{int(total_low/1000)}K–₹{int(total_high/1000)}K/mo")
    c3.metric("Available Inspectors", "3")
    
    st.divider()
    
    # Separate queues
    col_insp, col_maint = st.columns([2,1])
    
    with col_insp:
        st.subheader("🔍 Inspection Queue (Theft/Bypass)")
        insp_q = queue_df[queue_df["queue_type"] == "INSPECTION"]
        
        for _, row in insp_q.iterrows():
            badge = {"CRITICAL":"🔴","HIGH":"🟠","MEDIUM":"🟡"}.get(row["priority"],"⚪")
            color = PRIORITY_COLOR.get(row["priority"],"#888")
            
            with st.container():
                st.markdown(f"""
<div class="metric-card {'critical' if row['priority']=='CRITICAL' else 'high' if row['priority']=='HIGH' else 'medium'}">
  <h4 style="margin:0;color:{color}">{badge} RANK {row['rank']} — {row['dt_id']} [{row['priority']}]</h4>
  <p style="margin:4px 0;font-size:0.85rem;color:#aaa">Pattern: <b style="color:white">{row['pattern']}</b> &nbsp;|&nbsp; Type: {row['consumer_type']}</p>
  <p style="margin:4px 0;font-size:1.0rem"><b style="color:{color}">₹{int(row['rev_low']):,}–₹{int(row['rev_high']):,}/month</b> estimated recoverable loss</p>
  <p style="margin:4px 0;font-size:0.8rem;color:#ccc">📋 {str(row['evidence'])[:130]}{'...' if len(str(row['evidence'])) > 130 else ''}</p>
  <p style="margin:4px 0;font-size:0.8rem">🕐 Visit: <b>{row['visit_time']}</b> &nbsp;|&nbsp; Persistent: <b>{int(row['days_persistent'])} days</b></p>
  <p style="margin:4px 0;font-size:0.75rem;color:#21c55d">✅ 4-Conditions Met: {row['4_conditions']}</p>
</div>
""", unsafe_allow_html=True)
                st.markdown("")
    
    with col_maint:
        st.subheader("🔧 Maintenance Queue")
        aging = fp_df[fp_df["pattern"] == "AGING_TRANSFORMER"].copy()
        if len(aging) > 0:
            for _, row in aging.iterrows():
                st.info(
                    f"🔵 **{row['dt_id']}**  \n"
                    f"Age: {row['dt_age_years']} yrs | Avg Loss: {row['avg_loss_pct']}%  \n"
                    f"Intercept: {row['intercept']}% (clean DTs ~1.2%)  \n"
                    f"**Action: Schedule transformer replacement**  \n"
                    f"*(Physics loss — not recoverable by inspection)*"
                )
        else:
            st.success("No aging transformer alerts.")
        
        st.divider()
        st.subheader("✅ Demo Validation (Synthetic Data)")
        st.markdown("*Verifying system detected all 3 planted anomalies*")
        gt_dts = {"BTMSTG-DT-14":"BYPASS", "JAYNGR-DT-07":"TAMPER", "KORAMGLA-DT-22":"UNBILLED"}
        for dt, atype in gt_dts.items():
            found = queue_df[queue_df["dt_id"] == dt]
            if len(found) > 0:
                st.success(f"✅ {dt}\n{atype} → Rank {int(found['rank'].values[0])}")
            else:
                st.error(f"❌ {dt} not detected")

# ─── PAGE 3: DT DRILL-DOWN ────────────────────────────────────────────────────
elif page == "🔬 DT Drill-Down":
    st.title("🔬 DT Drill-Down Analysis")
    
    dt_selected = st.selectbox(
        "Select Distribution Transformer",
        options=list(fp_df.sort_values("priority", key=lambda x: x.map(
            {"CRITICAL":0,"HIGH":1,"MEDIUM":2,"LOW":3,"DEFERRED":4,"MAINTENANCE":3}
        ))["dt_id"]),
    )
    
    fp_rows = fp_df[fp_df["dt_id"] == dt_selected]
    if len(fp_rows) == 0:
        st.error(f"No fingerprint data available for {dt_selected}.")
        st.stop()
    fp_row  = fp_rows.iloc[0]
    dt_info = DT_REG.get(dt_selected, {})
    
    # Header
    color  = PRIORITY_COLOR.get(fp_row["priority"], "#888")
    icon   = PATTERN_ICON.get(fp_row["pattern"], "⚪")
    st.markdown(f"### {icon} {dt_selected} — <span style='color:{color}'>{fp_row['priority']}</span>", unsafe_allow_html=True)
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Pattern", fp_row["pattern"].replace("_"," "))
    c2.metric("Avg Loss", f"{fp_row['avg_loss_pct']}%")
    c3.metric("R²", str(fp_row["r_squared"]))
    c4.metric("Est. Monthly Loss", f"₹{int(fp_row['monthly_rev_loss_low'] or 0):,}–₹{int(fp_row['monthly_rev_loss_high'] or 0):,}")
    
    st.divider()
    
    col_scatter, col_ts = st.columns(2)
    
    # ── Loss-Load Scatter (the core fingerprint) ──
    with col_scatter:
        st.subheader("📉 Loss-Load Regression")
        st.caption("The fingerprint: does loss scale with load (physics) or stay flat/night-heavy (fraud)?")
        
        df_dt = balance_df[balance_df["dt_id"] == dt_selected].copy()
        df_dt = df_dt.tail(30 * 96)  # last 30 days
        
        # Sample for plotting speed
        df_plot = df_dt.sample(min(500, len(df_dt)), random_state=42)
        df_plot["hour"] = df_plot["timestamp"].dt.hour
        df_plot["is_night"] = df_plot["hour"].isin([22, 23, 0, 1, 2, 3])
        
        # Regression line
        x_arr = df_dt["dt_input_kwh"].values
        y_arr = df_dt["loss_pct"].values
        if len(x_arr) > 10:
            slope, intercept, r_val, _, _ = linregress(x_arr, y_arr)
            x_line = np.linspace(x_arr.min(), x_arr.max(), 100)
            y_line = slope * x_line + intercept
        
        fig_scatter = go.Figure()
        # Day points
        day_pts = df_plot[~df_plot["is_night"]]
        fig_scatter.add_trace(go.Scatter(
            x=day_pts["dt_input_kwh"], y=day_pts["loss_pct"],
            mode="markers", marker=dict(color="#4a9eff", size=4, opacity=0.5),
            name="Daytime readings"
        ))
        # Night points (highlighted for bypass detection)
        night_pts = df_plot[df_plot["is_night"]]
        fig_scatter.add_trace(go.Scatter(
            x=night_pts["dt_input_kwh"], y=night_pts["loss_pct"],
            mode="markers", marker=dict(color="#ff6b6b", size=6, opacity=0.7),
            name="Night readings (10pm–4am)"
        ))
        # Regression line
        fig_scatter.add_trace(go.Scatter(
            x=x_line, y=y_line,
            mode="lines", line=dict(color="#ffd700", width=2, dash="dash"),
            name=f"Fit line (R²={fp_row['r_squared']}, intercept={fp_row['intercept']}%)"
        ))
        
        fig_scatter.update_layout(
            title=f"Loss% vs DT Input Load | {dt_selected}",
            xaxis_title="DT Input (kWh per 15-min)",
            yaxis_title="Loss %",
            paper_bgcolor="#1e2530",
            plot_bgcolor="#1e2530",
            font_color="white",
            legend=dict(bgcolor="rgba(0,0,0,0)", font_color="white"),
            height=380,
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
        
        # Explanation — truth source is the fingerprinting engine, not raw R²/intercept
        pattern_fp   = fp_row["pattern"]
        priority_fp  = fp_row["priority"]
        r2           = fp_row["r_squared"]
        intercept_fp = fp_row["intercept"]

        if pattern_fp == "BYPASS_SIGNATURE":
            st.error("🔴 **CRITICAL — Bypass signature detected.** Night-time loss spike is load-independent. Physical wiring bypass suspected.")
        elif pattern_fp == "NEW_TAMPER_EVENT":
            step_mag = fp_row["step_magnitude"]
            step_str = f"+{step_mag}%" if step_mag else "detected"
            st.warning(f"🟠 **HIGH — Sudden intercept step ({step_str}) indicates meter tampering.** Loss elevated from a specific date forward.")
        elif pattern_fp == "COMMERCIAL_DOMINANT":
            st.warning(f"🟠 **HIGH — Load-independent loss (R²={r2}).** Constant loss regardless of feeder load — unbilled connection or bypass suspected.")
        elif pattern_fp == "AGING_TRANSFORMER":
            st.info(f"🔵 **MAINTENANCE — Elevated losses due to aging asset ({fp_row['dt_age_years']} yrs).** Loss scales with load (physics). No fraud signal. Schedule replacement.")
        elif pattern_fp == "INSUFFICIENT_LOAD_VARIANCE":
            st.info("⚪ **DEFERRED — Insufficient load variation** in 30-day window to form a reliable regression fingerprint.")
        else:
            st.success(f"🟢 **CLEAN — Loss correlates with load (R²={r2}).** Technical I²R losses only. No commercial loss signal.")

        # Evidence caption — always shown regardless of pattern
        st.caption(
            f"R²={r2} | Intercept={intercept_fp}% | "
            f"Avg Loss={fp_row['avg_loss_pct']}% | "
            f"Days Persistent={fp_row['days_persistent']}"
        )
        if fp_row["night_peak"] and pattern_fp == "BYPASS_SIGNATURE":
            st.caption("🌙 Night-peak loss confirmed — loss is highest when feeder load is lowest (10pm–4am)")
        if fp_row["step_change"]:
            st.caption(f"📈 Step change in intercept: +{fp_row['step_magnitude']}% — loss escalated from a specific date")
    
    # ── Time-Series Loss ──
    with col_ts:
        st.subheader("📈 Loss % Over Time")
        st.caption("30-day rolling view — look for step changes or night-pattern anomalies")
        
        df_dt_daily = df_dt.copy()
        df_dt_daily["date"] = df_dt_daily["timestamp"].dt.date
        daily_loss = df_dt_daily.groupby("date")["loss_pct"].mean().reset_index()
        
        fig_ts = go.Figure()
        fig_ts.add_trace(go.Scatter(
            x=daily_loss["date"], y=daily_loss["loss_pct"],
            mode="lines+markers", line=dict(color="#4a9eff", width=2),
            marker=dict(size=4),
            fill="tozeroy", fillcolor="rgba(74,158,255,0.15)",
            name="Daily avg loss%"
        ))
        
        # Anomaly start markers
        data_start = balance_df["timestamp"].min()
        for anom_id, anom in GT.items():
            if anom["dt"] == dt_selected and "start_day" in anom:
                start_ts   = data_start + pd.Timedelta(days=anom["start_day"])
                start_date = start_ts.date()
                fig_ts.add_shape(
                    type="line",
                    x0=start_date, x1=start_date,
                    y0=0, y1=1,
                    xref="x", yref="paper",
                    line=dict(color="#ff4b4b", width=2, dash="dash"),
                )
                fig_ts.add_annotation(
                    x=start_date, y=1,
                    xref="x", yref="paper",
                    text=f"⚠️ {anom['type'].replace('_',' ')} starts",
                    showarrow=False,
                    font=dict(color="#ff4b4b", size=11),
                    xanchor="left", yanchor="top",
                )
        
        # Technical loss expected band
        expected = 3.5
        fig_ts.add_shape(
            type="line",
            x0=0, x1=1, y0=expected, y1=expected,
            xref="paper", yref="y",
            line=dict(color="#21c55d", width=1, dash="dot"),
        )
        fig_ts.add_annotation(
            x=1, y=expected,
            xref="paper", yref="y",
            text="Expected technical loss (~3.5%)",
            showarrow=False,
            font=dict(color="#21c55d", size=10),
            xanchor="right",
        )
        
        fig_ts.update_layout(
            paper_bgcolor="#1e2530", plot_bgcolor="#1e2530",
            font_color="white", height=380,
            xaxis_title="Date", yaxis_title="Loss %",
            legend=dict(bgcolor="rgba(0,0,0,0)"),
        )
        st.plotly_chart(fig_ts, use_container_width=True)
    
    # ── Asset Info + Evidence Brief ──
    st.divider()
    col_asset, col_brief = st.columns(2)
    
    with col_asset:
        st.subheader("🏗️ Asset Information")
        st.json({
            "DT ID":          dt_selected,
            "Feeder":         dt_info["feeder"],
            "Consumer Type":  dt_info["type"],
            "Age (years)":    dt_info["age"],
            "Nameplate (kVA)":dt_info["nameplate_kva"],
            "Registered Meters": dt_info["meters"],
        })
    
    with col_brief:
        st.subheader("📋 Evidence Brief")
        ev_row = queue_df[queue_df["dt_id"] == dt_selected]
        if len(ev_row) > 0:
            ev = ev_row.iloc[0]
            st.markdown(f"""
**Pattern:** {fp_row['pattern']}  
**Priority:** {fp_row['priority']}  
**Evidence:** {ev['evidence']}  
**Days Persistent:** {int(ev['days_persistent'])}  
**Recommended Visit:** {ev['visit_time']}  
**4-Conditions Gate:** {'✅ Passed' if ev['4_conditions'] else '❌ Not met'}  
**Est. Revenue Loss:** ₹{int(ev['rev_low']):,}–₹{int(ev['rev_high']):,}/month
""")
        else:
            st.info("This DT is in MONITOR state — no inspection required.")

# ─── PAGE 4: LOSS SCATTER OVERVIEW ───────────────────────────────────────────
elif page == "📊 Loss Scatter":
    st.title("📊 Loss Pattern Overview — All DTs")
    st.caption("Each point = one DT's 30-day regression. R² vs Intercept reveals the pattern class.")
    
    fp_valid = fp_df[fp_df["r_squared"].notna()].copy()
    
    fp_valid["marker_size"] = fp_valid["avg_loss_pct"].clip(lower=4)
    fig = px.scatter(
        fp_valid,
        x="r_squared",
        y="intercept",
        color="priority",
        size="marker_size",
        text="dt_id",
        color_discrete_map=PRIORITY_COLOR,
        hover_data=["pattern","consumer_type","monthly_rev_loss_high"],
        labels={"r_squared":"R² (loss-load correlation)","intercept":"Intercept (base loss %)"},
        title="R² vs Intercept — DT Loss Fingerprint Map",
    )
    
    # Quadrant lines
    fig.add_shape(type="line", x0=0.30, x1=0.30, y0=0, y1=1,
                  xref="x", yref="paper", line=dict(color="#888", width=1, dash="dash"))
    fig.add_annotation(x=0.30, y=1, xref="x", yref="paper",
                       text="R²=0.30 threshold", showarrow=False,
                       font=dict(color="#aaa", size=10), xanchor="left", yanchor="top")
    fig.add_shape(type="line", x0=0, x1=1, y0=8.0, y1=8.0,
                  xref="paper", yref="y", line=dict(color="#888", width=1, dash="dash"))
    fig.add_annotation(x=1, y=8.0, xref="paper", yref="y",
                       text="Intercept=8% threshold", showarrow=False,
                       font=dict(color="#aaa", size=10), xanchor="right")
    
    # Quadrant labels
    fig.add_annotation(x=0.05,  y=40, text="🔴 FRAUD ZONE\n(Low R², High intercept)", font_color="#ff4b4b", showarrow=False)
    fig.add_annotation(x=0.85,  y=1,  text="🟢 TECHNICAL\n(High R², Low intercept)", font_color="#21c55d", showarrow=False)
    
    fig.update_traces(textposition="top center", textfont_size=9)
    fig.update_layout(
        paper_bgcolor="#1e2530", plot_bgcolor="#1e2530",
        font_color="white", height=500,
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.info("""
**How to read this chart:**
- **X-axis (R²):** How much loss correlates with load. High R² = physics. Low R² = suspicious.
- **Y-axis (Intercept):** Base loss regardless of load level. High intercept = unexplained constant loss.
- **Bottom-right (green zone):** Technical losses — normal I²R physics. No action needed.
- **Top-left (red zone):** Commercial losses — load-independent. Theft, bypass, or unbilled connections.
    """)

# ─── FOOTER ───────────────────────────────────────────────────────────────────
st.sidebar.divider()
st.sidebar.markdown("""
**Urja AI v2.0**  
Energy Forensics Platform  
*For BESCOM · Bengaluru*  

**Lead Full Stack Developer:** Yeshwanth Reddy P G  
**Researcher:** Aditya S  
**Debugger:** Bezaleel Paul N  
CMR University · 2026
""")
st.sidebar.caption("⚠️ Demo uses simulated BESCOM-representative data for illustration purposes.")
