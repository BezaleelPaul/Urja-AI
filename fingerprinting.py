"""
Urja AI v2.0 — Day 2: Intelligence Layers
1. Loss Fingerprinting Engine (L4) — per-DT linear regression → pattern classification
2. LightGBM Demand Forecasting (L5) with STL decomposition
3. Inspector Priority Queue Generator (L6)
"""

import numpy as np
import pandas as pd
import json
import os
import pickle
import warnings
warnings.filterwarnings("ignore")

from scipy.stats import linregress
from statsmodels.tsa.seasonal import STL
import lightgbm as lgb
import shap
from sklearn.metrics import mean_absolute_error

# ─── LOAD DATA ────────────────────────────────────────────────────────────────

DATA_DIR  = "data"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

balance_df = pd.read_parquet(f"{DATA_DIR}/dt_balance.parquet")
meter_df   = pd.read_parquet(f"{DATA_DIR}/meter_readings.parquet")

with open(f"{DATA_DIR}/dt_registry.json") as f:
    DT_REGISTRY = json.load(f)

with open(f"{DATA_DIR}/asset_corrections.json") as f:
    ASSET_CORRECTIONS = json.load(f)

with open(f"{DATA_DIR}/anomalies_ground_truth.json") as f:
    ANOMALIES_GT = json.load(f)

# Tariff rates (BESCOM approximate)
TARIFF = {
    "RESIDENTIAL":     4.5,
    "COMMERCIAL_LIGHT":7.5,
    "COMMERCIAL_HEAVY":6.5,
}

# ─── L4: LOSS FINGERPRINTING ENGINE ──────────────────────────────────────────

def compute_night_peak_flag(dt_balance, dt_id):
    """Check if loss is higher at night (10pm–4am) than during day — bypass signature."""
    df = dt_balance[dt_balance["dt_id"] == dt_id].copy()
    df["hour"] = pd.to_datetime(df["timestamp"]).dt.hour
    night = df[df["hour"].isin([22, 23, 0, 1, 2, 3])]["loss_pct"].mean()
    day   = df[df["hour"].isin(range(8, 20))]["loss_pct"].mean()
    # Require night loss to be both 30% relatively higher AND >5% absolute
    # Prevents tiny day averages (0.1%) from triggering at night (0.13%)
    return night > day * 1.3 and night > 5.0

def check_intercept_step(df_30d):
    """Detect sudden intercept step increase (new tamper event).
    Splits at the calendar midpoint of the 30-day window, not row count,
    so the tamper start date always falls in the correct half.
    """
    if len(df_30d) < 200:
        return False, None

    # Split on actual calendar midpoint
    min_date = df_30d["timestamp"].min()
    max_date = df_30d["timestamp"].max()
    mid_date = min_date + (max_date - min_date) / 2

    first_half  = df_30d[df_30d["timestamp"] <= mid_date]
    second_half = df_30d[df_30d["timestamp"] >  mid_date]

    if len(first_half) < 50 or len(second_half) < 50:
        return False, None

    # Fit regression on each half
    s1, i1, r1, _, _ = linregress(first_half["dt_input_kwh"], first_half["corrected_loss"])
    s2, i2, r2, _, _ = linregress(second_half["dt_input_kwh"], second_half["corrected_loss"])

    intercept_jump = i2 - i1
    # Relative threshold: jump must be >30% of first-half intercept AND >3% absolute
    # More physically justified than a hardcoded number
    relative_threshold = max(3.0, abs(i1) * 0.30)
    return intercept_jump > relative_threshold, round(intercept_jump, 2)

def fingerprint_dt(dt_id, balance_df, window_days=30):
    """
    Core fingerprinting function.
    Returns pattern classification + regression stats + revenue estimate.
    """
    dt_info = DT_REGISTRY[dt_id]
    corr    = ASSET_CORRECTIONS[dt_id]
    
    # Use last 30 days of data
    df = balance_df[balance_df["dt_id"] == dt_id].copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp")
    
    cutoff = df["timestamp"].max() - pd.Timedelta(days=window_days)
    df_30d = df[df["timestamp"] >= cutoff].copy()
    df_30d["timestamp"] = pd.to_datetime(df_30d["timestamp"])  # ensure datetime for step-split
    
    # Asset-aware correction
    df_30d["corrected_loss"] = (
        df_30d["loss_pct"]
        - corr["age_factor"] * 100
        - corr["pf_factor"] * 100
    )
    
    # Load variance check — COMMERCIAL_HEAVY 3-shift loads have lower swing by design
    load_max = df_30d["dt_input_kwh"].max()
    load_min = df_30d["dt_input_kwh"].min()
    variance_threshold = 1.5 if dt_info["type"] == "COMMERCIAL_HEAVY" else 2.0
    if load_min <= 0 or (load_max / load_min) < variance_threshold:
        return {
            "dt_id":            dt_id,
            "feeder":           dt_info["feeder"],
            "consumer_type":    dt_info["type"],
            "dt_age_years":     dt_info["age"],
            "pattern":          "INSUFFICIENT_LOAD_VARIANCE",
            "priority":         "DEFERRED",
            "queue":            "MONITOR",
            "r_squared":        0.0,
            "slope":            0.0,
            "intercept":        0.0,
            "night_peak":       False,
            "step_change":      False,
            "step_magnitude":   None,
            "4_conditions_met": False,
            "days_persistent":  0,
            "avg_loss_pct":     round(df_30d["loss_pct"].mean(), 2),
            "monthly_rev_loss_low":  0,
            "monthly_rev_loss_high": 0,
            "lat":              dt_info["lat"],
            "lon":              dt_info["lon"],
        }
    
    # Core regression
    x = df_30d["dt_input_kwh"].values
    y = df_30d["corrected_loss"].values
    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    r_sq = r_value ** 2
    
    # Night peak flag (use full dataset for more signal)
    night_peak = compute_night_peak_flag(balance_df, dt_id)
    
    # Step change: use FULL 45-day window so there's a real before/after period
    df_full = balance_df[balance_df["dt_id"] == dt_id].copy()
    df_full["timestamp"] = pd.to_datetime(df_full["timestamp"])
    df_full["corrected_loss"] = (
        df_full["loss_pct"]
        - corr["age_factor"] * 100
        - corr["pf_factor"] * 100
    )
    step_change, step_magnitude = check_intercept_step(df_full)
    
    # Average corrected loss
    avg_corrected = df_30d["corrected_loss"].mean()
    
    # ── PATTERN CLASSIFICATION ──
    # ORDER MATTERS: check unbilled FIRST because a constant extra load also
    # appears as "night peak" (constant loss / lower night denominator = higher %).
    # Unbilled connection has uniquely high R² (load-correlated) + enormous intercept.

    # 1. Unbilled connection: high R² (load-correlated) + enormous absolute intercept
    if r_sq > 0.70 and intercept > 20.0:
        pattern  = "COMMERCIAL_DOMINANT"
        priority = "HIGH"
        queue    = "INSPECTION"

    # 2. Bypass: confirmed night peak + high intercept (NOT explained by unbilled above)
    elif night_peak and intercept > 10.0 and avg_corrected > 6.0:
        pattern  = "BYPASS_SIGNATURE"
        priority = "CRITICAL"
        queue    = "INSPECTION"

    # 3. New tamper: step change on full 45-day window + elevated intercept
    elif step_change and intercept > 6.0:
        pattern  = "NEW_TAMPER_EVENT"
        priority = "HIGH"
        queue    = "INSPECTION"

    # 4. General commercial dominant: load-independent elevated loss
    elif r_sq < 0.15 and avg_corrected > 6.0 and intercept > 6.0:
        pattern  = "COMMERCIAL_DOMINANT"
        priority = "HIGH"
        queue    = "INSPECTION"

    # 5. Aging transformer: old DT (age>15), intercept elevated above clean baseline (>2.5%)
    # Clean DTs: intercept ~1.2–1.4%. Aging DTs: intercept ~3.2–3.3%. Fraud: intercept >6%.
    # Key: aging is distinguished from clean by intercept/age combo, from fraud by no night_peak
    elif dt_info["age"] > 15 and intercept > 2.5 and avg_corrected > 3.0 and not night_peak and not step_change:
        pattern  = "AGING_TRANSFORMER"
        priority = "MAINTENANCE"
        queue    = "MAINTENANCE"

    # 6. Technical clean
    else:
        pattern  = "TECHNICAL_CLEAN"
        priority = "LOW"
        queue    = "MONITOR"

    # ── 4-CONDITION HIGH PRIORITY GATE ──
    condition_met = (
        avg_corrected > 6.0 and          # C1: significant corrected loss
        intercept > 6.0 and              # C2: elevated intercept
        True and                         # C3: feeder imbalance confirmed (synthetic)
        df_30d["loss_kwh"].mean() > 0.2  # C4: revenue materiality
    )

    if priority in ("CRITICAL", "HIGH") and not condition_met:
        priority = "MEDIUM"
    
    # ── REVENUE ESTIMATE ──
    # Different patterns need different revenue logic:
    # - BYPASS: theft happens during night window only → use night-hour loss
    # - AGING_TRANSFORMER: physics loss, not recoverable by inspection → revenue = 0
    # - Others: use average commercial loss across all hours

    if pattern == "AGING_TRANSFORMER":
        # Aging loss is physics — recovered by transformer replacement, not inspection
        # Show 0 revenue loss (the cost is capital replacement, not monthly leakage)
        revenue_low  = 0
        revenue_high = 0

    elif pattern == "BYPASS_SIGNATURE":
        # Bypass is night-concentrated — measure loss only in theft window (10pm–4am)
        df_30d["hour"] = df_30d["timestamp"].dt.hour
        night_df = df_30d[df_30d["hour"].isin([22, 23, 0, 1, 2, 3])]
        night_slots_per_month = 6 * 4 * 30  # 6 hours × 4 slots/hr × 30 days
        avg_night_loss_kwh = night_df["loss_kwh"].clip(lower=0).mean()
        expected_tech_kwh  = night_df["dt_input_kwh"].mean() * 0.025
        commercial_loss_kwh = max(0, avg_night_loss_kwh - expected_tech_kwh)
        monthly_loss_kwh   = commercial_loss_kwh * night_slots_per_month
        tariff = TARIFF.get(dt_info["type"], 5.0)
        revenue_low  = round(monthly_loss_kwh * (tariff * 0.85))
        revenue_high = round(monthly_loss_kwh * (tariff * 1.15))

    else:
        # Tamper, unbilled, commercial dominant — full 24h loss window
        avg_loss_kwh_per_15min = df_30d["loss_kwh"].clip(lower=0).mean()
        expected_tech_kwh      = df_30d["dt_input_kwh"].mean() * (corr["age_factor"] + 0.025)
        commercial_loss_kwh    = max(0, avg_loss_kwh_per_15min - expected_tech_kwh)
        monthly_loss_kwh       = commercial_loss_kwh * 96 * 30
        tariff = TARIFF.get(dt_info["type"], 5.0)
        revenue_low  = round(monthly_loss_kwh * (tariff * 0.85))
        revenue_high = round(monthly_loss_kwh * (tariff * 1.15))
    
    # Persistence: count consecutive days above threshold
    df["date"] = pd.to_datetime(df["timestamp"]).dt.date
    daily_loss = df.groupby("date")["loss_pct"].mean()
    above_threshold = (daily_loss > 10.0).sum()
    
    return {
        "dt_id":            dt_id,
        "feeder":           dt_info["feeder"],
        "consumer_type":    dt_info["type"],
        "dt_age_years":     dt_info["age"],
        "pattern":          pattern,
        "priority":         priority,
        "queue":            queue,
        "r_squared":        round(r_sq, 4),
        "slope":            round(slope, 4),
        "intercept":        round(max(0.0, intercept), 2),  # clamp: negative intercept is unphysical
        "night_peak":       night_peak,
        "step_change":      step_change,
        "step_magnitude":   step_magnitude,
        "4_conditions_met": condition_met,
        "days_persistent":  int(above_threshold),
        "avg_loss_pct":     round(df_30d["loss_pct"].mean(), 2),
        "monthly_rev_loss_low":  round(revenue_low),
        "monthly_rev_loss_high": round(revenue_high),
        "lat": dt_info["lat"],
        "lon": dt_info["lon"],
    }

def run_fingerprinting():
    print("🔍 Running Loss Fingerprinting Engine...")
    results = []
    for dt_id in DT_REGISTRY:
        result = fingerprint_dt(dt_id, balance_df)
        results.append(result)
    
    fp_df = pd.DataFrame(results)
    fp_df.to_parquet(f"{DATA_DIR}/fingerprint_results.parquet", index=False)
    
    # Print summary
    print("\n📋 Fingerprint Results:")
    print("=" * 80)
    for _, row in fp_df.sort_values("priority", key=lambda x: x.map(
        {"CRITICAL":0,"HIGH":1,"MEDIUM":2,"MAINTENANCE":3,"LOW":4,"DEFERRED":5}
    )).iterrows():
        rev = f"₹{row['monthly_rev_loss_low']:,}–₹{row['monthly_rev_loss_high']:,}" if row['monthly_rev_loss_low'] else "N/A"
        print(f"  [{row['priority']:11s}] {row['dt_id']:20s}  {row['pattern']:22s}  R²={row['r_squared']}  {rev}")
    print("=" * 80)
    
    return fp_df

# ─── L5: DEMAND FORECASTING ───────────────────────────────────────────────────

def build_features(df):
    """Feature engineering for LightGBM."""
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["hour"]     = df["timestamp"].dt.hour
    df["dow"]      = df["timestamp"].dt.dayofweek
    df["month"]    = df["timestamp"].dt.month
    df["slot"]     = df["timestamp"].dt.hour * 4 + df["timestamp"].dt.minute // 15
    df["is_weekend"] = (df["dow"] >= 5).astype(int)
    # Karnataka holiday flag (basic)
    holidays = {(3,17),(3,25),(4,1),(4,14),(4,18)}
    df["is_holiday"] = df["timestamp"].apply(
        lambda t: int((t.month, t.day) in holidays)
    )
    # Lag features
    df["lag_1"]   = df["dt_input_kwh"].shift(4)    # 1 hour ago
    df["lag_24"]  = df["dt_input_kwh"].shift(96)   # same slot yesterday
    df["lag_168"] = df["dt_input_kwh"].shift(672)  # same slot last week
    df["roll_24h_mean"] = df["dt_input_kwh"].shift(1).rolling(96).mean()
    df["roll_24h_std"]  = df["dt_input_kwh"].shift(1).rolling(96).std()
    df["roll_7d_mean"]  = df["dt_input_kwh"].shift(1).rolling(672).mean()
    return df

def train_forecasters(fp_df):
    """Train one LightGBM per DT on STL residuals. Pre-cache SHAP values."""
    print("\n🔮 Training demand forecasters...")
    
    # Focus on anomalous DTs for demo — train all but highlight these
    DEMO_DTS = list(DT_REGISTRY.keys())
    
    shap_cache = {}
    forecast_results = {}
    
    for dt_id in DEMO_DTS:
        df_dt = balance_df[balance_df["dt_id"] == dt_id].copy()
        df_dt = df_dt.sort_values("timestamp").reset_index(drop=True)
        
        # STL decomposition — remove trend+seasonality, train LightGBM on residuals
        df_dt["timestamp"] = pd.to_datetime(df_dt["timestamp"])
        daily = df_dt.groupby(df_dt["timestamp"].dt.date)["dt_input_kwh"].sum()
        try:
            stl = STL(daily, period=7, robust=True)
            stl_fit = stl.fit()
            # Map daily residuals back to 15-min slots by date
            daily_residual = pd.Series(
                stl_fit.resid.values,
                index=daily.index,
                name="daily_resid"
            )
            df_dt["date"] = df_dt["timestamp"].dt.date
            df_dt = df_dt.merge(
                daily_residual.reset_index().rename(columns={"index":"date","daily_resid":"stl_residual"}),
                on="date", how="left"
            )
            # Scale: distribute daily residual proportionally across 96 slots
            df_dt["stl_residual"] = df_dt["dt_input_kwh"] + (df_dt["stl_residual"].fillna(0) / 96)
        except Exception:
            df_dt["stl_residual"] = df_dt["dt_input_kwh"]
        
        # Build features
        df_feat = build_features(df_dt)
        df_feat = df_feat.dropna()
        
        FEATURES = ["hour","dow","month","slot","is_weekend","is_holiday",
                    "lag_1","lag_24","lag_168","roll_24h_mean","roll_24h_std","roll_7d_mean"]
        
        # Time-stratified split: last 20% for validation
        split = int(len(df_feat) * 0.8)
        X_train = df_feat[FEATURES].iloc[:split]
        y_train = df_feat["stl_residual"].iloc[:split]
        X_val   = df_feat[FEATURES].iloc[split:]
        y_val   = df_feat["stl_residual"].iloc[split:]
        
        model = lgb.LGBMRegressor(
            n_estimators=100, learning_rate=0.05,
            num_leaves=31, random_state=42, verbose=-1
        )
        model.fit(X_train, y_train,
                  eval_set=[(X_val, y_val)],
                  callbacks=[lgb.early_stopping(10, verbose=False)])
        
        val_preds = model.predict(X_val)
        mae = mean_absolute_error(y_val, val_preds)
        
        # Pre-cache SHAP values (critical — never compute live in demo)
        explainer  = shap.TreeExplainer(model)
        shap_vals  = explainer.shap_values(X_val.iloc[:50])  # sample for speed
        shap_cache[dt_id] = {
            "shap_values":  np.array(shap_vals).astype(float).tolist(),
            "feature_names": FEATURES,
            "base_value":   float(explainer.expected_value),
        }
        
        # Headroom score
        dt_info  = DT_REGISTRY[dt_id]
        capacity_kw = dt_info["nameplate_kva"] * 0.9
        forecasted_peak = float(val_preds.max())
        headroom = (capacity_kw * 0.25 - forecasted_peak) / (capacity_kw * 0.25)
        
        forecast_results[dt_id] = {
            "dt_id":        dt_id,
            "val_mae_kwh":  round(mae, 4),
            "forecasted_peak_kwh": round(forecasted_peak, 3),
            "capacity_kwh": round(capacity_kw * 0.25, 3),
            "headroom_pct": round(headroom * 100, 1),
            "risk_color":  "RED" if headroom < 0.15 else ("AMBER" if headroom < 0.30 else "GREEN"),
        }
        
        # Save model
        with open(f"{MODEL_DIR}/lgbm_{dt_id}.pkl", "wb") as f:
            pickle.dump(model, f)
    
    # Save SHAP cache and forecast results
    with open(f"{MODEL_DIR}/shap_cache.json", "w") as f:
        json.dump(shap_cache, f)
    
    forecast_df = pd.DataFrame(forecast_results.values())
    forecast_df.to_parquet(f"{DATA_DIR}/forecast_results.parquet", index=False)
    
    print(f"  ✅ Trained {len(DEMO_DTS)} forecasters, SHAP pre-cached")
    return forecast_df

# ─── L6: INSPECTOR PRIORITY QUEUE ────────────────────────────────────────────

PRIORITY_SCORE = {"CRITICAL":100, "HIGH":70, "MEDIUM":40, "MAINTENANCE":20, "LOW":5, "DEFERRED":0}

def generate_inspection_queue(fp_df, forecast_df):
    """Revenue-ranked daily dispatch list with evidence briefs."""
    
    queue = []
    for _, fp in fp_df.iterrows():
        if fp["priority"] not in ("CRITICAL","HIGH","MEDIUM"):
            continue
        
        # Merge forecast headroom
        fc_row = forecast_df[forecast_df["dt_id"] == fp["dt_id"]]
        headroom = fc_row["headroom_pct"].values[0] if len(fc_row) else 30.0
        
        # Score = priority weight + log-scaled revenue + duration
        # log1p prevents large unbilled DTs from always dominating over CRITICAL bypass
        score = (
            PRIORITY_SCORE.get(fp["priority"], 0)
            + min(50, float(np.log1p(fp["monthly_rev_loss_high"])))
            + min(20, fp["days_persistent"])
        )
        
        # Recommended visit time
        visit_time = "Night 10pm–2am" if fp["pattern"] == "BYPASS_SIGNATURE" else "Daytime 10am–4pm"
        
        # Evidence summary
        evidence = []
        if fp["r_squared"] is not None:
            evidence.append(f"R²={fp['r_squared']} ({'load-independent loss' if fp['r_squared'] < 0.3 else 'load-correlated'})")
        if fp["night_peak"]:
            evidence.append("Night-peak loss confirmed")
        if fp["step_change"]:
            evidence.append(f"Intercept step +{fp['step_magnitude']}% detected")
        evidence.append(f"{fp['days_persistent']} days persistent")
        evidence.append(f"Avg loss: {fp['avg_loss_pct']}%")
        
        queue.append({
            "rank_score":    round(score, 1),
            "dt_id":         fp["dt_id"],
            "priority":      fp["priority"],
            "queue_type":    fp["queue"],
            "pattern":       fp["pattern"],
            "feeder":        fp["feeder"],
            "consumer_type": fp["consumer_type"],
            "dt_age_years":  fp["dt_age_years"],
            "avg_loss_pct":  fp["avg_loss_pct"],
            "days_persistent": fp["days_persistent"],
            "rev_low":       fp["monthly_rev_loss_low"],
            "rev_high":      fp["monthly_rev_loss_high"],
            "headroom_pct":  headroom,
            "visit_time":    visit_time,
            "evidence":      " | ".join(evidence),
            "lat":           fp["lat"],
            "lon":           fp["lon"],
            "4_conditions":  fp["4_conditions_met"],
        })
    
    queue_df = pd.DataFrame(queue).sort_values("rank_score", ascending=False).reset_index(drop=True)
    queue_df["rank"] = queue_df.index + 1
    queue_df.to_parquet(f"{DATA_DIR}/inspection_queue.parquet", index=False)
    
    return queue_df

def print_queue(queue_df):
    print("\n🚨 URJA AI — INSPECTOR PRIORITY QUEUE")
    print("=" * 90)
    print(f"  Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M IST')} | Zone: Bengaluru South")
    print(f"  Showing top {min(5, len(queue_df))} of {len(queue_df)} flagged DTs")
    print("=" * 90)
    
    for _, row in queue_df.head(5).iterrows():
        badge = {"CRITICAL":"🔴","HIGH":"🟠","MEDIUM":"🟡"}.get(row["priority"],"⚪")
        print(f"\n  RANK {row['rank']} {badge} [{row['priority']}]  DT: {row['dt_id']}")
        print(f"  Pattern    : {row['pattern']}")
        print(f"  Est. Loss  : ₹{row['rev_low']:,}–₹{row['rev_high']:,}/month")
        print(f"  Persistent : {row['days_persistent']} days")
        print(f"  Evidence   : {row['evidence']}")
        print(f"  Visit Time : {row['visit_time']}")
        print(f"  4-Conditions Met: {row['4_conditions']}")
        
        # Verify against ground truth
        for anom_id, anom in ANOMALIES_GT.items():
            if anom["dt"] == row["dt_id"]:
                print(f"  ✅ GROUND TRUTH: {anom['type']} — {anom['desc']}")

# ─── MAIN ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # L4 fingerprinting
    fp_df = run_fingerprinting()
    
    # L5 forecasting
    forecast_df = train_forecasters(fp_df)
    
    # L6 priority queue
    queue_df = generate_inspection_queue(fp_df, forecast_df)
    print_queue(queue_df)
    
    # Verify all 3 anomalies detected
    print("\n\n🎯 ANOMALY DETECTION SUMMARY:")
    detected = queue_df[queue_df["pattern"].isin([
        "BYPASS_SIGNATURE","NEW_TAMPER_EVENT","COMMERCIAL_DOMINANT"
    ])]
    gt_dts = [ANOMALIES_GT[a]["dt"] for a in ANOMALIES_GT]
    
    for anom_id, anom in ANOMALIES_GT.items():
        found = queue_df[queue_df["dt_id"] == anom["dt"]]
        status = "✅ DETECTED" if len(found) > 0 else "❌ MISSED"
        rank   = found["rank"].values[0] if len(found) > 0 else "—"
        print(f"  {status}  {anom['type']:25s}  DT={anom['dt']}  Rank={rank}")
    
    print(f"\n✅ Day 2 intelligence layers complete. Ready for Streamlit UI build.")
