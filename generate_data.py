"""
Urja AI v2.0 — Day 1: Synthetic Network Data Generator
Generates 45-day × 15-min smart meter data for 3 feeders, 15 DTs, 60 meters
Plants 3 anomalies at known locations for demo
"""

import numpy as np
import pandas as pd
import json
import os
from datetime import datetime, timedelta

np.random.seed(42)

# ─── NETWORK TOPOLOGY ────────────────────────────────────────────────────────

FEEDERS = {
    "F_JAYANAGAR": {"label": "Jayanagar",   "lat": 12.9253, "lon": 77.5936},
    "F_BTM":       {"label": "BTM Stage 2", "lat": 12.9165, "lon": 77.6101},
    "F_KORAMANGALA":{"label": "Koramangala","lat": 12.9352, "lon": 77.6245},
}

# 5 DTs per feeder → 15 total
DT_REGISTRY = {
    # Jayanagar feeder — mostly residential
    "JAYNGR-DT-01": {"feeder":"F_JAYANAGAR","type":"RESIDENTIAL",        "age":6,  "lat":12.9271,"lon":77.5912,"nameplate_kva":100,"meters":4},
    "JAYNGR-DT-02": {"feeder":"F_JAYANAGAR","type":"RESIDENTIAL",        "age":10, "lat":12.9248,"lon":77.5955,"nameplate_kva":100,"meters":4},
    "JAYNGR-DT-03": {"feeder":"F_JAYANAGAR","type":"COMMERCIAL_LIGHT",   "age":14, "lat":12.9260,"lon":77.5940,"nameplate_kva":160,"meters":4},
    "JAYNGR-DT-04": {"feeder":"F_JAYANAGAR","type":"RESIDENTIAL",        "age":20, "lat":12.9235,"lon":77.5960,"nameplate_kva":100,"meters":4},
    "JAYNGR-DT-07": {"feeder":"F_JAYANAGAR","type":"RESIDENTIAL",        "age":12, "lat":12.9255,"lon":77.5925,"nameplate_kva":100,"meters":4},
    # BTM feeder — mixed/commercial
    "BTMSTG-DT-08": {"feeder":"F_BTM","type":"COMMERCIAL_LIGHT",         "age":8,  "lat":12.9172,"lon":77.6088,"nameplate_kva":160,"meters":4},
    "BTMSTG-DT-09": {"feeder":"F_BTM","type":"RESIDENTIAL",              "age":5,  "lat":12.9180,"lon":77.6115,"nameplate_kva":100,"meters":4},
    "BTMSTG-DT-10": {"feeder":"F_BTM","type":"COMMERCIAL_HEAVY",         "age":18, "lat":12.9155,"lon":77.6095,"nameplate_kva":250,"meters":4},
    "BTMSTG-DT-11": {"feeder":"F_BTM","type":"RESIDENTIAL",              "age":3,  "lat":12.9168,"lon":77.6120,"nameplate_kva":100,"meters":4},
    "BTMSTG-DT-14": {"feeder":"F_BTM","type":"COMMERCIAL_LIGHT",         "age":8,  "lat":12.9175,"lon":77.6105,"nameplate_kva":160,"meters":4},
    # Koramangala feeder — mixed
    "KORAMGLA-DT-19":{"feeder":"F_KORAMANGALA","type":"RESIDENTIAL",     "age":7,  "lat":12.9365,"lon":77.6230,"nameplate_kva":100,"meters":4},
    "KORAMGLA-DT-20":{"feeder":"F_KORAMANGALA","type":"COMMERCIAL_LIGHT","age":11, "lat":12.9340,"lon":77.6250,"nameplate_kva":160,"meters":4},
    "KORAMGLA-DT-21":{"feeder":"F_KORAMANGALA","type":"RESIDENTIAL",     "age":22, "lat":12.9355,"lon":77.6260,"nameplate_kva":100,"meters":4},
    "KORAMGLA-DT-22":{"feeder":"F_KORAMANGALA","type":"RESIDENTIAL",     "age":9,  "lat":12.9348,"lon":77.6238,"nameplate_kva":100,"meters":4},
    "KORAMGLA-DT-23":{"feeder":"F_KORAMANGALA","type":"COMMERCIAL_HEAVY","age":15, "lat":12.9360,"lon":77.6255,"nameplate_kva":250,"meters":4},
}

# Planted anomalies (per the proposal)
ANOMALIES = {
    "A1": {
        "dt": "BTMSTG-DT-14",
        "type": "BYPASS_SIGNATURE",
        "desc": "Direct Bypass (Night) — 3 meters reduced 45% at 10pm–4am",
        "affected_meters": [0, 1, 2],   # meter indices within DT
        "start_day": 0,                 # from day 0
        "night_reduction": 0.45,        # 45% reduction
        "hours": (22, 4),               # 10pm to 4am
    },
    "A2": {
        "dt": "JAYNGR-DT-07",
        "type": "NEW_TAMPER_EVENT",
        "desc": "Slow Meter Tampering — 1 meter reads 30% low from Day 15",
        "affected_meters": [0],
        "start_day": 15,
        "tamper_factor": 0.70,          # meter reads only 70% of real consumption
    },
    "A3": {
        "dt": "KORAMGLA-DT-22",
        "type": "COMMERCIAL_DOMINANT",
        "desc": "Unbilled Connection — 15% constant extra load on DT input, no meter",
        "start_day": 0,
        "unbilled_kwh_per_15min": 2.5,  # constant unbilled load added to DT input
    },
}

# ─── TIME SETUP ───────────────────────────────────────────────────────────────

START = datetime(2026, 3, 1, 0, 0, 0)
DAYS  = 45
SLOTS_PER_DAY = 96  # 15-min slots
TOTAL_SLOTS   = DAYS * SLOTS_PER_DAY
timestamps = [START + timedelta(minutes=15*i) for i in range(TOTAL_SLOTS)]

def slot_hour(ts):
    return ts.hour + ts.minute / 60.0

def is_holiday(ts):
    # Karnataka holidays in March-April
    holidays = {(3,17),(3,25),(4,1),(4,14),(4,18)}
    return (ts.month, ts.day) in holidays

# ─── LOAD PROFILE GENERATORS ─────────────────────────────────────────────────

def residential_load(ts, base_kw=3.5):
    """Typical Bengaluru residential load profile."""
    h = slot_hour(ts)
    # Morning peak 6-9am, evening peak 6-10pm
    if 6 <= h < 9:
        factor = 1.4 + 0.3 * np.sin((h - 6) * np.pi / 3)
    elif 18 <= h < 22:
        factor = 1.8 + 0.4 * np.sin((h - 18) * np.pi / 4)
    elif 23 <= h or h < 5:
        factor = 0.3
    else:
        factor = 0.8
    # Weekend boost
    if ts.weekday() >= 5:
        factor *= 1.1
    # April AC surge (heatwave)
    if ts.month == 4:
        factor *= 1.25
    noise = np.random.normal(0, 0.05)
    return max(0.1, base_kw * factor + noise)

def commercial_light_load(ts, base_kw=8.0):
    """Light commercial (shops, offices)."""
    h = slot_hour(ts)
    if 9 <= h < 21:
        factor = 1.0 + 0.3 * np.sin((h - 9) * np.pi / 12)
    elif 21 <= h or h < 7:
        factor = 0.15
    else:
        factor = 0.4
    if ts.weekday() >= 5:
        factor *= 0.7  # lower on weekends
    if ts.month == 4:
        factor *= 1.15
    noise = np.random.normal(0, 0.06)
    return max(0.1, base_kw * factor + noise)

def commercial_heavy_load(ts, base_kw=25.0):
    """Heavy commercial/industrial — 3-shift pattern."""
    h = slot_hour(ts)
    if 8 <= h < 20:
        factor = 1.0
    elif 20 <= h < 24:
        factor = 0.85
    else:
        factor = 0.6
    noise = np.random.normal(0, 0.04)
    return max(0.5, base_kw * factor + noise)

def get_load_fn(consumer_type):
    if consumer_type == "RESIDENTIAL":      return residential_load
    if consumer_type == "COMMERCIAL_LIGHT": return commercial_light_load
    if consumer_type == "COMMERCIAL_HEAVY": return commercial_heavy_load
    return residential_load

# ─── TECHNICAL LOSS MODEL ─────────────────────────────────────────────────────

def technical_loss_pct(dt_load_kw, dt_info):
    """I²R technical loss scales with load² + base transformer core loss."""
    nameplate = dt_info["nameplate_kva"] * 0.9  # assume 0.9 PF → kW
    load_fraction = dt_load_kw / nameplate
    core_loss_pct  = 0.015  # 1.5% core loss (constant)
    copper_loss_pct = 0.04 * (load_fraction ** 2)  # I²R scales with load²
    age_factor = max(0, (dt_info["age"] - 10) * 0.0008)  # aging degradation
    return core_loss_pct + copper_loss_pct + age_factor

# ─── GENERATE METER DATA ──────────────────────────────────────────────────────

def generate_network():
    print("Generating synthetic network data...")
    
    all_meter_data   = []
    all_dt_data      = []
    dt_balance_rows  = []

    for dt_id, dt_info in DT_REGISTRY.items():
        load_fn = get_load_fn(dt_info["type"])
        n_meters = dt_info["meters"]
        
        # Generate per-meter consumption
        # Store base_kw per meter so DT input loop reuses identical values (Fix #1)
        meter_base_kw = {
            m: {
                "RESIDENTIAL":     np.random.uniform(2.5, 4.5),
                "COMMERCIAL_LIGHT":np.random.uniform(6.0, 10.0),
                "COMMERCIAL_HEAVY":np.random.uniform(20.0, 30.0),
            }.get(dt_info["type"], 3.0)
            for m in range(n_meters)
        }
        meter_readings = {}
        for m_idx in range(n_meters):
            meter_id = f"{dt_id}-M{m_idx+1:02d}"
            base_kw  = meter_base_kw[m_idx]
            
            readings = []
            for i, ts in enumerate(timestamps):
                day = i // SLOTS_PER_DAY
                kwh_15min = load_fn(ts, base_kw) * 0.25  # kW × 0.25h = kWh
                
                # Apply anomaly A1 — night bypass on BTMSTG-DT-14
                if dt_id == "BTMSTG-DT-14" and m_idx in ANOMALIES["A1"]["affected_meters"]:
                    h = slot_hour(ts)
                    # 10pm to 4am → bypass reduces meter reading
                    if h >= 22 or h < 4:
                        kwh_15min *= (1 - ANOMALIES["A1"]["night_reduction"])
                
                # Apply anomaly A2 — slow tamper on JAYNGR-DT-07 from day 15
                if dt_id == "JAYNGR-DT-07" and m_idx in ANOMALIES["A2"]["affected_meters"]:
                    if day >= ANOMALIES["A2"]["start_day"]:
                        kwh_15min *= ANOMALIES["A2"]["tamper_factor"]
                
                readings.append(kwh_15min)
                all_meter_data.append({
                    "timestamp": ts,
                    "dt_id":     dt_id,
                    "meter_id":  meter_id,
                    "consumer_type": dt_info["type"],
                    "meter_type": "POSTPAID",
                    "kwh_15min": round(kwh_15min, 4),
                    "reading_class": "VALID",
                })
            
            meter_readings[m_idx] = readings
        
        # Compute DT input = sum of consumer loads + technical losses
        # (DT input is what the transformer actually delivers — before meter tampering)
        for i, ts in enumerate(timestamps):
            day = i // SLOTS_PER_DAY
            
            # True consumer load = sum of each meter's real load (pre-tampering)
            # Uses same base_kw per meter that was used to generate meter readings
            true_consumer_kwh = 0
            for m_idx in range(n_meters):
                true_consumer_kwh += load_fn(ts, meter_base_kw[m_idx]) * 0.25
            
            tech_loss = technical_loss_pct(true_consumer_kwh / 0.25, dt_info)
            dt_input_kwh = true_consumer_kwh * (1 + tech_loss)

            # Aging transformers (age>15): extra winding resistance → loss scales with load²
            # This produces HIGH R² in regression (physics pattern), not low R² (fraud pattern)
            if dt_id in ("JAYNGR-DT-04", "KORAMGLA-DT-21"):
                # Loss proportional to load² — mimics real I²R degradation in old windings
                nameplate_kw = DT_REGISTRY[dt_id]["nameplate_kva"] * 0.9
                load_fraction = (true_consumer_kwh / 0.25) / nameplate_kw
                aging_extra_loss = true_consumer_kwh * 0.08 * (load_fraction ** 2 + 0.3)
                dt_input_kwh += aging_extra_loss
            
            # Apply anomaly A3 — unbilled connection adds constant load to DT input
            if dt_id == "KORAMGLA-DT-22":
                dt_input_kwh += ANOMALIES["A3"]["unbilled_kwh_per_15min"]
            
            # Sum of meter readings (what billing sees — may be lower due to tampering)
            sum_meters = sum(meter_readings[m][i] for m in range(n_meters))
            
            loss_kwh = dt_input_kwh - sum_meters
            loss_pct = (loss_kwh / dt_input_kwh * 100) if dt_input_kwh > 0 else 0
            
            dt_balance_rows.append({
                "timestamp":      ts,
                "dt_id":          dt_id,
                "feeder_id":      dt_info["feeder"],
                "dt_input_kwh":   round(dt_input_kwh, 4),
                "sum_meters_kwh": round(sum_meters, 4),
                "loss_kwh":       round(loss_kwh, 4),
                "loss_pct":       round(loss_pct, 4),
                "consumer_type":  dt_info["type"],
                "dt_age_years":   dt_info["age"],
                "nameplate_kva":  dt_info["nameplate_kva"],
                "n_meters":       n_meters,
            })

    meter_df  = pd.DataFrame(all_meter_data)
    balance_df = pd.DataFrame(dt_balance_rows)

    return meter_df, balance_df

# ─── L1 DATA QUALITY GATE ────────────────────────────────────────────────────

def apply_l1_quality_gate(meter_df):
    """6-class reading classifier — L1 layer."""
    def classify(row):
        v = row["kwh_15min"]
        if v < 0:                      return "SUSPECT_NEGATIVE"
        if v == 0:                     return "COMMUNICATION_DROPOUT"
        # spike: >5σ from rolling mean would be computed per meter
        # for synthetic data, flag extreme values
        if v > 50:                     return "SUSPECT_SPIKE"
        return "VALID"
    
    meter_df["reading_class"] = meter_df.apply(classify, axis=1)
    return meter_df

# ─── L2 ASSET-AWARE BASELINE ─────────────────────────────────────────────────

def compute_asset_corrections():
    """Per-DT age and PF correction factors."""
    corrections = {}
    for dt_id, info in DT_REGISTRY.items():
        age_factor = max(0, (info["age"] - 10) * 0.0008)  # extra loss per year >10yr
        pf_factor  = 0.02 if info["type"] == "COMMERCIAL_HEAVY" else 0.0
        corrections[dt_id] = {
            "age_factor":    round(age_factor, 4),
            "pf_factor":     round(pf_factor, 4),
            "total_correction": round(age_factor + pf_factor, 4),
            "pf_uncertain":  info["type"] == "COMMERCIAL_HEAVY",
        }
    return corrections

# ─── VERIFY ANOMALY VISIBILITY ───────────────────────────────────────────────

def verify_anomalies(balance_df):
    """Quick sanity check — print average loss% for anomalous DTs vs clean DTs."""
    print("\n📊 Anomaly Visibility Check:")
    print("=" * 55)
    
    for dt_id in ["BTMSTG-DT-14","JAYNGR-DT-07","KORAMGLA-DT-22",
                  "JAYNGR-DT-01","BTMSTG-DT-09"]:  # last 2 are clean
        subset = balance_df[balance_df["dt_id"] == dt_id]
        avg_loss = subset["loss_pct"].mean()
        max_loss = subset["loss_pct"].max()
        label = ""
        if dt_id == "BTMSTG-DT-14":  label = "← A1 BYPASS (should be HIGH)"
        if dt_id == "JAYNGR-DT-07":  label = "← A2 TAMPER (high after day 15)"
        if dt_id == "KORAMGLA-DT-22":label = "← A3 UNBILLED (constant elevated)"
        print(f"  {dt_id:20s}  avg_loss={avg_loss:5.1f}%  max={max_loss:5.1f}%  {label}")
    
    print("=" * 55)

# ─── MAIN ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    BASE_DIR = "data"
    os.makedirs(BASE_DIR, exist_ok=True)
    
    # Generate data
    meter_df, balance_df = generate_network()
    
    # L1 quality gate
    meter_df = apply_l1_quality_gate(meter_df)
    
    # L2 asset corrections
    asset_corrections = compute_asset_corrections()
    
    # Save
    meter_df.to_parquet(f"{BASE_DIR}/meter_readings.parquet", index=False)
    balance_df.to_parquet(f"{BASE_DIR}/dt_balance.parquet", index=False)
    
    with open(f"{BASE_DIR}/dt_registry.json", "w") as f:
        json.dump(DT_REGISTRY, f, indent=2)
    
    with open(f"{BASE_DIR}/asset_corrections.json", "w") as f:
        json.dump(asset_corrections, f, indent=2)
    
    with open(f"{BASE_DIR}/anomalies_ground_truth.json", "w") as f:
        json.dump(ANOMALIES, f, indent=2)
    
    # Report
    print(f"✅ Meter readings:  {len(meter_df):,} rows  ({len(meter_df.meter_id.unique())} meters)")
    print(f"✅ DT balance:      {len(balance_df):,} rows ({len(balance_df.dt_id.unique())} DTs × {TOTAL_SLOTS} slots)")
    print(f"✅ Asset corrections saved for {len(asset_corrections)} DTs")
    
    verify_anomalies(balance_df)
    
    # Quick sample
    print("\nSample DT balance rows:")
    print(balance_df[balance_df["dt_id"]=="BTMSTG-DT-14"].tail(5)[
        ["timestamp","dt_input_kwh","sum_meters_kwh","loss_pct"]].to_string())
    
    print("\n✅ Day 1 data generation complete. Ready for fingerprinting engine.")
