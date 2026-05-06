"""
convert_real_data_patched.py — Fixed Real CSV → Urja AI Pipeline
=================================================================
Fixes two bugs in the original convert_real_data.py:

  BUG 1: fingerprinting.py requires asset_corrections.json
          Original script never generates it → immediate crash on real data
          FIX: generate it from the DT registry (same logic as generate_data.py)

  BUG 2: UCI dataset = 1 household = 1 meter = 1 DT
          With only 1 DT, fingerprinting has nothing to compare and always
          returns TECHNICAL_CLEAN or INSUFFICIENT_LOAD_VARIANCE
          FIX: UCI mode splits the single meter's time-series into N synthetic
          "sub-meters" with different usage patterns, groups them into multiple
          DTs, then plants realistic anomalies so you can see varied results.

Usage (UCI — most common):
    python convert_real_data_patched.py --input household_power_consumption.txt --dataset uci

Usage (CEEW / multi-meter India datasets):
    python convert_real_data_patched.py --input ceew.csv --dataset generic \
        --time_col timestamp --kwh_col energy_kwh --meter_col meter_id

Usage (Pecan Street):
    python convert_real_data_patched.py --input pecan.csv --dataset pecan
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

np.random.seed(42)

BASE_DIR      = "."
DATA_DIR      = f"{BASE_DIR}/data"   # matches fingerprinting.py + app.py ("data/")
METERS_PER_DT = 4
RESAMPLE_FREQ = "15min"

BANGALORE_LAT_RANGE = (12.90, 12.98)
BANGALORE_LON_RANGE = (77.55, 77.65)

# Feeder/type assignments to make real data look like a BESCOM network
FEEDER_POOL = ["F_JAYANAGAR", "F_BTM", "F_KORAMANGALA"]
TYPE_POOL   = ["RESIDENTIAL", "RESIDENTIAL", "RESIDENTIAL",
               "COMMERCIAL_LIGHT", "COMMERCIAL_HEAVY"]

TARIFF = {
    "RESIDENTIAL":     4.5,
    "COMMERCIAL_LIGHT":7.5,
    "COMMERCIAL_HEAVY":6.5,
}


# ─── TECHNICAL LOSS (same as generate_data.py) ────────────────────────────────

def technical_loss_pct(load_kw, nameplate_kva=100, age_years=10):
    nameplate_kw  = nameplate_kva * 0.9
    load_fraction = load_kw / max(nameplate_kw, 1)
    core_loss     = 0.015
    copper_loss   = 0.04 * (load_fraction ** 2)
    age_factor    = max(0, (age_years - 10) * 0.0008)
    return core_loss + copper_loss + age_factor


# ─── LOADERS ─────────────────────────────────────────────────────────────────

def load_uci(path: str) -> pd.DataFrame:
    print("  Loading UCI dataset...")
    df = pd.read_csv(
        path, sep=";",
        parse_dates={"timestamp": ["Date", "Time"]},
        dayfirst=True, na_values=["?"], low_memory=False,
    )
    df = df.dropna(subset=["Global_active_power"])
    df["kwh"] = pd.to_numeric(df["Global_active_power"], errors="coerce") / 60
    df["meter_id"] = "METER-001"
    return df[["timestamp", "meter_id", "kwh"]].dropna()


def load_pecan(path: str) -> pd.DataFrame:
    print("  Loading Pecan Street dataset...")
    df = pd.read_csv(path, parse_dates=["localminute"], low_memory=False)
    df = df.rename(columns={"localminute": "timestamp", "dataid": "meter_id"})
    col = "use" if "use" in df.columns else "grid"
    df["kwh"] = df[col] / 60
    df["kwh"] = df["kwh"].clip(lower=0)
    df["meter_id"] = "METER-" + df["meter_id"].astype(str).str.zfill(4)
    return df[["timestamp", "meter_id", "kwh"]].dropna()


def load_generic(path, time_col, kwh_col, meter_col=None) -> pd.DataFrame:
    print("  Loading generic CSV dataset...")
    df = pd.read_csv(path, parse_dates=[time_col], low_memory=False)
    df = df.rename(columns={time_col: "timestamp", kwh_col: "kwh"})
    if meter_col and meter_col in df.columns:
        df["meter_id"] = "METER-" + df[meter_col].astype(str).str.zfill(4)
    else:
        df["meter_id"] = "METER-0001"
    df["kwh"] = pd.to_numeric(df["kwh"], errors="coerce").clip(lower=0)
    return df[["timestamp", "meter_id", "kwh"]].dropna()


# ─── UCI EXPANSION (the real fix) ─────────────────────────────────────────────

def expand_uci_to_network(df: pd.DataFrame, n_meters: int = 20) -> pd.DataFrame:
    """
    UCI = 1 meter. Fingerprinting needs many meters across many DTs.

    Strategy: take the real UCI time-series and create N synthetic sub-meters
    by applying realistic scaling + noise. This preserves the real temporal
    patterns (morning peaks, evening peaks, weekends, seasons) while giving
    each 'meter' a different load level — exactly like real households.

    We then plant anomalies on specific meters so fingerprinting sees:
      - 1 DT with bypass signature   (night-time underreporting)
      - 1 DT with tamper step-change (mid-period sudden drop)
      - 1 DT with unbilled load      (constant extra DT input)
      - remaining DTs clean / aging  (normal physics)
    """
    print(f"  Expanding 1 UCI meter → {n_meters} synthetic sub-meters...")

    # Work with 45 days for demo (same as generate_data.py)
    df = df.sort_values("timestamp").copy()
    df = df[df["timestamp"] >= df["timestamp"].min() + pd.Timedelta(days=30)]
    df = df.head(45 * 24 * 60)  # ~45 days at 1-min resolution

    base_series = df.set_index("timestamp")["kwh"]

    rows = []
    rng = np.random.default_rng(42)

    for m in range(n_meters):
        meter_id = f"METER-{m+1:03d}"

        # Each meter gets a different scale (household size diversity)
        scale   = rng.uniform(0.4, 2.5)
        # Phase shift: different households peak at slightly different times
        phase_h = rng.integers(-1, 2)  # -1, 0, or +1 hour shift
        noise_std = 0.03

        shifted = base_series.copy()
        if phase_h != 0:
            shifted.index = shifted.index + pd.Timedelta(hours=phase_h)
            shifted = shifted.reindex(base_series.index, method="nearest")

        kwh_col = (shifted * scale + rng.normal(0, noise_std, len(shifted))).clip(lower=0)
        tmp = pd.DataFrame({
            "timestamp": kwh_col.index,
            "meter_id":  meter_id,
            "kwh":       kwh_col.values,
        })
        rows.append(tmp)

    expanded = pd.concat(rows, ignore_index=True)
    print(f"  → Expanded to {len(expanded):,} rows, {n_meters} meters")
    return expanded


def plant_anomalies_on_resampled(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Plant 3 known anomalies into the resampled 15-min data.
    Returns modified df and ground_truth dict (same format as generate_data.py).

    Anomaly assignment (by DT number):
      REAL-DT-01: BYPASS — meters 1,2,3 read 45% less at night (10pm–4am)
      REAL-DT-02: TAMPER — meter 5 reads 30% less from the midpoint of data
      REAL-DT-03: UNBILLED — 2.5 kWh constant extra added to DT input (dt_balance)
      Others: clean / physics only
    """
    print("  Planting 3 anomalies on DTs 01–03...")
    df = df.copy()

    # --- A1: Night bypass on REAL-DT-01, meters 1-3 ---
    bypass_meters = {"METER-001", "METER-002", "METER-003"}
    night_mask = (
        (df["dt_id"] == "REAL-DT-01") &
        (df["meter_id"].isin(bypass_meters)) &
        (df["timestamp"].dt.hour.isin([22, 23, 0, 1, 2, 3]))
    )
    df.loc[night_mask, "kwh_15min"] *= 0.55  # meter reads 55% of real

    # --- A2: Tamper on REAL-DT-02, meter 5 from midpoint ---
    midpoint = df["timestamp"].min() + (df["timestamp"].max() - df["timestamp"].min()) / 2
    tamper_mask = (
        (df["dt_id"] == "REAL-DT-02") &
        (df["meter_id"] == "METER-005") &
        (df["timestamp"] >= midpoint)
    )
    df.loc[tamper_mask, "kwh_15min"] *= 0.70

    ground_truth = {
        "A1": {
            "dt":   "REAL-DT-01",
            "type": "BYPASS_SIGNATURE",
            "desc": "Night bypass — meters 001-003 read 45% less from 10pm–4am",
        },
        "A2": {
            "dt":   "REAL-DT-02",
            "type": "NEW_TAMPER_EVENT",
            "desc": "Slow tamper — meter 005 reads 30% less from data midpoint",
        },
        "A3": {
            "dt":   "REAL-DT-03",
            "type": "COMMERCIAL_DOMINANT",
            "desc": "Unbilled connection — 2.5 kWh/15min constant extra on DT input",
        },
    }
    return df, ground_truth


# ─── CORE PIPELINE ────────────────────────────────────────────────────────────

def resample_to_15min(df: pd.DataFrame) -> pd.DataFrame:
    print("  Resampling to 15-min slots...")
    df = df.set_index("timestamp")
    df = (
        df.groupby("meter_id")
          .resample(RESAMPLE_FREQ)["kwh"]
          .sum()
          .reset_index()
    )
    df = df.rename(columns={"kwh": "kwh_15min"})
    df = df[df["kwh_15min"] > 0]
    print(f"  → {len(df):,} rows, {df['meter_id'].nunique()} meters")
    return df


def assign_dts(df: pd.DataFrame) -> pd.DataFrame:
    unique_meters = sorted(df["meter_id"].unique())
    meter_to_dt = {}
    for i, m in enumerate(unique_meters):
        dt_num = i // METERS_PER_DT
        meter_to_dt[m] = f"REAL-DT-{dt_num+1:02d}"
    df["dt_id"] = df["meter_id"].map(meter_to_dt)
    n_dts = df["dt_id"].nunique()
    print(f"  → {len(unique_meters)} meters → {n_dts} DTs ({METERS_PER_DT} meters each)")
    return df


def apply_l1_quality_gate(df: pd.DataFrame) -> pd.DataFrame:
    def classify(v):
        if v < 0:   return "SUSPECT_NEGATIVE"
        if v == 0:  return "COMMUNICATION_DROPOUT"
        if v > 50:  return "SUSPECT_SPIKE"
        return "VALID"
    df["reading_class"] = df["kwh_15min"].apply(classify)
    return df


def build_registry(dt_ids) -> dict:
    registry = {}
    lat_min, lat_max = BANGALORE_LAT_RANGE
    lon_min, lon_max = BANGALORE_LON_RANGE
    rng = np.random.default_rng(0)

    feeders = FEEDER_POOL
    types   = TYPE_POOL

    for i, dt_id in enumerate(sorted(dt_ids)):
        registry[dt_id] = {
            "feeder":        feeders[i % len(feeders)],
            "type":          types[i % len(types)],
            "age":           int(rng.integers(5, 23)),
            "lat":           round(rng.uniform(lat_min, lat_max), 4),
            "lon":           round(rng.uniform(lon_min, lon_max), 4),
            "nameplate_kva": 100,
            "meters":        METERS_PER_DT,
        }
    return registry


# ─── FIX: generate asset_corrections.json (was missing) ──────────────────────

def build_asset_corrections(registry: dict) -> dict:
    """
    Generate asset_corrections.json.
    fingerprinting.py REQUIRES this file — original convert_real_data.py
    never created it, causing an immediate FileNotFoundError on real data.
    Same formula as generate_data.py's compute_asset_corrections().
    """
    corrections = {}
    for dt_id, info in registry.items():
        age_factor = max(0, (info["age"] - 10) * 0.0008)
        pf_factor  = 0.02 if info["type"] == "COMMERCIAL_HEAVY" else 0.0
        corrections[dt_id] = {
            "age_factor":       round(age_factor, 4),
            "pf_factor":        round(pf_factor, 4),
            "total_correction": round(age_factor + pf_factor, 4),
            "pf_uncertain":     info["type"] == "COMMERCIAL_HEAVY",
        }
    return corrections


# ─── DT BALANCE ───────────────────────────────────────────────────────────────

def build_dt_balance(df: pd.DataFrame, registry: dict,
                     ground_truth: dict) -> pd.DataFrame:
    print("  Building DT balance with physics losses + anomaly A3...")

    dt_df = (
        df.groupby(["timestamp", "dt_id"])["kwh_15min"]
          .sum()
          .reset_index()
          .rename(columns={"kwh_15min": "sum_meters_kwh"})
    )

    def add_loss(row):
        info     = registry[row["dt_id"]]
        load_kw  = row["sum_meters_kwh"] / 0.25
        loss_pct = technical_loss_pct(load_kw, info["nameplate_kva"], info["age"])
        dt_input = row["sum_meters_kwh"] * (1 + loss_pct)

        # Anomaly A3: constant unbilled load on DT-03
        if row["dt_id"] == "REAL-DT-03":
            dt_input += 2.5  # kWh per 15-min slot, unmetered

        loss_kwh = dt_input - row["sum_meters_kwh"]
        loss_pct_val = (loss_kwh / dt_input * 100) if dt_input > 0 else 0
        return pd.Series({
            "dt_input_kwh": round(dt_input, 4),
            "loss_kwh":     round(loss_kwh, 4),
            "loss_pct":     round(loss_pct_val, 4),
        })

    extras = dt_df.apply(add_loss, axis=1)
    dt_df  = pd.concat([dt_df, extras], axis=1)

    dt_df["feeder_id"]    = dt_df["dt_id"].map(
        {k: v["feeder"] for k, v in registry.items()})
    dt_df["consumer_type"] = dt_df["dt_id"].map(
        {k: v["type"] for k, v in registry.items()})
    dt_df["dt_age_years"] = dt_df["dt_id"].map(
        {k: v["age"] for k, v in registry.items()})
    dt_df["nameplate_kva"] = 100
    n_meters_map = df.groupby("dt_id")["meter_id"].nunique().to_dict()
    dt_df["n_meters"] = dt_df["dt_id"].map(n_meters_map)

    return dt_df[[
        "timestamp", "dt_id", "feeder_id",
        "dt_input_kwh", "sum_meters_kwh", "loss_kwh", "loss_pct",
        "consumer_type", "dt_age_years", "nameplate_kva", "n_meters",
    ]]


def build_meter_readings(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["consumer_type"] = "RESIDENTIAL"
    df["meter_type"]    = "POSTPAID"
    return df[["timestamp", "dt_id", "meter_id",
               "consumer_type", "meter_type",
               "kwh_15min", "reading_class"]]


# ─── SAVE ─────────────────────────────────────────────────────────────────────

def save_outputs(meter_df, balance_df, registry,
                 asset_corrections, ground_truth):
    os.makedirs(DATA_DIR, exist_ok=True)

    meter_df.to_parquet(f"{DATA_DIR}/meter_readings.parquet", index=False)
    balance_df.to_parquet(f"{DATA_DIR}/dt_balance.parquet",   index=False)

    with open(f"{DATA_DIR}/dt_registry.json", "w") as f:
        json.dump(registry, f, indent=2)

    with open(f"{DATA_DIR}/asset_corrections.json", "w") as f:  # ← FIX: was missing
        json.dump(asset_corrections, f, indent=2)

    with open(f"{DATA_DIR}/anomalies_ground_truth.json", "w") as f:
        json.dump(ground_truth, f, indent=2)

    print(f"\n✅ meter_readings.parquet       → {len(meter_df):,} rows")
    print(f"✅ dt_balance.parquet           → {len(balance_df):,} rows")
    print(f"✅ dt_registry.json             → {len(registry)} DTs")
    print(f"✅ asset_corrections.json       → {len(asset_corrections)} DTs  ← NEW")
    print(f"✅ anomalies_ground_truth.json  → {len(ground_truth)} anomalies")
    print(f"\nAll files saved to: {DATA_DIR}/")


def sanity_check(balance_df):
    print("\n📊 Loss% by DT (top 8):")
    print("=" * 60)
    summary = (
        balance_df.groupby("dt_id")["loss_pct"]
        .agg(["mean", "max"])
        .round(2)
        .sort_values("mean", ascending=False)
        .head(8)
    )
    for dt_id, row in summary.iterrows():
        label = ""
        if dt_id == "REAL-DT-01": label = " ← A1 BYPASS (night loss)"
        if dt_id == "REAL-DT-02": label = " ← A2 TAMPER (step change)"
        if dt_id == "REAL-DT-03": label = " ← A3 UNBILLED (constant extra)"
        print(f"  {dt_id}  avg={row['mean']:5.1f}%  max={row['max']:5.1f}%{label}")
    print("=" * 60)
    print("\nIf DT-01/02/03 show elevated loss → anomalies planted correctly ✅")


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Convert real smart meter CSV → Urja AI (patched)"
    )
    parser.add_argument("--input",   required=True)
    parser.add_argument("--dataset", required=True,
                        choices=["uci", "pecan", "generic"])
    parser.add_argument("--time_col",  default="timestamp")
    parser.add_argument("--kwh_col",   default="kwh")
    parser.add_argument("--meter_col", default=None)
    parser.add_argument("--n_meters",  type=int, default=20,
                        help="[uci only] How many sub-meters to synthesise (default 20 = 5 DTs)")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"ERROR: File not found: {args.input}")
        sys.exit(1)

    print(f"\n🔄 Converting {args.input} ({args.dataset} format)...")

    # 1. Load raw data
    if args.dataset == "uci":
        raw = load_uci(args.input)
        # UCI is 1 meter → expand to n_meters synthetic sub-meters
        raw = expand_uci_to_network(raw, n_meters=args.n_meters)
    elif args.dataset == "pecan":
        raw = load_pecan(args.input)
    else:
        raw = load_generic(args.input, args.time_col, args.kwh_col, args.meter_col)

    print(f"  Raw rows: {len(raw):,}")

    # 2. Resample → 15-min
    resampled = resample_to_15min(raw)

    # 3. Assign DTs
    resampled = assign_dts(resampled)

    # 4. Plant anomalies (UCI + generic: always; Pecan: skipped if already multi-meter)
    if args.dataset in ("uci", "generic") and resampled["dt_id"].nunique() >= 3:
        resampled, ground_truth = plant_anomalies_on_resampled(resampled)
    else:
        ground_truth = {}

    # 5. L1 quality gate
    resampled = apply_l1_quality_gate(resampled)

    # 6. Build registry + asset corrections (FIX: generate what fingerprinting.py needs)
    registry          = build_registry(resampled["dt_id"].unique())
    asset_corrections = build_asset_corrections(registry)

    # 7. Build output tables
    meter_df   = build_meter_readings(resampled)
    balance_df = build_dt_balance(resampled, registry, ground_truth)

    # 8. Save everything
    save_outputs(meter_df, balance_df, registry, asset_corrections, ground_truth)

    # 9. Sanity check
    sanity_check(balance_df)

    print("\n🚀 Done! Run fingerprinting.py next, then app.py")
    print("   Expected: DT-01=BYPASS, DT-02=TAMPER, DT-03=UNBILLED")


if __name__ == "__main__":
    main()
