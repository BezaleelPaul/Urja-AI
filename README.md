# ⚡ Urja AI v2.0 — BESCOM Smart Grid Intelligence





> **Network-Topology-Aware Energy Intelligence Platform**
> AI-Powered Electricity Theft Detection \& Revenue Recovery for BESCOM, Bengaluru
---

**PAN IIT · AI for Bharat 2026 · Theme 8 · Smart Meters**
📍 Taj Yeshwantpur, Bengaluru · May 16, 2026

\---





## 👥 Team

|Role|Name|
|-|-|
|Lead Full Stack Developer|Yeshwanth Reddy P G|
|Researcher|Aditya S|
|Debugger|Bezaleel Paul N|

**Institution:** CMR University · 1st Year CSE / AI-ML

\---





## 🎥 Demo Video



> 📺 \*\*\[Watch Demo Video — YouTube](#)\*\* ← \*(replace with your link)\*

> 🌐 \*\*\[Live Streamlit App](#)\*\* ← \*(replace with your deployment link)\*

\---





## 🚨 The Problem



India's electricity distribution utilities lose over **₹50,000 Crore annually** due to Aggregate Technical \& Commercial (AT\&C) losses averaging **26.5% nationally**. BESCOM, Bengaluru's primary distributor, faces:

* **Direct theft / meter bypass** — wires bypassed at night, completely invisible to SCADA systems
* **Meter tampering** — meters manipulated to under-read by 30–70% from a specific date
* **Unbilled connections** — illegal taps drawing constant load with zero billing record
* **Inspector alert fatigue** — hundreds of low-confidence alerts daily cause inspectors to stop responding within weeks
* **Aging transformer noise** — old DTs misclassified as fraud, wasting inspector bandwidth on futile visits

Every 1% reduction in AT\&C loss = direct, measurable revenue recovery for the utility.

\---





## 💡 The Core Insight — Energy Forensics



The entire detection engine rests on one physics-backed observation:

|Loss Type|Behaviour|Statistical Signature|
|-|-|-|
|**Technical (I²R)**|Scales with load² — heat dissipation in cables|R² → 1.0, intercept < 3%|
|**Commercial (Fraud)**|Fixed daily steal regardless of feeder load|R² → 0, intercept > 10%|

A meter bypass stealing 200 kWh/night does so whether the feeder runs at 30% or 90% load. This **load-independence** is the fraud fingerprint — detectable with **5 lines of `scipy.stats.linregress`**.

```python
corrected\_loss = loss\_pct - age\_factor - pf\_factor
slope, intercept, r\_value, p\_value, std\_err = linregress(dt\_input\_kwh, corrected\_loss)
r\_squared = r\_value \*\* 2
# → Classify into 5 patterns using R², intercept, night\_peak, step\_change
```

This approach is **fully explainable** — any BESCOM field engineer can verify the logic on a whiteboard. No black-box ML required for the core detection.

\---





## 🏗️ System Architecture — 6-Layer Hardened Pipeline


┌─────────────────────────────────────────────────────────────────┐
│  L1 │ Data Quality \& Clock Sync Gate                            │
│     │ 6-class reading classifier: VALID / SUSPECT\_SPIKE /       │
│     │ MISSING / DROPOUT / PREPAID\_DEPLETION / NEGATIVE          │
├─────────────────────────────────────────────────────────────────┤
│  L2 │ Asset-Aware Baseline Correction                           │
│     │ DT age factor + power factor correction per consumer      │
│     │ type. Aging DTs get separate intercept baseline.          │
├─────────────────────────────────────────────────────────────────┤
│  L3 │ Network Energy Balance Engine                             │
│     │ Loss = DT Input − ΣMeters, every 15 minutes.              │
│     │ Cohort-median imputation for dead/missing meters.         │
├─────────────────────────────────────────────────────────────────┤
│  L4 │ Loss Fingerprinting Engine  ← CORE                        │
│     │ Per-DT linear regression over 30-day rolling window.      │
│     │ R² + intercept + night-peak → 5-pattern taxonomy.         │
├─────────────────────────────────────────────────────────────────┤
│  L5 │ STL + LightGBM Demand Forecasting                         │
│     │ STL decomposition → LightGBM on residuals.                │
│     │ 12 features. SHAP pre-cached. 24h headroom score per DT.  │
├─────────────────────────────────────────────────────────────────┤
│  L6 │ Inspector Priority Queue Generator                        │
│     │ Revenue-ranked dispatch via log1p scoring.                │
│     │ Separate inspection vs maintenance queues. Evidence briefs│
└─────────────────────────────────────────────────────────────────┘


Each layer is independently testable. No silent failures. All signals are typed and logged.

\---





## ✨ Features



### 🔍 Loss Fingerprinting Engine (L4)



Classifies every Distribution Transformer into one of **5 patterns**:

|Pattern|Trigger|Priority|Action|
|-|-|-|-|
|`BYPASS\_SIGNATURE`|Night peak > 20× day + intercept > 10%|🔴 CRITICAL|Night inspection (10pm–2am)|
|`NEW\_TAMPER\_EVENT`|Step jump > 30% of first-half baseline|🟠 HIGH|Daytime meter audit|
|`COMMERCIAL\_DOMINANT`|R² > 0.70 + intercept > 20%|🟠 HIGH|Unbilled connection hunt|
|`AGING\_TRANSFORMER`|Age > 15yr + intercept 2.5–6%|🔵 MAINTENANCE|Transformer replacement queue|
|`TECHNICAL\_CLEAN`|Low intercept, no fraud signal|🟢 LOW|Monitor only|

### 

### 📊 STL + LightGBM Demand Forecasting (L5)



* **STL decomposition** separates trend, weekly \& daily seasonality, then LightGBM learns only the residuals — preventing overfit to repeating patterns
* **12 engineered features**: time slots, lag features (t-1h, t-24h, t-168h), rolling stats, Karnataka public holidays
* **SHAP explainability** pre-computed and cached — zero live-inference latency during demo
* **Headroom Score** = (Capacity − Forecasted Peak) / Capacity → RED < 15% · AMBER 15–30% · GREEN > 30%
* **Cold-start protocol**: DTs with < 21 days history get proxy forecasts from 3 nearest similar DTs





### 🚨 Revenue-Ranked Inspector Queue (L6)


score = priority\_weight (CRITICAL=100) + min(50, log1p(estimated\_revenue\_loss)) + min(20, days\_persistent)

---

`log1p` ensures a CRITICAL night bypass stays **Rank #1** even when its rupee value is lower than an unbilled connection — physical risk takes precedence over financial weight.





### 🌐 4-Page Streamlit Dashboard



|Page|What it shows|
|-|-|
|📍 Zone Risk Map|Plotly Scattermap, 15 DTs color-coded by priority, live KPI row|
|🚨 Inspector Queue|Revenue-ranked dispatch cards, evidence strings, dual queues|
|🔬 DT Drill-Down|Loss-Load scatter, regression overlay, time-series anomaly markers|
|📊 Loss Scatter|R² vs Intercept fingerprint map for all 15 DTs|

\---



## 🧪 Detection Results — 3/3 Anomalies Found, 0 False Positives



|Rank|Priority|DT|Pattern|Key Signal|Revenue/Month|
|-|-|-|-|-|-|
|#1|🔴 CRITICAL|BTMSTG-DT-14|BYPASS\_SIGNATURE|R²=0.34, Night/Day=20×, Intercept=20.93%|₹1,847–₹2,499|
|#2|🟠 HIGH|KORAMGLA-DT-22|COMMERCIAL\_DOMINANT|R²=0.86, Intercept=80.57% (constant unbilled load)|₹26,534–₹35,900|
|#3|🟠 HIGH|JAYNGR-DT-07|NEW\_TAMPER\_EVENT|R²=0.02, Step change +5.65% from Day 15|₹3,631–₹4,912|

**Correctly routed to MAINTENANCE (not inspection):**

* JAYNGR-DT-04 (age 20yr) + KORAMGLA-DT-21 (age 22yr) — intercept 3.25% explained by physics, not fraud

**Total recoverable revenue: ₹31,312–₹43,311/month → ₹3.75L–₹5.2L/year**

\---



## 📁 Repository Structure


urja-ai-v2/
├── app.py                  # Streamlit dashboard (4 pages)
├── generate\_data.py        # Synthetic network + energy balance engine
├── fingerprinting.py       # Loss fingerprinting + LightGBM + priority queue
├── requirements.txt        # All pip-installable dependencies
├── README.md
│
├── data/
│   ├── meter\_readings.parquet      # 259,200 smart meter readings
│   ├── dt\_registry.json            # DT metadata (age, type, capacity)
│   ├── feeder\_topology.json        # Network topology map
│   ├── energy\_balance.parquet      # 15-minute loss calculations
│   ├── fingerprinting\_results.parquet
│   └── inspector\_queue.parquet
│
└── models/
    ├── lgbm\_dt\_\*.pkl           # 15 trained LightGBM models (one per DT)
    └── shap\_cache.json         # Pre-computed SHAP values

\---





## 🚀 Quick Start



### Prerequisites

* Python 3.9+
* No GPU required

### 1\. Clone \& install

```bash
git clone https://github.com/YOUR\_USERNAME/urja-ai-v2.git
cd urja-ai-v2
pip install -r requirements.txt

### 2\. Generate synthetic data

```bash
python generate\_data.py
```

Generates 259,200 meter readings across 3 feeders, 15 DTs, 60 smart meters over 45 days. Plants 3 anomalies. Runs in \~2–3 minutes.

### 3\. Run intelligence layers

```bash
python fingerprinting.py
```

Trains 15 LightGBM models, runs fingerprinting on all DTs, pre-computes SHAP values, generates inspector queue. Runs in \~3–5 minutes.

### 4\. Launch demo

```bash
streamlit run app.py

Opens at `http://localhost:8501`

\---





## 🛠️ Tech Stack



|Layer|Tools|
|-|-|
|**Data**|pandas, numpy, scipy.stats, pyarrow|
|**Core Detection**|scipy.stats.linregress (5 lines = full algorithm)|
|**ML / Forecasting**|lightgbm, statsmodels STLForecast, scikit-learn|
|**Explainability**|shap (TreeExplainer, pre-cached)|
|**Dashboard**|streamlit, plotly (Scattermap + express + graph\_objects)|
|**Persistence**|parquet files, JSON configs, pickle models|

100% open source. 100% pip-installable. No proprietary APIs. No cloud dependencies.

\---



## 🧠 Synthetic Network Specification



**3 Feeders · 15 Distribution Transformers · 60 Smart Meters · 45 Days · 15-minute slots**

|Feeder|DTs|Consumer Mix|Transformer Age|
|-|-|-|-|
|F\_JAYANAGAR|5|Residential heavy|6–20 years|
|F\_BTM|5|Commercial Light + Heavy|3–18 years|
|F\_KORAMANGALA|5|Mixed (holds all 3 anomalies)|7–22 years|

**Load profile models:**

* 🏠 **RESIDENTIAL** — morning + evening peaks; April AC surge +25%
* 🏪 **COMMERCIAL\_LIGHT** — 9am–9pm; weekends −30%
* 🏭 **COMMERCIAL\_HEAVY** — 3-shift pattern (factors: 0.6 / 0.85 / 1.0)
* Technical loss modelled as: `I²R(load²) + core 1.5% + DT age factor`

**Planted anomalies:**

|ID|Pattern|Location|Specification|
|-|-|-|-|
|A1|BYPASS\_SIGNATURE|BTMSTG-DT-14|3 meters × 45% reduction, nightly 10pm–4am|
|A2|NEW\_TAMPER\_EVENT|JAYNGR-DT-07|1 meter reads 70% of true value from Day 15|
|A3|COMMERCIAL\_DOMINANT|KORAMGLA-DT-22|+2.5 kWh/slot constant unbilled load, no meter|

\---





## 🛡️ 11 Hardening Guards (v2.0)



Urja AI v2.0 was stress-tested from the perspective of a hostile technical judge, a BESCOM field engineer, and a competing team. 11 loopholes were identified and resolved:

|ID|Loophole|Resolution|
|-|-|-|
|L01|Clock sync noise|1hr rolling avg for high-variance DTs + materiality floor|
|L02|Power factor on industrial DTs|Consumer-type PF correction; PF-UNCERTAIN raises threshold|
|L03|April AC surge breaks persistence|Persistence window extends 21→35 days during seasonal surge|
|L04|Cohort too small for imputation|Min 15 meters; fallback feeder+type → zone+type|
|L05|Aging DTs mimic fraud intercept|Age + intercept + no night\_peak → MAINTENANCE, not inspection|
|L06|Flat tariff ignores consumer type|Residential ₹4.5/unit, Commercial ₹7.5/unit — shown as range|
|L07|DT input meter itself tampered|DT vs feeder topology mismatch → routes to metering team|
|L08|Cold-start new DTs get wrong scores|Proxy forecast; max MEDIUM priority until 21-day native window|
|L09|Prepaid meter depletion = false spike|`meter\_type` field; depletion events excluded from anomaly logic|
|L10|Inspectors game the queue|Closure outcome embedded in record; quality score per inspector|
|L11|Demo build risk (live SHAP crash)|SHAP pre-cached; all values cast to `np.float64` before JSON serialisation|

\---





## 🔭 Future Scope



* **Real BESCOM AMI integration** — replace synthetic data with live AMI feed via REST/MQTT
* **GIS topology import** — ingest actual feeder topology from BESCOM GIS for true network-aware loss attribution
* **Inspector mobile app** — offline-capable PWA with on-site evidence capture and closure feedback loop
* **Multi-utility expansion** — parameterise for TANGEDCO, MSEDCL, CESC with utility-specific tariff configs
* **Ensemble anomaly detection** — add Isolation Forest as a secondary signal for novel theft patterns
* **Regulatory dashboard** — automated AT\&C loss reporting in CERC/SERC formats

\---





## 🙏 Acknowledgements



Built for **PAN IIT · AI for Bharat 2026 · Theme 8** in partnership with BESCOM, Bengaluru.



###### ***> "The Model Classifies. The Inspector Decides. The Revenue Recovers. The System Improves."***

