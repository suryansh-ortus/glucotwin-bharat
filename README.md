# GlucoTwin Bharat
### A live, privacy-first digital twin for Type 2 diabetes in Indian patients

> Submission for the **Happiest Health Digital Twin Challenge 2026**: Reimagining and Reforming Healthcare in India Summit, Bengaluru.

---

## 1. Team details

**Team name:** Because I'm Happy Clap Along

| Name | Branch | Email |
|---|---|---|
| Diya Jain *(Team Leader)* | B.Tech, Computer Science and Engineering (CSE) | dia.jainn.ug23@nsut.ac.in |
| Suryansh Malhotra | B.Tech, Instrumentation and Control Engineering (ICE) | suryansh.malhotra.ug23@nsut.ac.in |

## 2. College / Incubator information

| | |
|---|---|
| College | Netaji Subhas University of Technology (NSUT), New Delhi |
| Program | B.Tech |

## 3. Project title

**GlucoTwin Bharat: a Type 2 diabetes digital twin that predicts glucose spikes and lows 2 hours ahead by fusing health-record data with wearable streams.**

## 4. Problem statement

India has more than 100 million people living with diabetes, and many of them see a doctor only every few months. Between visits, doctors are blind to what actually happens: post-meal spikes from high-GI Indian meals, dangerous lows on fasting days for patients on sulfonylureas, the effect of poor sleep, or an infection quietly pushing glucose up.

Existing tools either show raw CGM graphs or run black-box predictors trained on foreign datasets. Neither explains *why* something is happening, and neither lets a doctor test a change before prescribing it. Patient data is also protected by the **DPDP Act**, so cloud-heavy AI is hard to deploy.

**Goal:** build a proof-of-concept digital twin that fuses **static EHR data** (labs, diagnoses, medicines, genetics) with **dynamic wearable data** (CGM, heart rate, HRV, PPG, sleep, steps, meals), and **predicts adverse events** with a doctor-facing dashboard. It must run on the device and keep identity private.

## 5. Healthcare use case

The dashboard is used by the **doctor**. The patient wears a CGM and a smartwatch and logs meals on their phone.

| Adverse event | How the twin catches it | Demo patient |
|---|---|---|
| **Post-meal glucose spike** | 2-h forecast + calibrated probability of crossing 180 mg/dL, with about 35 min median warning | Rajesh, 52M: late biryani after two short nights |
| **Hypoglycaemia** | Probability of falling below 70 mg/dL, with the medicine flagged | Lakshmi, 63F: on glimepiride, fasting day + temple walk |
| **Hidden illness / missed medication** | "Twin divergence": the twin re-fits itself every 2 h and flags when the patient no longer matches their record | Rajesh: viral fever not in the EHR |
| **Cardiovascular strain** | HRV computed from raw PPG + resting HR + sleep vs personal baseline | Arjun, 44M: thin-fat phenotype, stressful work week |

The doctor can also:
- **Ask "what if"** (e.g. "swap to millet roti and walk 20 min") and see both futures simulated.
- **Chat** with an assistant grounded in the live twin.
- Get a **one-click clinical brief**.

## 6. Technical stack

| Layer | Technology |
|---|---|
| Language | Python 3.10+ |
| Numerics / modelling | NumPy, SciPy, pandas |
| Backend API | FastAPI, Uvicorn, Pydantic |
| Time-series foundation model | Amazon **Chronos-Bolt** (`chronos-forecasting`, PyTorch) |
| PPG foundation model | **PaPaGei-S** (Nokia Bell Labs), PyTorch |
| LLM | **DeepInfra** (OpenAI-compatible API), `Meta-Llama-3.1-8B-Instruct`. Can be swapped for local Ollama / llama.cpp for zero data egress |
| Frontend | React 18, Vite 5, Recharts 2 |
| Config | python-dotenv |
| Data | Synthetic FHIR-style EHR + synthetic wearable streams (DPDP / HIPAA sandbox compliant) |

## 7. AI/ML model and framework details

No model was trained from scratch. Every component is a published equation, open pretrained weights, or an LLM.

| Expert | What it does | Source |
|---|---|---|
| **Physiology twin** | Bergman minimal model extended with gut absorption, insulin secretion, exercise, liver suppression and dawn effect. Parameters come from the patient's EHR (HOMA-IR, HOMA-B, metformin / sulfonylurea effects, TCF7L2 genotype). Sleep and HRV modulate insulin sensitivity live. | Bergman 1979; Dalla Man 2007; Matthews 1985 |
| **Data assimilation** | Every 2 h, re-fits insulin sensitivity and basal glucose to recent CGM, giving the twin divergence alert | Weather-model style state estimation |
| **Chronos-Bolt** | Zero-shot glucose forecasting from the CGM curve | Ansari et al. 2024, `amazon/chronos-bolt-small` |
| **PaPaGei-S + beat detection** | HR and HRV from raw PPG; a 512-d pulse embedding gives a nightly waveform-shape drift score | Pillai et al., ICLR 2025 |
| **Clinical risk** | Framingham 10-yr CVD risk, Indian Diabetes Risk Score, CGM consensus metrics (TIR, GMI, CV) | D'Agostino 2008; Mohan 2005; Battelino 2019 |
| **LLM router** | Reads the clinical context and sets trust weights over the experts, explains why, and is sanity-checked against measured error | Llama-3.1-8B via DeepInfra |
| **Fusion** | ½ skill weights (1/MAE² on the patient's last 8 h of hindcasts) + ½ router weights. Split-conformal 80% band gives P(>180) and P(<70) | Angelopoulos & Bates |
| **Online learning** | 192-parameter per-patient adapter, recursive least squares, on device, self-gating | |
| **Federated learning** | FedAvg over adapter weights only, with clipping + Gaussian noise (differential privacy) | McMahan 2017; Abadi 2016 |
| **Privacy gateway** | Pseudonymises names and cities, strips Aadhaar / ABHA / PAN / phone / email, bands age, and logs every outgoing payload | DPDP Act 2023 |

**Results** on a synthetic cohort (23 patients, 3,481 forecasts):

| Metric | Result |
|---|---|
| Post-meal 2-h MAE | 28 mg/dL (vs 76 for curve-only) |
| Spike prediction after a logged meal | AUROC 0.95, precision 0.98, recall 0.86, median 35 min warning |
| Hypoglycaemia | AUROC 0.97, recall 0.90 |
| Federated prior on unseen patients | MAE 16.6 → 15.8 |

Full output: `results/backtest.txt`, `results/federated.txt`.

## 8. Demo video

**Unlisted YouTube link:** https://youtu.be/j5wbkDxeSIs

## 9. Architecture diagram

📄 **[Architecture Diagram (PDF)](docs/GlucoTwin_Architecture.pdf)**

```
 EHR (labs, meds, dx, genetics)          Wearables (CGM, HR, PPG, sleep, steps, meals)
              │                                          │
              ▼                                          ▼
 ┌────────────────────────────── EXPERTS (no training) ──────────────────────────────┐
 │ Physiology twin + data assimilation │ Chronos-Bolt │ PaPaGei PPG │ Risk equations │
 └──────────────────────────────────────────────────────────────────────────────────┘
              │ forecasts + 8 h hindcasts
              ▼
 FUSION: skill weights + LLM router → online adapter → conformal band → alerts
              │
              ▼
 LLM layer (router, chat, brief, what-if parser) ◄── PRIVACY GATEWAY (de-identified only)
              │
              ▼
 FastAPI ──► React doctor dashboard (live, auto-refresh)

 Edge: all models run on device │ Federated: only 192 clipped + noised weights leave the device
```

## 10. Presentation

📄 **[Project Presentation (PDF/PPT)](docs/GlucoTwin_Presentation.pdf)**

## 11. How to run

```bash
pip install -r requirements.txt --break-system-packages
bash scripts/get_papagei.sh          # PaPaGei weights (23 MB)
cp .env.example .env                 # add DEEPINFRA_API_KEY
uvicorn server:app --port 8000       # open http://localhost:8000
```

**Optional:**
- Fully offline LLM: `LLM_BASE_URL=http://localhost:11434/v1` (Ollama).
- Re-run the evaluation: `python scripts/backtest.py` and `python scripts/federated_demo.py`.

## 12. Repository structure

```
server.py              FastAPI backend + replay clock
twin/                  physiology, EHR + wearable generators, engine, LLM, chat, privacy, adaptive learning
twin/experts/          mechanistic, foundation (Chronos), PPG (PaPaGei), autonomic, risk
scripts/               backtest, federated simulation, weight download
web/                   React + Recharts dashboard
results/               evaluation outputs
tools/demo_video/      demo video pipeline
```

## 13. Data and sandbox compliance

All patient data is **synthetic**, as required by the challenge rules (DPDP Act / HIPAA). This covers EHRs, CGM, heart rate, PPG, sleep and meal logs. Ground truth comes from a physiology simulator with hidden per-patient parameters and hidden events. The models use real open pretrained weights, zero-shot.

## 14. Open-source license details

This project is released under the **MIT License** (see `LICENSE`).

| Third-party component | License |
|---|---|
| Chronos / Chronos-Bolt (Amazon) | Apache-2.0 |
| PaPaGei (Nokia Bell Labs), vendored architecture in `twin/experts/papagei_net.py` | BSD-3-Clause-Clear |
| Llama 3.1 | Llama 3.1 Community License |
| React, Vite, Recharts, FastAPI | MIT |
| NumPy, SciPy, pandas, PyTorch | BSD-3-Clause |

## 15. Accessibility

All files and links in this repository are publicly accessible without additional permissions.
