# GlucoTwin Bharat

A digital twin for **Type 2 diabetes in Indian patients**. It forecasts glucose 2 hours ahead, raises spike and hypo alerts, notices when the real patient stops matching their health record, and answers a doctor's *"what if"* questions.

No model is trained anywhere in this repo. Every part either has a published physiological equation, uses public pretrained weights, or calls a cheap LLM.

---

## The arc (why this one is different)

Most teams will train an LSTM or XGBoost model on a CGM dataset and plot a forecast line. That gives you a *predictor*. It does not give you a *twin*.

A twin has to do three things a black-box predictor cannot:

1. **Be built from the patient's record.** Our physiology core takes its parameters straight from the EHR, using published formulas: HOMA-IR, HOMA-B, medication effects, and TCF7L2 genotype.
2. **Notice when the patient stops matching that record.** Every 2 hours the twin re-fits two latent states (insulin sensitivity and basal glucose) to the CGM, much like weather data assimilation. When they drift, it raises a **twin divergence** alert. In the demo, Rajesh catches a viral fever that is not in his EHR, and the twin flags it from glucose alone, with no fever data.
3. **Answer counterfactuals.** "What if he had millet roti instead of the biryani he just ate, and walked 20 minutes?" The twin rewinds to before the meal and replays both plans. A statistical model can't do this, because it has never seen the patient eat millet.

Around that physiology core sits a **mixture of experts with an LLM router**, the same pattern as our supply-chain agent, where an LLM replaced the statistical forecaster:

```
          STATIC STREAM (EHR, FHIR-lite)               DYNAMIC STREAM (wearables)
   labs · meds · genotype · history · vitals    CGM · HR · steps · sleep stages · HRV · meal log
                    │                                          │
                    ▼                                          ▼
   ┌───────────────────────────────── experts (no training) ─────────────────────────────────┐
   │ 1 Physiology twin     Bergman minimal model + gut absorption + exercise + dawn effect,  │
   │                       params from HOMA/meds/TCF7L2; sleep & HRV modulate insulin        │
   │                       sensitivity live; data assimilation every 2 h -> drift signal     │
   │ 2 Foundation model    Amazon Chronos-Bolt, pretrained weights, zero-shot on CGM          │
   │ 3 LLM forecaster      DeepInfra Llama-3.1-8B reads CGM + meals + sleep as text (opt.)   │
   │ 4 Autonomic expert    personal-baseline z-scores on HRV / resting HR / sleep -> CV flag │
   │ 5 Risk equations      Framingham CVD, Indian Diabetes Risk Score, CGM consensus metrics │
   │ 6 PPG expert          raw overnight pulse -> HR/HRV (beat detection) + PaPaGei-S        │
   │                       (Nokia Bell Labs, ICLR 2025, open weights) waveform-shape drift   │
   └─────────────────────────────────────────────────────────────────────────────────────────┘
                    │ forecasts + 8 h of self-graded hindcasts
                    ▼
   FUSION   w = ½ · skill (1/MAE² on the last 8 h, for this patient)  +  ½ · LLM router
            split-conformal 80% band  ->  P(spike > 180), P(hypo < 70) within 2 h
                    │
                    ▼
   LLM      clinical brief for the doctor · parses natural-language what-ifs into simulator plans
                    │
                    ▼
   React dashboard (live replay, auto-refresh every 2 s)
```

### Edge-first, privacy by design

```
 ┌──────────────────────── ON THE DEVICE (phone / clinic laptop) ────────────────────────┐
 │ physiology twin · Chronos-Bolt · PaPaGei · risk scores · what-if simulator           │
 │ online adapter (learns this patient) · alerts · raw CGM / PPG / health record        │
 │                                                                                       │
 │        PRIVACY GATEWAY  names/IDs -> pseudonyms · Aadhaar/ABHA/PAN/phone/email        │
 │                         scrubbed · age -> 5-yr band · every payload audit-logged      │
 └──────────────┬──────────────────────────────────────────────┬────────────────────────┘
                │ de-identified text only                       │ 192 adapter weights only
                ▼                                               ▼ (clipped + DP noise)
     LLM (DeepInfra, or local Ollama = zero egress)     federated server: FedAvg -> global prior
```

- **Every model runs on the device.** Chronos-Bolt (9M–48M params) and PaPaGei-S (5.8M) run on a laptop CPU; the physiology twin is plain numpy.
- **The LLM is the only thing that can be remote.** Every call (router, brief, chat, what-if parser, forecaster) goes through `twin/privacy.py`. The model sees `Patient-ED61` from `City-0ABD`, never a name, Aadhaar, ABHA ID, phone or email, and the reply is re-identified locally. The Privacy & edge tab shows the exact text that left the machine. Set `LLM_BASE_URL` to Ollama or llama.cpp and nothing leaves at all.
- **Learning without sharing data.** A 192-weight adapter on top of the frozen models learns each patient's own forecast error online (recursive least squares, predict-then-update, self-gating). Devices share only those weights; the server runs FedAvg with clipping and Gaussian noise (differential privacy). New patients start from the federated prior. See `results/federated.txt`.
- **Patient-aware chat.** The Chat tab answers from the live twin: record, glucose, forecast, alerts, sleep, PPG, meals. If the question is a what-if, the simulator runs first and the LLM explains the numbers.

### What is live today vs roadmap

| Live in this repo | Next step (same design, bigger layer) |
|---|---|
| Online RLS adapter per patient, on-device | LoRA adapters on Chronos-Bolt / PaPaGei, still frozen base weights |
| FedAvg + clipping + Gaussian noise over adapter weights (simulated 15 devices) | Real devices via Flower; formal (ε, δ) accounting with a privacy accountant |
| Privacy gateway + egress audit log | On-device NER for free-text clinical notes |
| Local LLM via any OpenAI-compatible server | Quantised Chronos / PaPaGei (ONNX, int8) on Android / wearable SoC |

**Why the fusion layer matters:** each expert is good at different moments. Right after a meal is logged, only the physiology twin knows 90 g of carbs is still being absorbed. Overnight, the foundation model reads the curve better. If the patient is ill, the physiology parameters are stale. The router reads that context, and the skill weights grade every expert against the last 8 hours of this patient's own data. A bad expert loses weight automatically, so the LLM forecaster is safe to include.

**Indian context, built in:** a South Asian "thin-fat" phenotype (Arjun has a normal BMI but a 94 cm waist), an Indian meal library with GI values (poha, dosa, rajma chawal, biryani, millet roti…), vrat fasting on a sulfonylurea, late dinners, IDRS, and TCF7L2 (the strongest T2D GWAS signal, replicated in Indians).

---

## Demo patients

| Patient | Story | What the twin shows |
|---|---|---|
| **Rajesh Kumar**, 52M, Delhi, T2D, metformin, smoker | Two short nights, then a 22:30 biryani. Catches a viral fever on day 5 that isn't in his record. | Spike alert ~20 min after logging the meal. *Twin divergence* alert as his basal glucose climbs about 20–40 mg/dL above his record. What-if: millet roti plus a walk cuts the peak by about 80 mg/dL. |
| **Lakshmi Iyer**, 63F, Chennai, T2D, metformin + glimepiride | Keeps a vrat fast on day 4, skips lunch, walks to the temple. | Hypoglycaemia alert before she drops below 70, pointing at the sulfonylurea. |
| **Arjun Mehta**, 44M, Bengaluru, prediabetes, normal BMI | Release week at work: sleep falls to 4.5 h, HRV halves, resting HR climbs. | *Sustained autonomic strain* flag (cardiovascular), plus post-dosa spikes. |

Use **Jump to** in the left rail to go straight to each moment, or press **Play week** to watch it live.

---

## Run it (Mac)

```bash
# backend
pip install -r requirements.txt --break-system-packages
cp .env.example .env            # add DEEPINFRA_API_KEY (optional)
uvicorn server:app --port 8000

# frontend (second terminal)
cd web
npm install
npm run dev                      # http://localhost:5173
```

PaPaGei PPG weights (23 MB, Zenodo): `bash scripts/get_papagei.sh`. Without them, HR and HRV are still computed from the raw pulse; only the waveform-shape drift is skipped.

On first start, Chronos-Bolt downloads its weights (~190 MB for `small`) from Hugging Face. To skip torch entirely, set `DISABLE_CHRONOS=1` and a damped-trend forecaster takes its place.

To serve everything from one port, run `npm run build` in `web/` and open `http://localhost:8000`.

Backtest (writes `results/backtest.txt` and `results/backtest_rows.csv`):

```bash
python scripts/backtest.py --cohort 20          # add --llm to include the LLM forecaster
```

**LLM cost:** Llama-3.1-8B on DeepInfra is about $0.02 / $0.05 per million tokens. A full demo session (router calls, briefs, what-ifs) costs well under one cent. The footer shows the running cost.

---

## Repo

```
server.py                 FastAPI: replay clock, /api/state, /api/whatif, /api/brief, /api/chat,
                          /api/privacy, /api/learning
twin/privacy.py           privacy gateway: pseudonymise, scrub Indian IDs, audit every LLM payload
twin/adaptive.py          online RLS adapter + FedAvg with clipping / DP noise
twin/chat.py              patient-aware chatbot grounded in the live twin
scripts/federated_demo.py 15-device federated simulation -> results/federated.txt + global prior
twin/physiology.py        mechanistic glucose-insulin model, EHR -> parameters
twin/ehr.py               synthetic FHIR-lite EHRs (3 personas + random cohort)
twin/wearables.py         synthetic CGM / HR / steps / sleep / HRV + hidden ground truth
twin/meals.py             Indian meal library (carbs, GI)
twin/experts/             mechanistic (+assimilation, what-if), foundation (Chronos),
                          autonomic, risk equations, ppg (PaPaGei), papagei_net (vendored, BSD-3-Clear)
twin/engine.py            skill + router fusion, conformal band, alerts
twin/llm.py               DeepInfra router, LLM forecaster, clinical brief, what-if parser
scripts/backtest.py       evaluation on personas + cohort
web/                      React + Recharts dashboard (Vite)
```

## Sandbox rules

All data is synthetic, generated in `twin/ehr.py` and `twin/wearables.py`. Ground truth comes from running the physiology model with **hidden** per-patient parameters and hidden events, such as an illness missing from the EHR and ±22% carb-logging error. The twin never sees these. It has to infer them from the data it is given. The generators can be swapped for Synthea / MIMIC-IV EHRs and OhioT1DM / Shanghai T2DM CGM without touching the engine.

## References

- Bergman RN et al. Quantitative estimation of insulin sensitivity. *Am J Physiol* 1979. (minimal model)
- Dalla Man C, Rizza RA, Cobelli C. Meal simulation model of the glucose-insulin system. *IEEE TBME* 2007.
- Matthews DR et al. Homeostasis model assessment (HOMA). *Diabetologia* 1985.
- Ansari AF et al. Chronos: Learning the language of time series. *TMLR* 2024. (weights: `amazon/chronos-bolt-small`)
- GlucoFM-Bench: Benchmarking time-series foundation models for blood glucose forecasting, 2026 (zero-shot TSFMs within ~5% of full-shot supervised).
- Pillai A et al. PaPaGei: Open foundation models for optical physiological signals. *ICLR* 2025. (weights: zenodo.org/records/13983110)
- McMahan B et al. Communication-efficient learning of deep networks from decentralized data (FedAvg). *AISTATS* 2017.
- Abadi M et al. Deep learning with differential privacy. *CCS* 2016.
- Haykin S. *Adaptive Filter Theory* (recursive least squares).
- Hu E et al. LoRA: Low-rank adaptation of large language models. *ICLR* 2022.
- Gruver N et al. Large language models are zero-shot time series forecasters. *NeurIPS* 2023.
- Angelopoulos AN, Bates S. A gentle introduction to conformal prediction. 2021.
- Shamanna P et al. Digital twin in managing hypertension among people with T2D: 1-year RCT (India). *JACC Advances* 2024; and Personalized nutrition in T2D remission using digital twin technology. *Front Endocrinol* 2024.
- Donga E et al. A single night of partial sleep deprivation induces insulin resistance. *JCEM* 2010.
- Hillebrand S et al. Heart rate variability and first cardiovascular event. *Europace* 2013.
- Grant SFA et al. Variant of TCF7L2 confers risk of type 2 diabetes. *Nat Genet* 2006.
- D'Agostino RB et al. General cardiovascular risk profile (Framingham). *Circulation* 2008.
- Mohan V et al. Indian Diabetes Risk Score. *JAPI* 2005.
- Battelino T et al. International consensus on time in range. *Diabetes Care* 2019; Bergenstal RM et al. Glucose Management Indicator. *Diabetes Care* 2018.
- Kovatchev BP et al. LBGI / HBGI risk indices.

## Results (23 synthetic patients, 3,481 forecasts, zero training)

Run with the damped-trend fallback in place of Chronos. Expect the foundation row to improve once the real Chronos weights are loaded.

| 2-h glucose forecast, MAE mg/dL | @30 min | @60 min | @120 min |
|---|---|---|---|
| Persistence | 18.4 | 31.0 | 51.3 |
| Curve-only forecaster | 13.9 | 25.9 | 45.9 |
| Physiology twin | 15.7 | 19.0 | 33.4 |
| **Fused twin** | **11.6** | **17.6** | 33.6 |

- **Right after a meal is logged:** fused MAE is 28 mg/dL, against 76 for the curve-only model.
- **Spike > 180 within 2 h of a logged meal:** AUROC 0.95, precision 0.98, recall 0.86, with a median 35 min warning.
- **Hypo < 70 within 2 h:** AUROC 0.97, recall 0.90. The threshold is deliberately set to over-alert.
- **Twin divergence:** Rajesh's hidden illness shifts his inferred basal glucose from −2 mg/dL to +30 mg/dL, with no other alert across the cohort's personas.

At any random hour, spike AUROC is 0.62. That is expected: nothing can predict a meal that hasn't been eaten or logged yet. This is why the twin pairs forecasting with what-if planning.

Full output is in `results/backtest.txt`.
