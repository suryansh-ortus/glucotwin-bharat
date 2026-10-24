"""Backtest the twin on the 3 demo personas + a synthetic cohort. No training anywhere.

For every patient, every hour from day 1 10:00 to the end of the week:
  issue a 2-h forecast, then compare with what the CGM actually did.
Reports forecast error per expert vs the fused twin, and adverse-event detection
(new glucose spike > 180 within 2 h; hypoglycaemia < 70 within 2 h).

Usage:  python scripts/backtest.py [--cohort 20] [--llm]
Output: results/backtest.txt and results/backtest_rows.csv
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from twin.ehr import PERSONAS, random_patient  # noqa: E402
from twin.engine import H, TwinEngine  # noqa: E402
from twin.experts.foundation import FoundationExpert  # noqa: E402
from twin.llm import LLM  # noqa: E402
from twin.wearables import simulate_week  # noqa: E402


def auroc(y, p):
    y, p = np.asarray(y), np.asarray(p)
    pos, neg = p[y == 1], p[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    ranks = pd.Series(np.r_[pos, neg]).rank().to_numpy()
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", type=int, default=20)
    ap.add_argument("--llm", action="store_true", help="include the LLM forecaster expert (costs API calls)")
    args = ap.parse_args()

    fm = FoundationExpert()
    llm = LLM() if args.llm else None
    patients = list(PERSONAS.values()) + [random_patient(s) for s in range(args.cohort)]
    rows = []
    t0 = time.time()
    for ehr in patients:
        b = simulate_week(ehr, seed=1)
        eng = TwinEngine(b, llm=llm or _NoLLM(), foundation=fm, use_llm_forecaster=args.llm)
        cgm = eng.cgm
        meal_origins = {(m["t"] + 5) // 5 * 5 for m in b["meal_log"]}
        hourly = set(range(1440 + 600, 7 * 1440 - 125, 60))
        for o in sorted(hourly | {o for o in meal_origins if 1440 + 600 <= o < 7 * 1440 - 125}):
            s = eng.snapshot(o)
            act = eng._actual(o)
            cur = cgm[o // 5]
            row = {"patient": ehr["id"], "origin": o, "current": cur, "act_max": act.max(), "act_min": act.min(),
                   "p_spike": s["p_spike"], "p_hypo": s["p_hypo"], "theta": s["theta"], "dgb": s["dgb"],
                   "alert_spike": any(a["type"] == "spike" for a in s["alerts"]),
                   "alert_hypo": any(a["type"] == "hypo" for a in s["alerts"]),
                   "post_meal": o in meal_origins, "hourly": o in hourly}
            fc = {"fused": s["forecast"]["raw_median"], "fused+online": s["forecast"]["median"], **s["forecast"]["experts"], "persistence": np.full(H, cur)}
            for name, f in fc.items():
                err = np.abs(f - act)
                row[f"{name}_mae30"] = err[5]
                row[f"{name}_mae60"] = err[11]
                row[f"{name}_mae120"] = err[23]
                row[f"{name}_mae"] = err.mean()
            cross = np.where(act > 180)[0]
            row["lead_min"] = (cross[0] + 1) * 5 if len(cross) else np.nan
            rows.append(row)
        print(f"{ehr['id']} done ({time.time() - t0:.0f}s)")

    df = pd.DataFrame(rows)
    os.makedirs("results", exist_ok=True)
    df.to_csv("results/backtest_rows.csv", index=False)

    lines = []
    P = lines.append
    P("GlucoTwin-Bharat backtest (zero training)")
    P(f"patients: {len(patients)}  forecast origins: {len(df)}  horizon: 2 h (5-min steps)")
    P(f"foundation backend: {fm.backend}" + (f"  [{fm.error}]" if fm.error else ""))
    P(f"LLM forecaster: {'on, ' + llm.model if args.llm else 'off'}")
    P("")
    P("Forecast error (MAE, mg/dL), hourly origins")
    hdf = df[df.hourly]
    P(f"{'model':<14}{'@30min':>9}{'@60min':>9}{'@120min':>10}{'mean':>9}")
    for name in ["persistence", "foundation", "mechanistic", "llm", "fused", "fused+online"]:
        if f"{name}_mae" not in hdf:
            continue
        P(f"{name:<14}{hdf[f'{name}_mae30'].mean():>9.1f}{hdf[f'{name}_mae60'].mean():>9.1f}"
          f"{hdf[f'{name}_mae120'].mean():>10.1f}{hdf[f'{name}_mae'].mean():>9.1f}")
    P("")
    P("Post-meal origins (meal logged 5 min ago) - MAE mean over 2 h")
    pm = df[df.post_meal]
    P("  " + "  ".join(f"{n}: {pm[f'{n}_mae'].mean():.1f}" for n in ["persistence", "foundation", "mechanistic", "llm", "fused"] if f"{n}_mae" in pm))
    P("")

    spike_y = ((df.current < 160) & (df.act_max > 180)).astype(int)
    hypo_y = ((df.current > 80) & (df.act_min < 70)).astype(int)
    for nm, y, sub, pcol, acol in [
            ("New spike >180 in 2 h | meal just logged", spike_y, df[(df.current < 160) & df.post_meal], "p_spike", "alert_spike"),
            ("New spike >180 in 2 h | any hour (meal may not be eaten yet)", spike_y, df[(df.current < 160) & df.hourly], "p_spike", "alert_spike"),
            ("Hypo <70 in 2 h | any hour", hypo_y, df[(df.current > 80) & df.hourly], "p_hypo", "alert_hypo")]:
        yy = y.loc[sub.index]
        tp = int(((sub[acol]) & (yy == 1)).sum()); fp = int(((sub[acol]) & (yy == 0)).sum())
        fn = int(((~sub[acol]) & (yy == 1)).sum())
        P(f"{nm}: events {int(yy.sum())} / {len(sub)} origins")
        P(f"  AUROC {auroc(yy, sub[pcol]):.3f}   alert precision {tp / max(tp + fp, 1):.2f}   recall {tp / max(tp + fn, 1):.2f}")
        if "meal just logged" in nm:
            lead = sub.loc[(sub[acol]) & (yy == 1), "lead_min"]
            if len(lead):
                P(f"  median warning lead time {lead.median():.0f} min (IQR {lead.quantile(.25):.0f}-{lead.quantile(.75):.0f})")
    P("")
    ill = df[df.patient == "P001"]
    P("Twin divergence (P001, hidden viral illness from day 5):")
    P(f"  mean basal-glucose shift days 1-4: {ill[ill.origin < 5 * 1440].dgb.mean():+.1f} mg/dL;"
      f"  day 5.5+: {ill[ill.origin >= 5.5 * 1440].dgb.mean():+.1f} mg/dL")
    P(f"runtime {time.time() - t0:.0f}s")
    txt = "\n".join(lines)
    open("results/backtest.txt", "w").write(txt + "\n")
    print(txt)


class _NoLLM:
    available = False
    model = "none"
    cost_usd = 0.0
    usage = {"calls": 0}


if __name__ == "__main__":
    main()
