"""Federated learning simulation for the GlucoTwin adapter layer.

15 "phones" (synthetic cohort patients) each run the twin on-device for a week and learn
their own 192-weight adapter online. Only adapter weights go to the server (FedAvg with
clipping, optionally Gaussian noise for differential privacy). The global adapter is then
given to 8 NEW patients the server never saw, as their starting point.

Compares on the new patients (prequential, out-of-sample MAE over 2 h, mg/dL):
  fused twin alone | + online adapter from zero | + online adapter from federated prior
  | + federated prior with DP noise

Usage: python scripts/federated_demo.py   -> results/federated.txt
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("DISABLE_PAPAGEI", "1")

from twin.adaptive import fedavg, replay  # noqa: E402
from twin.ehr import PERSONAS, random_patient  # noqa: E402
from twin.engine import TwinEngine  # noqa: E402
from twin.experts.foundation import FoundationExpert  # noqa: E402
from twin.wearables import simulate_week  # noqa: E402


class _NoLLM:
    available = False
    model = "none"
    cost_usd = 0.0
    usage = {"calls": 0}


def stream(ehr, fm):
    eng = TwinEngine(simulate_week(ehr, seed=1), llm=_NoLLM(), foundation=fm, use_llm_forecaster=False)
    for o in range(1440 + 600, 7 * 1440 - 125, 60):
        eng.snapshot(o)
    return eng.adapter.local_data


def main():
    t0 = time.time()
    fm = FoundationExpert()
    clients = [random_patient(s) for s in range(100, 115)]
    new = list(PERSONAS.values()) + [random_patient(s) for s in range(200, 205)]
    print("collecting on-device streams...")
    client_data = [stream(e, fm) for e in clients]
    new_data = [stream(e, fm) for e in new]

    lines = ["GlucoTwin federated adapter simulation (zero raw data shared)",
             f"clients: {len(clients)} devices, held-out new patients: {len(new)}",
             f"forecast backend: {fm.backend}", ""]
    rng = np.random.default_rng(0)
    results = {}
    for label, noise in [("FedAvg", 0.0), ("FedAvg + DP noise", 1.0)]:
        g = None
        for rnd in range(3):
            ws, ns = [], []
            for d in client_data:
                ad, _, _ = replay(d, g)          # local training on-device, from current global
                ws.append(ad.W); ns.append(len(d))
            g = fedavg(ws, ns, g, clip=5.0, noise_mult=noise, rng=rng)
        results[label] = g
        lines.append(f"{label}: 3 rounds, {g.size} parameters per upload, update clip 5.0, noise multiplier {noise}")

    rows = []
    for e, d in zip(new, new_data):
        _, raw, zero = replay(d, None)
        _, _, fed = replay(d, results["FedAvg"])
        _, _, dp = replay(d, results["FedAvg + DP noise"])
        first = slice(0, 24)                       # first day on the twin: cold start
        _, raw1, zero1 = replay(d[first], None)
        _, _, fed1 = replay(d[first], results["FedAvg"])
        rows.append((e["id"], raw, zero, fed, dp, raw1, zero1, fed1))
    R = np.array([r[1:] for r in rows])
    lines += ["", "Out-of-sample MAE (mg/dL, mean over 2 h) on NEW patients",
              f"{'patient':<8}{'twin':>8}{'+online':>9}{'+fed prior':>12}{'+fed DP':>9}"]
    for r in rows:
        lines.append(f"{r[0]:<8}{r[1]:>8.1f}{r[2]:>9.1f}{r[3]:>12.1f}{r[4]:>9.1f}")
    m = R.mean(0)
    lines.append(f"{'mean':<8}{m[0]:>8.1f}{m[1]:>9.1f}{m[2]:>12.1f}{m[3]:>9.1f}")
    lines += ["", "Cold start - first day only (first 24 matured forecasts)",
              f"  twin {m[4]:.1f} | online from zero {m[5]:.1f} | online from federated prior {m[6]:.1f}",
              "", f"runtime {time.time() - t0:.0f}s"]
    os.makedirs("results", exist_ok=True)
    np.save("results/federated_prior.npy", results["FedAvg + DP noise"])   # the server loads this as the global prior
    lines.append("global prior (DP version) saved to results/federated_prior.npy")
    txt = "\n".join(lines)
    open("results/federated.txt", "w").write(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()
