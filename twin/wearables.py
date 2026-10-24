"""Synthetic wearable / IoT stream (dynamic / real-time).

Generates 7 days of Apple-Health / Google-Fit / Libre-style data per patient:
  CGM (5-min, with interstitial lag + sensor noise), heart rate, steps,
  sleep stages, nightly RMSSD (HRV), resting HR, plus the patient's meal log.

Ground truth comes from running the mechanistic physiology with HIDDEN
per-patient parameters and hidden events (e.g. an illness that is not in the EHR),
so the twin has to discover them from data - that is the point of a digital twin.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .meals import MEALS
from .physiology import params_from_ehr, initial_state, simulate

DAYS = 7
MIN_PER_DAY = 1440
START = pd.Timestamp("2026-10-12 00:00")   # Monday


def _hm(h, m=0):
    return h * 60 + m


def _plan(pid: str, rng: np.random.Generator):
    """Per-day plan: meals [(minute_of_day, key, scale)], walks [(start, dur, intensity)],
    sleep (bed_minute_of_day, hours), rmssd baseline, rhr baseline, illness flag."""
    days = []
    for d in range(DAYS):
        j = lambda: int(rng.integers(-20, 21))
        if pid == "P001":
            meals = [(_hm(8, 40) + j(), rng.choice(["aloo_paratha", "poha"]), 1.0),
                     (_hm(13, 45) + j(), "chole_bhature" if d == 4 else "rice_dal", 1.0),
                     (_hm(17, 15) + j(), "chai_biscuit", 1.0),
                     (_hm(22, 30) if d == 3 else _hm(21, 30) + j(), "biryani" if d == 3 else "roti_sabzi", 1.0)]
            walks = [(_hm(9, 30), 15, 0.4), (_hm(19, 0), 15, 0.4)]
            sleep_h = 4.5 if d in (3, 4) else rng.uniform(6.2, 7.0)
            days.append(dict(meals=meals, walks=walks, bed=_hm(23, 30), sleep_h=sleep_h,
                             rmssd=28, rhr=72, ill=d >= 5))
        elif pid == "P002":
            fast = d == 3
            meals = [(_hm(8, 0) + j(), "none" if fast else rng.choice(["idli_sambar", "upma"]), 1.0),
                     (_hm(13, 0) + j(), "none" if fast else "rice_dal", 0.9),
                     (_hm(16, 45) + j(), "fruit", 1.0),
                     (_hm(20, 0) + j(), "roti_sabzi", 0.9)]
            walks = [(_hm(6, 45), 30, 0.8)] + ([(_hm(10, 30), 50, 0.9)] if fast else [])
            days.append(dict(meals=meals, walks=walks, bed=_hm(22, 15), sleep_h=rng.uniform(6.5, 7.5),
                             rmssd=24, rhr=70, ill=False))
        elif pid == "P003":
            crunch = 2 <= d <= 5
            meals = [(_hm(9, 15) + j(), rng.choice(["masala_dosa", "idli_sambar"]), 1.0),
                     (_hm(11, 30) + j(), "chai_biscuit", 1.0),
                     (_hm(14, 0) + j(), rng.choice(["rice_dal", "rajma_chawal"]), 1.0),
                     (_hm(17, 30) + j(), "samosa" if crunch else "chai_biscuit", 1.0),
                     (_hm(22, 0) + j(), "roti_sabzi", 1.0)]
            walks = [(_hm(18, 30), 10, 0.3)]
            sleep_h = (7.2 - 0.7 * (d - 1)) if crunch else 7.2
            days.append(dict(meals=meals, walks=walks, bed=_hm(1, 0) if crunch else _hm(23, 45),
                             sleep_h=max(sleep_h, 4.3), rmssd=44 - 5.5 * d, rhr=63 + 2.0 * d, ill=False))
        else:  # random cohort
            bk = rng.choice(["poha", "idli_sambar", "aloo_paratha", "upma", "masala_dosa", "besan_chilla", "eggs_toast"])
            ln = rng.choice(["rice_dal", "roti_sabzi", "rajma_chawal", "biryani", "millet_roti", "chole_bhature"])
            dn = rng.choice(["roti_sabzi", "rice_dal", "biryani", "millet_roti", "salad_paneer"])
            sn = rng.choice(["chai_biscuit", "samosa", "fruit", "sweets", "roasted_chana", "none"])
            meals = [(_hm(8, 30) + j() * 2, bk, 1.0), (_hm(13, 30) + j() * 2, ln, rng.uniform(0.8, 1.2)),
                     (_hm(17, 0) + j() * 2, sn, 1.0), (_hm(21, 0) + j() * 3, dn, rng.uniform(0.8, 1.2))]
            if rng.random() < 0.12:
                meals[1] = (meals[1][0], "none", 1.0)
            walks = [(_hm(int(rng.integers(6, 20)), 0), int(rng.integers(10, 45)), float(rng.uniform(0.3, 1.0)))] \
                if rng.random() < 0.6 else []
            days.append(dict(meals=meals, walks=walks, bed=_hm(23, int(rng.integers(0, 59))),
                             sleep_h=float(rng.uniform(4.5, 8.0)), rmssd=float(rng.uniform(20, 45)),
                             rhr=float(rng.uniform(60, 78)), ill=bool(rng.random() < 0.08)))
    return days


def si_multiplier_from_wearables(sleep_h_prev: float, rmssd_z: float) -> float:
    """Wearable -> physiology coupling (no training, literature-derived):
    * one night of 4 h sleep lowers insulin sensitivity ~20% (Donga et al. 2010, JCEM)
    * low vagal tone / high stress lowers insulin sensitivity (sympathetic activation)."""
    m = 1.0 - 0.07 * max(0.0, 7.0 - sleep_h_prev)
    if rmssd_z < -1.0:
        m *= 1.0 - 0.05 * min(-rmssd_z - 1.0, 2.0)
    return float(np.clip(m, 0.7, 1.05))


def simulate_week(ehr: dict, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed + 1000)
    pid = ehr["id"] if ehr["id"] in ("P001", "P002", "P003") else "RANDOM"
    plan = _plan(pid, rng)
    n = DAYS * MIN_PER_DAY

    # ---- hidden truth parameters (the "real" patient) ----
    p_true = params_from_ehr(ehr)
    p_true.SI *= float(np.exp(rng.normal(0, 0.15)))
    p_true.gamma *= float(np.exp(rng.normal(0, 0.10)))
    gb_true = np.full(n, p_true.Gb)

    activity = np.zeros(n)
    asleep = np.zeros(n, dtype=bool)
    si_true = np.ones(n)
    si_obs = np.ones(n)
    su_on = np.zeros(n)
    meals_true, meal_log, nightly = [], [], []
    rmssd_hist = []

    for d, day in enumerate(plan):
        base = d * MIN_PER_DAY
        # walks + incidental activity
        for st, dur, inten in day["walks"]:
            activity[base + st: base + st + dur] = np.maximum(activity[base + st: base + st + dur], inten)
        awake_idx = np.arange(base + 7 * 60, base + 22 * 60)
        bouts = rng.choice(awake_idx, size=12, replace=False)
        for b in bouts:
            L = int(rng.integers(3, 10))
            activity[b: b + L] = np.maximum(activity[b: b + L], rng.uniform(0.1, 0.35))
        # sleep window starting at bed time of this day
        bed_abs = base + day["bed"] + (MIN_PER_DAY if day["bed"] < 12 * 60 else 0)
        dur = int(day["sleep_h"] * 60)
        asleep[max(bed_abs, 0): min(bed_abs + dur, n)] = True
        # previous night's sleep for this day's SI
        prev_sleep = plan[d - 1]["sleep_h"] if d > 0 else 7.0
        rmssd = max(8.0, day["rmssd"] * (0.72 if day["ill"] else 1.0) + rng.normal(0, 2.0))
        rhr = day["rhr"] + (8 if day["ill"] else 0) + rng.normal(0, 1.0)
        rmssd_hist.append(rmssd)
        base_rm = np.mean(rmssd_hist[:3]) if len(rmssd_hist) >= 3 else np.mean(rmssd_hist)
        sd_rm = max(np.std(rmssd_hist[:3]) if len(rmssd_hist) >= 3 else 3.0, 3.0)
        z = (rmssd - base_rm) / sd_rm
        m_obs = si_multiplier_from_wearables(prev_sleep, z)
        si_obs[base: base + MIN_PER_DAY] = m_obs
        si_true[base: base + MIN_PER_DAY] = m_obs * (0.68 if day["ill"] else 1.0)
        if day["ill"]:   # stress hyperglycaemia: counter-regulatory hormones raise hepatic output
            gb_true[base: base + MIN_PER_DAY] += 30.0
        # sulfonylurea taken with breakfast, active ~14h
        if p_true.su_rate > 0:
            su_on[base + 8 * 60: base + 22 * 60] = 1.0
            su_on[base + 22 * 60: base + MIN_PER_DAY] = 0.3
            su_on[base: base + 8 * 60] = 0.3
        for mt, key, scale in day["meals"]:
            if key == "none":
                continue
            m = MEALS[key]
            carbs = m["carbs"] * scale
            t_abs = base + int(mt)
            meals_true.append((t_abs, carbs, m["gi"]))
            # patient logs meal (photo -> carb estimate) with ~25% error
            meal_log.append({"t": t_abs, "key": key, "label": m["label"], "gi": m["gi"],
                             "carbs_logged": round(carbs * float(np.exp(rng.normal(0, 0.22)))),
                             "carbs_true": round(carbs)})
        stages_frac = {"deep": float(np.clip(0.2 - 0.02 * max(0, 7 - day["sleep_h"]) + rng.normal(0, 0.02), 0.05, 0.3)),
                       "rem": float(np.clip(0.22 + rng.normal(0, 0.02), 0.1, 0.3))}
        nightly.append({"day": d, "date": (START + pd.Timedelta(days=d)).date().isoformat(),
                        "sleep_hours": round(day["sleep_h"], 2), "deep_frac": round(stages_frac["deep"], 3),
                        "rem_frac": round(stages_frac["rem"], 3), "rmssd_ms": round(rmssd, 1),
                        "resting_hr": round(rhr, 1), "ill_hidden": day["ill"],
                        "steps": 0})

    activity[asleep] = 0.0
    if p_true.su_rate == 0:
        su_on[:] = 1.0  # irrelevant

    # ---- ground-truth physiology run ----
    s0 = initial_state(p_true)
    G_true, _ = simulate(p_true, s0, 0, n, meals_true, activity, si_true, su_on, gb_t=gb_true)

    # ---- sensors ----
    cgm_ist = np.empty(n)
    acc = G_true[0]
    for k in range(n):          # interstitial lag (tau ~ 8 min)
        acc += (G_true[k] - acc) / 8.0
        cgm_ist[k] = acc
    idx5 = np.arange(0, n, 5)
    cgm = cgm_ist[idx5] * (1 + rng.normal(0, 0.03, len(idx5))) + rng.normal(0, 3.0, len(idx5))
    cgm = np.clip(cgm, 40, 400)

    rhr_min = np.repeat([nt["resting_hr"] for nt in nightly], MIN_PER_DAY)
    hr = rhr_min + 48 * activity + rng.normal(0, 2.5, n) + 6 * (~asleep)
    hr[asleep] -= 6
    steps_min = np.where(activity > 0.05, activity * 115 + rng.normal(0, 8, n), 0).clip(0)

    ts = START + pd.to_timedelta(idx5, unit="min")
    stage = np.full(len(idx5), "awake", dtype=object)
    sl5 = asleep[idx5]
    cyc = (idx5 % 90) / 90.0
    stage[sl5 & (cyc < 0.15)] = "light"
    stage[sl5 & (cyc >= 0.15) & (cyc < 0.45)] = "deep"
    stage[sl5 & (cyc >= 0.45) & (cyc < 0.75)] = "light"
    stage[sl5 & (cyc >= 0.75)] = "rem"

    df = pd.DataFrame({
        "ts": ts, "minute": idx5,
        "cgm": cgm.round(1),
        "glucose_true": G_true[idx5].round(1),
        "hr": hr.reshape(-1, 5).mean(1).round(1),
        "steps": steps_min.reshape(-1, 5).sum(1).round(),
        "activity": activity.reshape(-1, 5).mean(1).round(3),
        "sleep_stage": stage,
    })
    for nt in nightly:
        d = nt["day"]
        nt["steps"] = int(df.loc[(df.minute >= d * MIN_PER_DAY) & (df.minute < (d + 1) * MIN_PER_DAY), "steps"].sum())

    return {
        "ehr": ehr, "df": df, "nightly": pd.DataFrame(nightly), "meal_log": meal_log,
        "activity_min": activity, "si_obs_min": si_obs, "su_on_min": su_on,
        "hidden": {"si_true_min": si_true, "p_true": p_true.to_dict()},
    }
