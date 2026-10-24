"""Mechanistic glucose-insulin twin.

Bergman minimal model (Bergman et al. 1979) extended with
  * pancreatic insulin secretion (Toffolo/Bergman "extended minimal model")
  * two-compartment gut absorption (Dalla Man et al. 2007 style, GI-dependent rate)
  * exercise-driven insulin-independent uptake
  * hepatic counter-regulation below 70 mg/dL
  * dawn phenomenon
  * time-varying insulin sensitivity multiplier driven by WEARABLE data
    (sleep loss, low HRV / stress, illness)  <- this is where the dynamic stream
    modulates the physiology parameters that came from the static EHR stream.

Every parameter is derived from EHR labs with published formulas (HOMA-IR, HOMA-B).
Nothing here is trained.

Units: G mg/dL, I uU/mL, X 1/min, time in minutes.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, field
import numpy as np

from .meals import kabs_from_gi

V_DL_PER_KG = 1.88   # glucose distribution volume (dL/kg), Dalla Man 2007
F_BIOAVAIL = 0.9


@dataclass
class TwinParams:
    Gb: float            # basal glucose (mg/dL)
    Ib: float            # basal insulin (uU/mL)
    SI: float            # insulin sensitivity (1/min per uU/mL)
    p1: float = 0.015    # glucose effectiveness (1/min)
    p2: float = 0.025    # remote insulin action decay (1/min)
    n: float = 0.13      # insulin clearance (1/min)
    gamma: float = 0.04  # beta-cell responsivity (uU/mL/min per mg/dL)
    bw: float = 70.0     # body weight (kg)
    ex_gain: float = 0.018   # insulin-independent uptake per unit activity
    su_rate: float = 0.0     # sulfonylurea glucose-independent secretion (uU/mL/min)
    dawn: float = 12.0       # dawn phenomenon amplitude (mg/dL on Gb, 4-8am)
    notes: list = field(default_factory=list)

    def to_dict(self):
        return asdict(self)


def params_from_ehr(ehr: dict) -> TwinParams:
    """Personalise the twin from the static EHR stream."""
    labs = ehr["labs"]
    fpg = float(labs["fasting_glucose"])
    fi = float(labs["fasting_insulin"])
    homa_ir = fpg * fi / 405.0                       # Matthews et al. 1985
    homa_b = 360.0 * fi / max(fpg - 63.0, 5.0)        # Matthews et al. 1985
    si = 5.0e-4 / max(homa_ir, 0.5)
    gamma = float(np.clip(0.05 * homa_b / 100.0, 0.008, 0.08))
    p = TwinParams(Gb=fpg, Ib=fi, SI=si, gamma=gamma, bw=float(ehr["demographics"]["weight_kg"]))
    p.notes.append(f"HOMA-IR {homa_ir:.2f} -> SI {si:.2e}")
    p.notes.append(f"HOMA-B {homa_b:.0f}% -> beta-cell gain {gamma:.4f}")

    tcf = ehr.get("genetics", {}).get("TCF7L2_rs7903146", "CC")
    risk_alleles = tcf.count("T")
    if risk_alleles:
        p.gamma *= 1.0 - 0.1 * risk_alleles
        p.notes.append(f"TCF7L2 rs7903146 {tcf}: beta-cell secretion -{10 * risk_alleles}% (Grant et al. 2006)")

    meds = {m["name"].lower() for m in ehr.get("medications", [])}
    if "metformin" in meds:
        p.SI *= 1.2
        p.Gb *= 0.93
        p.notes.append("Metformin: hepatic output -7%, SI +20%")
    if any(m in meds for m in ("glimepiride", "gliclazide", "glibenclamide")):
        p.su_rate = 2.5
        p.notes.append("Sulfonylurea: glucose-independent insulin secretion (hypo risk)")
    if "insulin glargine" in meds:
        p.Ib += 6.0
        p.notes.append("Basal insulin: Ib +6 uU/mL")
    if ehr["demographics"].get("south_asian", True):
        p.notes.append("South Asian phenotype: higher insulin resistance at lower BMI (accounted via HOMA)")
    return p


def initial_state(p: TwinParams) -> np.ndarray:
    # [G, X, I, Q1, Q2]
    return np.array([p.Gb, 0.0, p.Ib + p.su_rate / p.n, 0.0, 0.0], dtype=float)


def simulate(
    p: TwinParams,
    state0: np.ndarray,
    t0: int,
    n_min: int,
    meals: list[tuple[int, float, float]] | None = None,
    activity: np.ndarray | None = None,
    si_mult: np.ndarray | float = 1.0,
    su_on: np.ndarray | None = None,
    gb_t: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Euler-integrate n_min minutes starting at absolute minute t0.

    meals: list of (absolute_minute, carbs_g, gi)
    activity: per-minute activity intensity (0..~1.5), length n_min
    si_mult: scalar or per-minute multiplier on SI (wearable-driven)
    su_on: per-minute 0/1 whether sulfonylurea is active (default: on)
    gb_t: optional per-minute basal glucose (used by the ground-truth simulator for illness)
    Returns (G per minute [n_min], final state).
    """
    G, X, I, Q1, Q2 = state0.astype(float)
    meals = meals or []
    meal_at: dict[int, list[tuple[float, float]]] = {}
    for tm, carbs, gi in meals:
        if t0 <= tm < t0 + n_min:
            meal_at.setdefault(int(tm), []).append((carbs * 1000.0, kabs_from_gi(gi)))
    # gut absorption uses one effective rate per compartment content; track weighted rate
    kabs_eff = 0.03
    act = activity if activity is not None else np.zeros(n_min)
    sim = np.broadcast_to(np.asarray(si_mult, dtype=float), (n_min,)) if np.ndim(si_mult) == 0 else si_mult
    su = su_on if su_on is not None else np.ones(n_min)
    out = np.empty(n_min)
    denom = p.bw * V_DL_PER_KG
    for k in range(n_min):
        t = t0 + k
        if t in meal_at:
            for mg, ka in meal_at[t]:
                tot = Q1 + Q2 + mg
                kabs_eff = (kabs_eff * (Q1 + Q2) + ka * mg) / max(tot, 1.0)
                Q1 += mg
        hod = (t / 60.0) % 24.0
        dawn = p.dawn if 4.0 <= hod < 8.0 else 0.0
        Gb_t = (gb_t[k] if gb_t is not None else p.Gb) + dawn
        Ra = F_BIOAVAIL * kabs_eff * Q2 / denom
        counter = 0.04 * max(70.0 - G, 0.0)
        a = act[k] if k < len(act) else 0.0
        # insulin suppresses hepatic glucose output (up to 70% of basal EGP)
        egp_supp = 0.7 * p.p1 * Gb_t * min(max(I - p.Ib, 0.0) / 40.0, 1.0) * min(sim[k], 1.3) ** 2
        dG = -(p.p1 + X) * G + p.p1 * Gb_t - egp_supp + Ra - p.ex_gain * a * G + counter
        dX = -p.p2 * X + p.p2 * p.SI * sim[k] * (1.0 + 1.2 * a) * (I - p.Ib)
        dI = -p.n * (I - p.Ib) + p.gamma * max(G - p.Gb, 0.0) + p.su_rate * su[k]
        dQ1 = -kabs_eff * Q1
        dQ2 = kabs_eff * Q1 - kabs_eff * Q2
        G += dG; X += dX; I += dI; Q1 += dQ1; Q2 += dQ2
        G = max(G, 30.0)
        out[k] = G
    return out, np.array([G, X, I, Q1, Q2])
