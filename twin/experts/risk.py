"""Expert 4 - Published clinical risk equations (static EHR stream) and CGM metrics.

* Framingham general CVD 10-yr risk (D'Agostino et al. 2008, Circulation) - lab version
* IDRS - Indian Diabetes Risk Score (Mohan et al. 2005, JAPI) - for non-diabetics
* CGM consensus metrics: TIR/TAR/TBR (Battelino et al. 2019, Diabetes Care),
  GMI (Bergenstal et al. 2018), CV%, LBGI/HBGI (Kovatchev et al.)
"""
from __future__ import annotations

import math
import numpy as np

_FRAM = {
    "M": dict(age=3.06117, tc=1.12370, hdl=-0.93263, sbp_u=1.93303, sbp_t=1.99881,
              smk=0.65451, dm=0.57367, s0=0.88936, mean=23.9802),
    "F": dict(age=2.32888, tc=1.20904, hdl=-0.70833, sbp_u=2.76157, sbp_t=2.82263,
              smk=0.52873, dm=0.69154, s0=0.95012, mean=26.1931),
}


def framingham_cvd(ehr: dict) -> float:
    d, lab, v = ehr["demographics"], ehr["labs"], ehr["vitals"]
    c = _FRAM[d["sex"]]
    dm = any(x["code"].startswith("E11") for x in ehr["conditions"])
    s = (c["age"] * math.log(d["age"]) + c["tc"] * math.log(lab["total_cholesterol"])
         + c["hdl"] * math.log(lab["hdl"])
         + (c["sbp_t"] if v["bp_treated"] else c["sbp_u"]) * math.log(v["sbp"])
         + c["smk"] * ehr["lifestyle"]["smoker"] + c["dm"] * dm)
    return float(1 - c["s0"] ** math.exp(s - c["mean"]))


def idrs(ehr: dict) -> int:
    d = ehr["demographics"]
    a = d["age"]
    score = 0 if a < 35 else 20 if a < 50 else 30
    w = d["waist_cm"]
    if d["sex"] == "M":
        score += 0 if w < 90 else 10 if w < 100 else 20
    else:
        score += 0 if w < 80 else 10 if w < 90 else 20
    pa = ehr["lifestyle"]["physical_activity"]
    score += {"vigorous": 0, "moderate": 10, "low": 20, "sedentary": 30}.get(pa, 20)
    score += {0: 0, 1: 10, 2: 20}.get(ehr["family_history"]["diabetes_parents"], 0)
    return score


def cgm_metrics(g: np.ndarray) -> dict:
    g = np.asarray(g, dtype=float)
    f = 1.509 * (np.log(g) ** 1.084 - 5.381)
    r = 10 * f ** 2
    return {
        "mean": float(g.mean()),
        "gmi": float(3.31 + 0.02392 * g.mean()),
        "cv": float(100 * g.std() / g.mean()),
        "tir": float(((g >= 70) & (g <= 180)).mean() * 100),
        "tar": float((g > 180).mean() * 100),
        "tbr": float((g < 70).mean() * 100),
        "lbgi": float(np.mean(np.where(f < 0, r, 0))),
        "hbgi": float(np.mean(np.where(f > 0, r, 0))),
    }
