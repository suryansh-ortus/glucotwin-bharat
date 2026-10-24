"""Synthetic EHR stream (static / historical).

FHIR-lite JSON records shaped like Synthea output, but with an Indian cohort:
South Asian "thin-fat" phenotype, Indian meal habits, TCF7L2 genotype, family history.
Three curated demo personas + a seeded random cohort for backtesting.
No real patient data is used (DPDP Act / HIPAA sandbox rule).
"""
from __future__ import annotations

import numpy as np


def _labs(fpg, fi, a1c, tc, hdl, ldl, tg, creat, egfr):
    return {
        "fasting_glucose": fpg, "fasting_insulin": fi, "hba1c": a1c,
        "total_cholesterol": tc, "hdl": hdl, "ldl": ldl, "triglycerides": tg,
        "creatinine": creat, "egfr": egfr,
    }


PERSONAS = {
    "P001": {
        "id": "P001",
        "demographics": {"name": "Rajesh Kumar", "age": 52, "sex": "M", "city": "Delhi",
                          "weight_kg": 78, "height_cm": 168, "waist_cm": 98, "south_asian": True},
        "conditions": [
            {"code": "E11.9", "name": "Type 2 diabetes mellitus", "since": 2019},
            {"code": "I10", "name": "Essential hypertension", "since": 2021},
            {"code": "E78.5", "name": "Dyslipidaemia", "since": 2021},
        ],
        "labs": _labs(158, 15, 8.6, 212, 38, 138, 210, 1.0, 84),
        "vitals": {"sbp": 144, "dbp": 90, "bp_treated": True},
        "medications": [{"name": "Metformin", "dose": "1000 mg BD"},
                        {"name": "Telmisartan", "dose": "40 mg OD"},
                        {"name": "Atorvastatin", "dose": "10 mg OD"}],
        "lifestyle": {"smoker": True, "physical_activity": "low", "diet": "North Indian, rice + roti, late dinners"},
        "family_history": {"diabetes_parents": 2, "premature_cvd": True},
        "genetics": {"TCF7L2_rs7903146": "CT", "t2d_prs_percentile": 81},
        "story": "Late dinners, poor sleep mid-week, picks up a viral fever on day 5 (not in the EHR).",
    },
    "P002": {
        "id": "P002",
        "demographics": {"name": "Lakshmi Iyer", "age": 63, "sex": "F", "city": "Chennai",
                          "weight_kg": 62, "height_cm": 155, "waist_cm": 86, "south_asian": True},
        "conditions": [{"code": "E11.9", "name": "Type 2 diabetes mellitus", "since": 2012},
                       {"code": "I10", "name": "Essential hypertension", "since": 2016}],
        "labs": _labs(118, 12, 7.1, 188, 46, 112, 150, 1.1, 62),
        "vitals": {"sbp": 136, "dbp": 82, "bp_treated": True},
        "medications": [{"name": "Metformin", "dose": "500 mg BD"},
                        {"name": "Glimepiride", "dose": "2 mg OD"},
                        {"name": "Amlodipine", "dose": "5 mg OD"}],
        "lifestyle": {"smoker": False, "physical_activity": "moderate", "diet": "South Indian vegetarian"},
        "family_history": {"diabetes_parents": 1, "premature_cvd": False},
        "genetics": {"TCF7L2_rs7903146": "TT", "t2d_prs_percentile": 74},
        "story": "On a sulfonylurea. Fasts on day 4 (vrat), skips lunch and walks to the temple.",
    },
    "P003": {
        "id": "P003",
        "demographics": {"name": "Arjun Mehta", "age": 44, "sex": "M", "city": "Bengaluru",
                          "weight_kg": 70, "height_cm": 172, "waist_cm": 94, "south_asian": True},
        "conditions": [{"code": "R73.03", "name": "Prediabetes", "since": 2024}],
        "labs": _labs(112, 13, 6.2, 205, 36, 132, 190, 0.9, 98),
        "vitals": {"sbp": 132, "dbp": 86, "bp_treated": False},
        "medications": [],
        "lifestyle": {"smoker": False, "physical_activity": "sedentary", "diet": "Dosa / idli, frequent chai + biscuits, work deliveries"},
        "family_history": {"diabetes_parents": 1, "premature_cvd": True},
        "genetics": {"TCF7L2_rs7903146": "CC", "t2d_prs_percentile": 66},
        "story": "Normal BMI but central obesity (thin-fat). Release week at work: sleep collapses, HRV falls, resting HR climbs.",
    },
}


def bmi(ehr: dict) -> float:
    d = ehr["demographics"]
    return d["weight_kg"] / (d["height_cm"] / 100.0) ** 2


def random_patient(seed: int) -> dict:
    """Seeded random cohort member for backtesting."""
    r = np.random.default_rng(seed)
    sex = "M" if r.random() < 0.55 else "F"
    age = int(r.integers(32, 72))
    h = float(r.normal(167 if sex == "M" else 154, 6))
    w = float(np.clip(r.normal(72 if sex == "M" else 63, 10), 45, 110))
    status = r.choice(["pre", "t2d", "t2d_su"], p=[0.3, 0.5, 0.2])
    fpg = {"pre": r.uniform(102, 124), "t2d": r.uniform(130, 190), "t2d_su": r.uniform(110, 140)}[status]
    fi = r.uniform(8, 20)
    a1c = round(4.4 + fpg / 33.0 + r.normal(0, 0.3), 1)
    meds = []
    if status != "pre":
        meds.append({"name": "Metformin", "dose": "500-1000 mg BD"})
    if status == "t2d_su":
        meds.append({"name": "Gliclazide", "dose": "80 mg OD"})
    return {
        "id": f"C{seed:03d}",
        "demographics": {"name": f"Cohort {seed}", "age": age, "sex": sex, "city": "Synthetic",
                          "weight_kg": round(w), "height_cm": round(h), "waist_cm": round(w * 1.25),
                          "south_asian": True},
        "conditions": [{"code": "R73.03" if status == "pre" else "E11.9",
                        "name": "Prediabetes" if status == "pre" else "Type 2 diabetes mellitus", "since": 2020}],
        "labs": _labs(round(fpg), round(fi, 1), a1c, round(r.uniform(160, 240)), round(r.uniform(32, 55)),
                      round(r.uniform(90, 160)), round(r.uniform(110, 260)), 1.0, round(r.uniform(60, 100))),
        "vitals": {"sbp": round(r.uniform(118, 152)), "dbp": round(r.uniform(74, 94)), "bp_treated": bool(r.random() < 0.4)},
        "medications": meds,
        "lifestyle": {"smoker": bool(r.random() < 0.2), "physical_activity": r.choice(["sedentary", "low", "moderate"]),
                      "diet": "Mixed Indian"},
        "family_history": {"diabetes_parents": int(r.integers(0, 3)), "premature_cvd": bool(r.random() < 0.3)},
        "genetics": {"TCF7L2_rs7903146": r.choice(["CC", "CT", "TT"], p=[0.55, 0.37, 0.08]),
                     "t2d_prs_percentile": int(r.integers(10, 95))},
        "story": "Random cohort member.",
    }
