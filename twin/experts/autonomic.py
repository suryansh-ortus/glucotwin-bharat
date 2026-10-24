"""Expert 3 - Autonomic / cardiovascular anomaly detector (week-scale).

Personal-baseline robust z-scores on nightly RMSSD (HRV), resting HR and sleep.
HRV comes from the PPG expert (computed from the raw pulse waveform) when present,
plus PaPaGei pulse-morphology drift. No training: the patient is their own control.

Evidence: low HRV predicts cardiovascular events (Hillebrand et al. 2013, Europace,
meta-analysis); rising resting HR is an independent CV risk marker (Fox et al. 2007);
short sleep worsens insulin sensitivity (Donga et al. 2010).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _robust_z(x: np.ndarray, base: np.ndarray, min_sd: float) -> np.ndarray:
    med = np.median(base)
    mad = 1.4826 * np.median(np.abs(base - med))
    return (x - med) / max(mad, min_sd)


def assess(nightly: pd.DataFrame, upto_day: int) -> dict:
    nt = nightly[nightly["day"] <= upto_day].copy()
    if len(nt) < 3:
        return {"level": "insufficient", "message": "Need 3 nights for a personal baseline.", "table": nt}
    base = nt.iloc[:3]
    hrv_col = "ppg_rmssd_ms" if "ppg_rmssd_ms" in nt and nt["ppg_rmssd_ms"].notna().all() else "rmssd_ms"
    nt["z_rmssd"] = _robust_z(nt[hrv_col].to_numpy(), base[hrv_col].to_numpy(), 3.0)
    nt["z_rhr"] = _robust_z(nt["resting_hr"].to_numpy(), base["resting_hr"].to_numpy(), 1.5)
    nt["z_sleep"] = _robust_z(nt["sleep_hours"].to_numpy(), base["sleep_hours"].to_numpy(), 0.4)
    nt["strain"] = (-nt["z_rmssd"]).clip(lower=0) + nt["z_rhr"].clip(lower=0) + 0.5 * (-nt["z_sleep"]).clip(lower=0)
    last2 = nt.tail(2)
    hrv_slope = float(np.polyfit(nt["day"], nt[hrv_col], 1)[0])
    rhr_slope = float(np.polyfit(nt["day"], nt["resting_hr"], 1)[0])
    sustained = bool(((last2["z_rmssd"] < -2) & (last2["z_rhr"] > 2)).all())
    if sustained or (hrv_slope < -2.5 and rhr_slope > 1.2):
        level = "high"
        msg = (f"Sustained autonomic strain: HRV {hrv_slope:+.1f} ms/night, resting HR {rhr_slope:+.1f} bpm/night "
               f"vs personal baseline. Pattern seen with infection, overtraining, sleep debt or emerging cardiac stress. "
               f"Recommend clinical review, ECG if symptomatic.")
    elif (last2["strain"] > 2.5).any():
        level = "moderate"
        msg = "Autonomic markers worsening over the last 2 nights - watch for illness or sleep debt."
    else:
        level = "normal"
        msg = "HRV, resting HR and sleep within personal baseline."
    morph = nt["ppg_embedding_drift"].iloc[-1] if "ppg_embedding_drift" in nt else np.nan
    if np.isfinite(morph) and morph >= 3.0:
        msg += (f" PaPaGei pulse-waveform shape has drifted {morph:.1f} SD from this patient's baseline nights "
                f"(vascular tone / stiffness change).")
        if level == "normal":
            level = "moderate"
    return {"level": level, "message": msg, "hrv_slope": hrv_slope, "rhr_slope": rhr_slope, "table": nt,
            "hrv_source": "PPG waveform" if hrv_col.startswith("ppg") else "wearable"}
