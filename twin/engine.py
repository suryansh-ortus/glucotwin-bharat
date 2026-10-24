"""The twin engine: experts -> skill-scored, LLM-routed fusion -> calibrated event alerts.

Fusion (no training):
  w_skill  ~ 1 / MAE^2 of each expert on rolling hindcasts over the last 8 h
             (the twin grades its own experts on this patient, right now)
  w_router = LLM / rule router weights from the clinical context
  w        = 0.5 * w_skill + 0.5 * w_router
  band     = split-conformal interval from the fused hindcast residuals (per horizon)
  P(event) = max over horizon of Gaussian exceedance, sigma from the conformal band
"""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from math import erf, sqrt

import numpy as np

from .experts import autonomic, risk
from .experts.foundation import FoundationExpert
from .experts.mechanistic import MechanisticExpert
from .experts.ppg import PPGExpert
from .adaptive import OnlineAdapter, features
from .privacy import GATEWAY
from .llm import LLM, clinical_brief, llm_forecast, llm_router, parse_whatif, rule_router

H = 24              # 24 x 5 min = 2 h
HIND_N = 16         # hindcast origins
HIND_STEP = 30      # minutes between hindcast origins
SPIKE, HYPO = 180.0, 70.0


def _phi(x):
    return 0.5 * (1 + erf(x / sqrt(2)))


class TwinEngine:
    def __init__(self, bundle: dict, llm: LLM | None = None, foundation: FoundationExpert | None = None,
                 use_llm_forecaster: bool = True, ppg: PPGExpert | None = None,
                 adapter_prior: np.ndarray | None = None, online: bool = True):
        self.b = bundle
        GATEWAY.register_patient(bundle["ehr"])          # pseudonymise before any LLM call
        self.adapter_prior = adapter_prior
        self.online = online
        self.adapter = OnlineAdapter(adapter_prior)
        self._last_origin = -1
        self._catching = False
        # PPG expert: raw overnight pulse waveforms -> HR, HRV, PaPaGei morphology drift (per night)
        self.ppg = ppg or PPGExpert()
        bundle["nightly"], self.ppg_samples = self.ppg.analyse_week(bundle["nightly"], bundle["ehr"])
        self.cgm = bundle["df"]["cgm"].to_numpy()
        self.mech = MechanisticExpert(bundle)
        self.fm = foundation or FoundationExpert()
        self.llm = llm or LLM()
        # LLM forecaster is opt-in (LLM_FORECASTER=1): ~17 extra calls per new forecast origin
        self.use_llm_fc = use_llm_forecaster and self.llm.available and os.getenv("LLM_FORECASTER") == "1"
        self._llm_cache: dict[int, np.ndarray | None] = {}
        self._fc_cache: dict[int, dict] = {}

    # ------------------------------------------------------------ helpers
    def _actual(self, origin: int) -> np.ndarray:
        i = origin // 5
        return self.cgm[i + 1: i + 1 + H]

    def _meals_text(self, origin: int) -> str:
        ms = [m for m in self.b["meal_log"] if origin - 240 <= m["t"] <= origin]
        return "; ".join(f"{m['label']} ({m['carbs_logged']} g carbs, GI {m['gi']}) {origin - m['t']} min ago" for m in ms)

    def _llm_fc(self, origin: int):
        if origin in self._llm_cache:
            return self._llm_cache[origin]
        i = origin // 5
        day = origin // 1440
        nt = self.b["nightly"]
        sleep = float(nt.loc[nt.day == max(day - 1, 0), "sleep_hours"].iloc[0])
        hod = (origin % 1440) / 60
        extra = (f"Time of day: {int(hod):02d}:{int((hod % 1) * 60):02d}. Last night's sleep: {sleep:.1f} h. "
                 f"Medications: {', '.join(m['name'] for m in self.b['ehr']['medications']) or 'none'}.")
        f = llm_forecast(self.llm, self.cgm[i - 35: i + 1], self._meals_text(origin), extra)
        self._llm_cache[origin] = None if f is None else np.array(f)
        return self._llm_cache[origin]

    def experts_at(self, origins: list[int]) -> dict:
        """Forecasts of every expert at each origin -> {name: (q10, q50, q90) arrays [len(origins), H]}.
        Per-origin results are cached, so hindcasts are computed once."""
        todo = [o for o in origins if o not in self._fc_cache]
        if todo:
            fq = self.fm.forecast([self.cgm[: o // 5 + 1] for o in todo], H)
            llm_fs = None
            if self.use_llm_fc:
                with ThreadPoolExecutor(8) as ex:
                    llm_fs = list(ex.map(self._llm_fc, todo))
            for j, o in enumerate(todo):
                m = self.mech.forecast(o, H)
                d = {"foundation": (fq[0][j], fq[1][j], fq[2][j]), "mechanistic": m[:3]}
                if llm_fs is not None and llm_fs[j] is not None:
                    sd = 10 * np.sqrt(np.arange(1, H + 1) / 4.0)
                    d["llm"] = (llm_fs[j] - 1.28 * sd, llm_fs[j], llm_fs[j] + 1.28 * sd)
                self._fc_cache[o] = d
        names = [n for n in ("mechanistic", "foundation", "llm") if all(n in self._fc_cache[o] for o in origins)]
        return {n: tuple(np.array([self._fc_cache[o][n][q] for o in origins]) for q in range(3)) for n in names}

    def adapter_features(self, o: int, fused_row: np.ndarray) -> np.ndarray:
        i = o // 5
        cur = float(self.cgm[i])
        trend = float((self.cgm[i] - self.cgm[i - 3]) / 15)
        past = [m for m in self.b["meal_log"] if m["t"] <= o]
        msm = o - past[-1]["t"] if past else None
        carbs = sum(m["carbs_logged"] for m in past if o - m["t"] <= 180)
        c = self._fc_cache[o]
        return features(cur, trend, msm, carbs, fused_row, c["mechanistic"][1], c["foundation"][1], (o % 1440) / 60)

    def router_context(self, origin: int, theta: float, skill: dict, dgb: float = 0.0) -> dict:
        last_meal = [m for m in self.b["meal_log"] if m["t"] <= origin]
        msm = origin - last_meal[-1]["t"] if last_meal else None
        hod = (origin % 1440) / 60
        i = origin // 5
        return {
            "time": f"{int(hod):02d}:{int((hod % 1) * 60):02d}",
            "night": hod < 6 or hod >= 23.5,
            "current_glucose": round(float(self.cgm[i]), 0),
            "trend_mg_dl_per_min": round(float((self.cgm[i] - self.cgm[i - 3]) / 15), 2),
            "minutes_since_meal": msm,
            "last_meal": last_meal[-1]["label"] if last_meal else None,
            "theta": theta,
            "basal_shift": round(dgb, 1),
            "hindcast_mae": {k: round(v, 1) for k, v in skill.items()},
            "llm_available": self.use_llm_fc,
        }

    # ------------------------------------------------------------ main
    def snapshot(self, origin: int, with_brief: bool = False) -> dict:
        origin = int(origin // 5 * 5)
        hind = [origin - 120 - HIND_STEP * j for j in range(HIND_N)][::-1]
        hind = [o for o in hind if o >= 1440]
        allx = self.experts_at(hind + [origin])
        names = list(allx.keys())

        # skill on hindcasts
        actual = np.array([self._actual(o) for o in hind]) if hind else None
        skill = {}
        for n in names:
            if actual is not None and len(actual):
                skill[n] = float(np.mean(np.abs(allx[n][1][:-1] - actual)))
            else:
                skill[n] = 20.0
        inv = {n: 1.0 / (skill[n] ** 2 + 1.0) for n in names}
        w_skill = {n: inv[n] / sum(inv.values()) for n in names}

        theta, fit_rmse, _, dgb = self.mech.assimilate(origin)
        ctx = self.router_context(origin, theta, skill, dgb)
        route = llm_router(self.llm, ctx) if (self.llm.available and not self._catching) else rule_router(ctx)
        w_route = {n: route["weights"].get(n, 0.0) for n in names}
        s = sum(w_route.values()) or 1.0
        w_route = {n: v / s for n, v in w_route.items()}
        w = {n: 0.5 * w_skill[n] + 0.5 * w_route[n] for n in names}
        s = sum(w.values())
        w = {n: v / s for n, v in w.items()}

        fused_all = sum(w[n] * allx[n][1] for n in names)            # [len(hind)+1, H]
        fused_raw = fused_all[-1]

        # ---- on-device online learning: matured forecasts update the adapter (predict, then update)
        if origin < self._last_origin:                                 # clock moved backwards -> no peeking
            self.adapter = OnlineAdapter(self.adapter_prior)
            self._last_origin = -1
        if self.online and not self._catching and origin - max(self._last_origin, 1440) > 360:
            # the device has been running all along: replay the past (only data before `origin`)
            self._catching = True
            for o in range(max(self._last_origin, 1440) + 240, origin - 120, 120):
                self.snapshot(o)
            self._catching = False
        self._last_origin = origin
        if self.online:
            for j, o in enumerate(hind):
                if o not in self.adapter.seen:
                    self.adapter.seen.add(o)
                    self.adapter.update(self.adapter_features(o, fused_all[j]), fused_all[j], actual[j])
            fused = fused_raw + self.adapter.correction(self.adapter_features(origin, fused_raw))
        else:
            fused = fused_raw
        if actual is not None and len(actual) >= 4:
            q_h = np.quantile(np.abs(fused_all[:-1] - actual), 0.8, axis=0)
        else:
            q_h = np.full(H, 15.0)
        half_mix = sum(w[n] * (allx[n][2][-1] - allx[n][0][-1]) / 2 for n in names)
        half = np.maximum(q_h, 0.6 * half_mix)
        sigma = np.maximum(half / 1.28, 3.0)
        p_spike = max(_phi((fused[k] - SPIKE) / sigma[k]) for k in range(H))
        p_hypo = max(_phi((HYPO - fused[k]) / sigma[k]) for k in range(H))
        cur = float(self.cgm[origin // 5])

        alerts = []
        if p_spike >= 0.5 and cur < SPIKE:
            k = int(np.argmax(fused > SPIKE)) if (fused > SPIKE).any() else int(np.argmax(fused))
            alerts.append({"type": "spike", "severity": "warning", "p": p_spike,
                           "message": f"Glucose predicted to cross 180 mg/dL in ~{(k + 1) * 5} min "
                                      f"(peak {fused.max():.0f}, p={p_spike:.2f})."})
        if cur >= 250:
            alerts.append({"type": "high", "severity": "warning", "p": 1.0,
                           "message": f"Currently {cur:.0f} mg/dL (sustained hyperglycaemia)."})
        if cur < HYPO:
            alerts.append({"type": "hypo", "severity": "critical", "p": 1.0,
                           "message": f"Glucose is low now ({cur:.0f} mg/dL) and the twin sees no recovery in the next 2 h "
                                      f"without carbs. Give 15 g fast carbs; review sulfonylurea dose."
                                      if fused.max() < 80 else f"Glucose is low now ({cur:.0f} mg/dL)."})
        elif p_hypo >= 0.3:
            k = int(np.argmax(fused < HYPO)) if (fused < HYPO).any() else int(np.argmin(fused))
            alerts.append({"type": "hypo", "severity": "critical", "p": p_hypo,
                           "message": f"Hypoglycaemia risk: predicted < 70 mg/dL around +{(k + 1) * 5} min "
                                      f"(p={p_hypo:.2f}). Check sulfonylurea / missed meal."})
        prev = self.mech.assimilate(max(origin - 240, 1440))
        drift_now = theta <= 0.65 or dgb >= 15
        drift_prev = prev[0] <= 0.65 or prev[3] >= 12
        if drift_now and drift_prev:
            alerts.append({"type": "drift", "severity": "warning", "p": None,
                           "message": f"Twin divergence: the real patient no longer matches their EHR twin - basal glucose "
                                      f"{dgb:+.0f} mg/dL, insulin sensitivity x{theta:.2f} (sustained > 4 h). "
                                      f"Consider infection, missed doses, steroids or acute stress."})
        day = origin // 1440
        auto = autonomic.assess(self.b["nightly"], day)
        if auto["level"] in ("high", "moderate"):
            alerts.append({"type": "autonomic", "severity": "warning" if auto["level"] == "high" else "info",
                           "p": None, "message": auto["message"]})

        i = origin // 5
        last24 = self.cgm[max(i - 287, 0): i + 1]
        snap = {
            "origin": origin,
            "patient": self.b["ehr"]["demographics"]["name"],
            "current": cur,
            "forecast": {"median": fused, "lo": fused - half, "hi": fused + half, "raw_median": fused_raw,
                         "experts": {n: allx[n][1][-1] for n in names}},
            "weights": w, "w_skill": w_skill, "w_route": w_route, "skill_mae": skill,
            "router": route, "theta": theta, "dgb": dgb, "fit_rmse": fit_rmse,
            "p_spike": p_spike, "p_hypo": p_hypo, "alerts": alerts,
            "cgm": risk.cgm_metrics(last24), "cgm_all": risk.cgm_metrics(self.cgm[: i + 1]),
            "framingham_10y": risk.framingham_cvd(self.b["ehr"]), "idrs": risk.idrs(self.b["ehr"]),
            "adapter": self.adapter.summary(),
            "autonomic": auto, "foundation_backend": self.fm.backend, "ppg_backend": self.ppg.backend,
            "llm": {"available": self.llm.available, "model": self.llm.model, "cost_usd": self.llm.cost_usd,
                    "calls": self.llm.usage["calls"]},
        }
        if with_brief:
            snap["brief"] = clinical_brief(self.llm, self.brief_payload(snap))
        return snap

    def brief_payload(self, snap: dict) -> dict:
        e = self.b["ehr"]
        return {
            "patient": snap["patient"], "age": e["demographics"]["age"], "sex": e["demographics"]["sex"],
            "conditions": [c["name"] for c in e["conditions"]], "hba1c": e["labs"]["hba1c"],
            "medications": [f"{m['name']} {m['dose']}" for m in e["medications"]],
            "current_glucose": round(snap["current"]),
            "forecast_2h_peak": round(float(snap["forecast"]["median"].max())),
            "forecast_2h_min": round(float(snap["forecast"]["median"].min())),
            "p_spike_2h": round(snap["p_spike"], 2), "p_hypo_2h": round(snap["p_hypo"], 2),
            "cgm": {k: round(v, 1) for k, v in snap["cgm"].items()},
            "theta": round(snap["theta"], 2), "basal_glucose_shift": round(snap["dgb"]), "framingham_10y": round(snap["framingham_10y"], 3),
            "autonomic": snap["autonomic"]["message"], "alerts": [{"message": a["message"]} for a in snap["alerts"]],
            "recent_meals": self._meals_text(snap["origin"]),
        }

    def drift_history(self, upto: int, step: int = 120) -> list[tuple[int, float, float]]:
        """(minute, theta, dGb) every `step` minutes - the twin's latent physiology over time."""
        return [(o, *(lambda a: (a[0], a[3]))(self.mech.assimilate(o))) for o in range(1440, upto + 1, step)]

    # ------------------------------------------------------------ what-if
    def whatif(self, origin: int, plan_a: dict, plan_b: dict, hours: float = 4.0) -> dict:
        origin = int(origin // 5 * 5)
        out = {}
        for tag, plan in (("A", plan_a), ("B", plan_b)):
            g = self.mech.counterfactual(origin, plan, hours)
            out[tag] = {"curve": g, "peak": float(g.max()), "min": float(g.min()),
                        "min_above_180": int((g > 180).sum()), "min_below_70": int((g < 70).sum()),
                        "iauc": float(np.clip(g - g[0], 0, None).sum() / 60.0), "plan": plan}
        return out

    def parse(self, text: str) -> dict:
        return parse_whatif(self.llm, text)
