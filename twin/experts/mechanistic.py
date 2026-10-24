"""Expert 1 - Mechanistic physiology twin with data assimilation.

Parameters come from the EHR (HOMA-IR/HOMA-B, meds, TCF7L2), wearables modulate insulin
sensitivity in real time, and the twin continuously re-calibrates one latent state -
an insulin-sensitivity scale theta - against the last 8 h of CGM (grid search,
like weather-model data assimilation; not training).

Why it matters: theta drifting away from 1.0 means the real patient no longer
behaves like their EHR says (illness, missed meds, steroids, stress). That is the
"twin divergence" signal - something a pure black-box forecaster cannot give you.

The same simulator answers counterfactual "what if" questions (meal swaps, a walk
after dinner, adding a drug) - something no statistical model can do.
"""
from __future__ import annotations

import numpy as np

from ..physiology import TwinParams, params_from_ehr, simulate

THETA_GRID = np.array([0.5, 0.65, 0.8, 1.0, 1.2, 1.45])


class MechanisticExpert:
    name = "mechanistic"

    def __init__(self, bundle: dict):
        self.ehr = bundle["ehr"]
        self.base = params_from_ehr(self.ehr)
        self.df = bundle["df"]
        self.cgm = self.df["cgm"].to_numpy()
        self.meal_log = bundle["meal_log"]
        # observed (not true) per-minute activity from step counts, and wearable SI modulation
        self.act_min = np.repeat(self.df["activity"].to_numpy(), 5)
        self.si_obs = bundle["si_obs_min"]
        self.su_on = bundle["su_on_min"]
        self._cache: dict[int, tuple] = {}

    # ---------------------------------------------------------------
    def _params(self, theta: float, overrides: dict | None = None, dgb: float = 0.0) -> TwinParams:
        p = TwinParams(**{k: v for k, v in self.base.to_dict().items() if k != "notes"})
        p.SI *= theta
        p.Gb += dgb
        for k, v in (overrides or {}).items():
            setattr(p, k, v)
        return p

    def _meals(self, t_from: int, t_to: int):
        return [(m["t"], m["carbs_logged"], m["gi"]) for m in self.meal_log if t_from <= m["t"] < t_to]

    def _run_window(self, theta: float, origin: int, win: int = 480, preroll: int = 240, dgb: float = 0.0):
        start = max(origin - win - preroll, 0)
        cgm_idx = start // 5
        G0 = self.cgm[cgm_idx]
        p = self._params(theta, dgb=dgb)
        s0 = np.array([G0, 0.0, p.Ib + p.su_rate * self.su_on[start] / p.n, 0.0, 0.0])
        n = origin - start
        G, s = simulate(p, s0, start, n, self._meals(start - 180, origin),
                        self.act_min[start:origin], self.si_obs[start:origin], self.su_on[start:origin])
        return start, G, s

    def assimilate(self, origin: int):
        """Estimate two latent states against the last 8 h of CGM:
          theta - insulin-sensitivity scale vs EHR (grid search)
          dGb   - shift of basal (hepatic) glucose vs EHR fasting glucose (one Gauss-Newton step)
        Returns (theta, rmse, state, dGb)."""
        if origin in self._cache:
            return self._cache[origin]
        best = None
        for th in THETA_GRID:
            dgb = 0.0
            for _ in range(2):
                start, G, s = self._run_window(th, origin, dgb=dgb)
                m_idx = np.arange(max(origin - 480, start), origin, 5)
                resid = self.cgm[m_idx // 5] - G[m_idx - start]
                if _ == 0:
                    dgb = float(np.clip(np.mean(resid), -40, 60))
            rmse = float(np.sqrt(np.mean(resid ** 2)))
            # priors: EHR is informative (theta ~ 1, dGb ~ 0)
            score = rmse + 6.0 * abs(np.log(th)) + 0.04 * abs(dgb)
            if best is None or score < best[0]:
                best = (score, th, rmse, s, dgb)
        _, th, rmse, s, dgb = best
        s = s.copy()
        s[0] = 0.5 * s[0] + 0.5 * self.cgm[origin // 5]   # nudge to latest observation
        self._cache[origin] = (float(th), rmse, s, dgb)
        return self._cache[origin]

    def forecast(self, origin: int, horizon_steps: int = 24):
        """Median forecast of CGM at 5-min steps after origin, plus a simple band."""
        th, rmse, s, dgb = self.assimilate(origin)
        n = horizon_steps * 5
        p = self._params(th, dgb=dgb)
        si = np.full(n, self.si_obs[min(origin, len(self.si_obs) - 1)])
        su = self.su_on[origin: origin + n] if origin + n <= len(self.su_on) else np.full(n, self.su_on[-1])
        G, _ = simulate(p, s, origin, n, [], np.zeros(n), si, su)
        med = G[4::5][:horizon_steps]
        sd = (6.0 + 0.5 * rmse) * np.sqrt(np.arange(1, horizon_steps + 1) / 4.0)
        return med - 1.28 * sd, med, med + 1.28 * sd, {"theta": th, "fit_rmse": rmse, "dgb": dgb}

    # ---------------------------------------------------------------
    def counterfactual(self, origin: int, plan: dict, hours: float = 4.0):
        """Simulate a plan from `origin`:
        plan = {meal_key, carbs, gi, delay_min, walk_start_after_meal, walk_minutes,
                walk_intensity, add_metformin, sleep_hours}"""
        th, _, s, dgb = self.assimilate(origin)
        overrides = {}
        if plan.get("add_metformin"):
            overrides = {"SI": self.base.SI * th * 1.2, "Gb": (self.base.Gb + dgb) * 0.93}
        p = self._params(th, overrides, dgb=dgb)
        n = int(hours * 60)
        meals = []
        t_meal = origin + int(plan.get("delay_min", 0))
        if plan.get("carbs", 0) > 0:
            meals.append((t_meal, float(plan["carbs"]), float(plan.get("gi", 65))))
        act = np.zeros(n)
        wm = int(plan.get("walk_minutes", 0))
        if wm > 0:
            ws = int(plan.get("delay_min", 0)) + int(plan.get("walk_start_after_meal", 15))
            act[ws: ws + wm] = float(plan.get("walk_intensity", 0.7))
        si_base = self.si_obs[min(origin, len(self.si_obs) - 1)]
        if plan.get("sleep_hours") is not None:
            from ..wearables import si_multiplier_from_wearables
            si_base = si_multiplier_from_wearables(float(plan["sleep_hours"]), 0.0)
        su = np.full(n, self.su_on[min(origin, len(self.su_on) - 1)])
        G, _ = simulate(p, s, origin, n, meals, act, np.full(n, si_base), su)
        return G
