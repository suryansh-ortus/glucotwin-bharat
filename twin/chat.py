"""Patient-aware chatbot for the doctor (DeepInfra LLM, or a local LLM on the edge).

Every turn the bot is grounded in the twin's live state: health record, current glucose,
2-h forecast, alerts, expert trust weights, twin divergence, sleep / HRV / PPG nights,
recent meals, risk scores and the online adapter. If the question is a what-if
("what if he skips dinner?"), the physiology twin simulates it first and the numbers
are handed to the LLM - the model explains, the simulator computes.

All of it passes through the privacy gateway: the LLM sees "Patient-3F9A", never a name.
"""
from __future__ import annotations

import json
import re

import numpy as np

from .llm import _whatif_rules

SYS = (
    "You are GlucoTwin, a clinical assistant inside an endocrinologist's dashboard in India. "
    "Answer ONLY from the patient context provided (it is the live digital twin). Quote the numbers you use. "
    "If something is not in the context, say it isn't available. Be concise: 2-6 sentences or a short list. "
    "Suggest culturally appropriate Indian food / lifestyle changes when relevant. You support, not replace, "
    "the doctor's judgement; do not add generic disclaimers."
)

WHATIF_HINT = re.compile(r"\b(what if|instead|swap|skip|walk|add metformin|if (he|she) (eats|had|ate|sleeps|slept))\b", re.I)


def context(engine, snap: dict) -> dict:
    e = engine.b["ehr"]
    nt = snap["autonomic"]["table"]
    o = snap["origin"]
    cols = [c for c in ("date", "sleep_hours", "ppg_rmssd_ms", "rmssd_ms", "resting_hr", "ppg_embedding_drift") if c in nt]
    return {
        "patient": {"name": e["demographics"]["name"], "id": e["id"], "age": e["demographics"]["age"],
                    "sex": e["demographics"]["sex"], "city": e["demographics"]["city"],
                    "conditions": [c["name"] for c in e["conditions"]], "labs": e["labs"], "vitals": e["vitals"],
                    "medications": [f"{m['name']} {m['dose']}" for m in e["medications"]],
                    "lifestyle": e["lifestyle"], "family_history": e["family_history"], "genetics": e["genetics"]},
        "now": {"day": o // 1440 + 1, "time": f"{(o % 1440) // 60:02d}:{o % 60:02d}",
                "glucose_mg_dl": round(snap["current"]),
                "forecast_2h": {"min": round(float(np.min(snap["forecast"]["median"]))),
                                "max": round(float(np.max(snap["forecast"]["median"]))),
                                "at_60min": round(float(snap["forecast"]["median"][11])),
                                "at_120min": round(float(snap["forecast"]["median"][23]))},
                "p_above_180_2h": round(snap["p_spike"], 2), "p_below_70_2h": round(snap["p_hypo"], 2),
                "alerts": [a["message"] for a in snap["alerts"]]},
        "cgm_last_24h": {k: round(v, 1) for k, v in snap["cgm"].items()},
        "cgm_week": {k: round(v, 1) for k, v in snap["cgm_all"].items()},
        "twin": {"insulin_sensitivity_vs_record": round(snap["theta"], 2),
                 "basal_glucose_shift_vs_record": round(snap["dgb"], 1),
                 "expert_trust": {k: round(v, 2) for k, v in snap["weights"].items()},
                 "router_reason": snap["router"]["rationale"],
                 "online_adapter": {k: v for k, v in snap["adapter"].items() if k in ("updates", "mae_without", "mae_with")}},
        "risk": {"framingham_10y_cvd": round(snap["framingham_10y"], 3), "idrs": snap["idrs"]},
        "autonomic": snap["autonomic"]["message"],
        "nights": nt[cols].round(1).to_dict("records"),
        "recent_meals": [{"minutes_ago": o - m["t"], "food": m["label"], "carbs_g": m["carbs_logged"], "gi": m["gi"]}
                         for m in engine.b["meal_log"] if o - 1440 <= m["t"] <= o],
        "twin_parameters_from_record": engine.mech.base.notes,
    }


def answer(engine, snap: dict, history: list[dict], message: str, whatif_fn=None) -> dict:
    ctx = context(engine, snap)
    sim = None
    if whatif_fn and WHATIF_HINT.search(message):
        sim = whatif_fn(message)
        if sim:
            ctx["what_if_simulation"] = sim
    llm = engine.llm
    msgs = [{"role": "system", "content": SYS + "\n\nPATIENT CONTEXT (live twin):\n" + json.dumps(ctx, default=str)}]
    msgs += [{"role": m["role"], "content": m["content"]} for m in history[-8:] if m["role"] in ("user", "assistant")]
    msgs.append({"role": "user", "content": message})
    reply = llm.chat_messages(msgs, 450, 0.3, purpose="chat") if llm.available else None
    if not reply:
        reply = _offline(ctx, message, sim)
    return {"reply": reply, "simulation": sim, "llm": llm.available}


def _offline(ctx: dict, q: str, sim) -> str:
    """No-key fallback: answer common questions straight from the twin."""
    t = q.lower()
    n, p = ctx["now"], ctx["patient"]
    if sim:
        return (f"Simulated: {sim['A']} peaks at {sim['A_peak']} mg/dL ({sim['A_min_above_180']} min above 180); "
                f"{sim['B']} peaks at {sim['B_peak']} mg/dL ({sim['B_min_above_180']} min above 180).")
    if any(w in t for w in ("forecast", "next", "predict", "spike", "hypo", "low", "high")):
        return (f"Now {n['glucose_mg_dl']} mg/dL. Next 2 h: {n['forecast_2h']['min']}-{n['forecast_2h']['max']} mg/dL, "
                f"P(>180) {n['p_above_180_2h']}, P(<70) {n['p_below_70_2h']}. Alerts: {'; '.join(n['alerts']) or 'none'}.")
    if any(w in t for w in ("sleep", "hrv", "heart", "ppg")):
        return ctx["autonomic"]
    if any(w in t for w in ("medic", "drug", "tablet")):
        return "Current medication: " + (", ".join(p["medications"]) or "none") + "."
    if any(w in t for w in ("risk", "cardio", "framingham")):
        return f"10-year CVD risk (Framingham) {ctx['risk']['framingham_10y_cvd'] * 100:.0f}%, IDRS {ctx['risk']['idrs']}."
    c = ctx["cgm_last_24h"]
    return (f"Last 24 h: time in range {c['tir']:.0f}%, GMI {c['gmi']:.1f}%, mean {c['mean']:.0f} mg/dL. "
            "(Offline mode - add DEEPINFRA_API_KEY or a local LLM for free-form answers.)")


def summarise_whatif(res: dict) -> dict | None:
    if "error" in res:
        return None
    return {"A": res["A"]["plan"]["summary"], "B": res["B"]["plan"]["summary"],
            "A_peak": round(res["A"]["peak"]), "B_peak": round(res["B"]["peak"]),
            "A_min_above_180": res["A"]["min_above_180"], "B_min_above_180": res["B"]["min_above_180"],
            "A_min_below_70": res["A"]["min_below_70"], "B_min_below_70": res["B"]["min_below_70"],
            "hours_simulated": 4, "from": res.get("from")}


__all__ = ["answer", "context", "summarise_whatif", "_whatif_rules"]
