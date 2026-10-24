"""LLM layer on DeepInfra (OpenAI-compatible). Cheap model by default.

Four jobs - none of them is "train a model":
  1. ROUTER        - reads the situation (meal just logged? twin drifting? night?) and
                     sets trust weights over the experts, with a one-line rationale.
  2. LLM FORECASTER- an extra zero-shot expert (LLMTime-style, Gruver et al. 2023):
                     reads recent CGM + meals + sleep as text, writes the next 2 h.
                     The fusion layer scores it on hindcasts like any other expert,
                     so if it is bad it gets down-weighted automatically.
  3. CLINICAL BRIEF- doctor-facing summary + actions from all expert outputs.
  4. WHAT-IF PARSER- natural-language question -> structured plan for the
                     mechanistic simulator ("what if he walks 20 min after dinner?").

Every job has a deterministic fallback, so the app works without an API key.
"""
from __future__ import annotations

import json
import os
import re
import threading

from .meals import MEALS
from .privacy import GATEWAY

DEFAULT_MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct"
# USD per 1M tokens (input, output) - check deepinfra.com/pricing
PRICES = {
    "meta-llama/Meta-Llama-3.1-8B-Instruct": (0.02, 0.05),
    "meta-llama/Llama-3.2-3B-Instruct": (0.02, 0.02),
    "openai/gpt-oss-20b": (0.04, 0.15),
}


class LLM:
    """OpenAI-compatible client. Default: DeepInfra cloud. Set LLM_BASE_URL to a local server
    (Ollama http://localhost:11434/v1, llama.cpp, vLLM) for fully on-device inference.
    Every call goes through the privacy gateway (twin/privacy.py)."""

    def __init__(self):
        self.base_url = os.getenv("LLM_BASE_URL", "https://api.deepinfra.com/v1/openai").strip()
        self.local = not self.base_url.startswith("https://api.deepinfra.com")
        self.key = os.getenv("DEEPINFRA_API_KEY", "").strip() or ("local" if self.local else "")
        self.model = os.getenv("DEEPINFRA_MODEL", DEFAULT_MODEL)
        self.client = None
        self.usage = {"calls": 0, "in": 0, "out": 0}
        self._lock = threading.Lock()
        if self.key:
            try:
                from openai import OpenAI
                self.client = OpenAI(api_key=self.key, base_url=self.base_url, timeout=40)
            except Exception:  # noqa: BLE001
                self.client = None

    @property
    def available(self) -> bool:
        return self.client is not None

    @property
    def cost_usd(self) -> float:
        pi, po = PRICES.get(self.model, (0.05, 0.10))
        return (self.usage["in"] * pi + self.usage["out"] * po) / 1e6

    def chat(self, system: str, user: str, max_tokens: int = 400, temperature: float = 0.2,
             purpose: str = "llm") -> str | None:
        return self.chat_messages([{"role": "system", "content": system}, {"role": "user", "content": user}],
                                  max_tokens, temperature, purpose)

    def chat_messages(self, messages: list[dict], max_tokens: int = 400, temperature: float = 0.2,
                      purpose: str = "llm") -> str | None:
        if not self.client:
            return None
        messages = GATEWAY.outbound(purpose, messages)        # de-identify + audit log
        try:
            r = self.client.chat.completions.create(
                model=self.model, max_tokens=max_tokens, temperature=temperature, messages=messages,
            )
            with self._lock:
                self.usage["calls"] += 1
                if r.usage:
                    self.usage["in"] += r.usage.prompt_tokens
                    self.usage["out"] += r.usage.completion_tokens
            return GATEWAY.inbound(r.choices[0].message.content)   # re-identify locally
        except Exception as e:  # noqa: BLE001
            print(f"[llm] {type(e).__name__}: {e}")
            return None


def _json(text: str | None):
    if not text:
        return None
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


# ------------------------------------------------------------------ 1. router
ROUTER_SYS = (
    "You are the routing layer of a diabetes digital twin. Experts: "
    "'mechanistic' (physiology ODE, knows logged meals, insulin action, exercise; can be wrong if the patient's "
    "physiology changed), 'foundation' (Chronos time-series foundation model, sees only the CGM curve, blind to meals), "
    "'llm' (language-model forecaster, if present). 'hindcast_mae' is each expert's recent error in mg/dL: "
    "LOWER IS BETTER, and 'most_accurate_recently' names the winner. Give more weight to the more accurate expert "
    "unless a meal was logged in the last hour (favour mechanistic) or the twin is drifting (favour foundation). "
    "Only state facts that are in the input. Output JSON "
    '{"weights": {"mechanistic": w1, "foundation": w2, "llm": w3}, "rationale": "one sentence"}. Weights sum to 1.'
)


def rule_router(ctx: dict) -> dict:
    w = {"mechanistic": 0.5, "foundation": 0.5, "llm": 0.0}
    why = []
    if ctx["minutes_since_meal"] is not None and ctx["minutes_since_meal"] <= 60:
        w = {"mechanistic": 0.75, "foundation": 0.25, "llm": 0.0}
        why.append("meal logged in last hour - only the physiology expert knows carbs are still being absorbed")
    elif ctx["night"]:
        w = {"mechanistic": 0.4, "foundation": 0.6, "llm": 0.0}
        why.append("overnight, no meals - curve-pattern model is reliable")
    if abs(ctx["theta"] - 1.0) >= 0.3 or abs(ctx.get("basal_shift", 0)) >= 18:
        w["mechanistic"] *= 0.7
        why.append(f"twin drift (SI x{ctx['theta']:.2f}, basal {ctx.get('basal_shift', 0):+.0f}) - physiology params less trustworthy")
    if ctx.get("llm_available"):
        w["llm"] = 0.15
    s = sum(w.values())
    return {"weights": {k: v / s for k, v in w.items()}, "rationale": "; ".join(why) or "steady state - equal trust",
            "source": "rules"}


def llm_router(llm: LLM, ctx: dict) -> dict:
    base = rule_router(ctx)
    if not llm.available:
        return base
    mae = ctx.get("hindcast_mae", {})
    payload = {k: v for k, v in ctx.items() if k != "llm_available"}
    if mae:
        payload["most_accurate_recently"] = min(mae, key=mae.get)
    out = _json(llm.chat(ROUTER_SYS, json.dumps(payload), 150, purpose="router"))
    try:
        w = {k: max(float(out["weights"].get(k, 0)), 0.0) for k in ("mechanistic", "foundation", "llm")}
        if not ctx.get("llm_available"):
            w["llm"] = 0.0
        s = sum(w.values())
        assert s > 0
        w = {k: v / s for k, v in w.items()}
        # sanity guard: a small LLM must not hand most of the trust to the clearly worse expert
        # without a contextual reason (meal just logged / twin drift)
        if "mechanistic" in mae and "foundation" in mae:
            top = max(("mechanistic", "foundation"), key=lambda k: w[k])
            other = "foundation" if top == "mechanistic" else "mechanistic"
            meal_now = ctx.get("minutes_since_meal") is not None and ctx["minutes_since_meal"] <= 60
            drift = abs(ctx.get("theta", 1) - 1) >= 0.3 or abs(ctx.get("basal_shift", 0)) >= 18
            justified = (top == "mechanistic" and meal_now) or (top == "foundation" and drift)
            if mae[top] > 1.3 * mae[other] and not justified:
                return {**base, "source": f"rules (LLM suggestion rejected: it favoured the less accurate expert)"}
        return {"weights": w, "rationale": str(out.get("rationale", ""))[:240],
                "source": f"LLM ({llm.model.split('/')[-1]})"}
    except Exception:  # noqa: BLE001
        return base


# ------------------------------------------------------------------ 2. forecaster
FORECAST_SYS = (
    "You forecast continuous glucose monitor (CGM) readings for a person with type 2 diabetes. "
    "Readings are every 5 minutes in mg/dL. Use physiology: carbs raise glucose for 1-3 h (high GI faster), "
    "walking lowers it, sulfonylureas can cause lows when meals are skipped. "
    'Reply ONLY with JSON {"forecast": [24 integers]} for the next 120 minutes.'
)


def llm_forecast(llm: LLM, recent_cgm, meals_text: str, extra: str) -> list[float] | None:
    prompt = (f"Last 3 hours CGM (oldest->newest): {', '.join(str(int(round(x))) for x in recent_cgm)}\n"
              f"Meals: {meals_text or 'none in last 4 h'}\n{extra}")
    out = _json(llm.chat(FORECAST_SYS, prompt, 220, 0.0, purpose="forecaster"))
    try:
        f = [float(x) for x in out["forecast"]][:24]
        if len(f) < 24:
            f += [f[-1]] * (24 - len(f))
        return [min(max(x, 40.0), 400.0) for x in f]
    except Exception:  # noqa: BLE001
        return None


# ------------------------------------------------------------------ 3. brief
BRIEF_SYS = (
    "You are a clinical decision-support assistant for an Indian endocrinologist reviewing a patient's "
    "digital twin. Be concise and specific, plain English, no disclaimers. Format:\n"
    "SUMMARY: 2 sentences.\nALERTS: bullet list (or 'none').\nACTIONS: 3 numbered, concrete, culturally "
    "appropriate (Indian food/lifestyle) actions. Never invent numbers not in the input."
)


def clinical_brief(llm: LLM, snap: dict) -> str:
    txt = llm.chat(BRIEF_SYS, json.dumps(snap, default=str), 380, 0.3, purpose="brief") if llm.available else None
    if txt:
        return txt
    a = snap
    alerts = [x["message"] for x in a["alerts"]] or ["none"]
    return (f"SUMMARY: {a['patient']} - TIR {a['cgm']['tir']:.0f}%, GMI {a['cgm']['gmi']:.1f}%, "
            f"10-yr CVD risk {a['framingham_10y']*100:.0f}%. Twin vs EHR: insulin sensitivity x{a['theta']:.2f}, basal glucose {a['basal_glucose_shift']:+d} mg/dL.\n"
            "ALERTS:\n" + "\n".join(f"- {x}" for x in alerts) +
            "\nACTIONS:\n1. Review the 2-h forecast and act on any spike/hypo alert.\n"
            "2. Swap high-GI dinner items (white rice, biryani) for millet roti / dal; 15-min walk after meals.\n"
            "3. If twin drift persists > 24 h, check for infection, missed doses or steroid use.\n"
            "(Template brief - set DEEPINFRA_API_KEY for the LLM version.)")


# ------------------------------------------------------------------ 4. what-if parser
WHATIF_SYS = (
    "Convert a doctor's what-if question about a diabetes patient into JSON for a physiology simulator. "
    "Meal keys allowed: " + ", ".join(k for k in MEALS if k != "none") + ". "
    'Output JSON: {"meal_key": str|null, "portion": float (1.0 = normal), "delay_min": int, '
    '"walk_minutes": int, "walk_start_after_meal": int, "walk_intensity": float 0.3-1.0, '
    '"add_metformin": bool, "sleep_hours": float|null, "summary": "short label"}'
)


def parse_whatif(llm: LLM, text: str) -> dict:
    out = _json(llm.chat(WHATIF_SYS, text, 200, 0.0, purpose="what-if parser")) if llm.available else None
    if not out:
        out = _whatif_rules(text)
        out["parser"] = "rules"
    else:
        out["parser"] = "LLM"
    key = out.get("meal_key")
    if key not in MEALS:
        key = None
    portion = float(out.get("portion") or 1.0)
    plan = {
        "meal_key": key,
        "carbs": MEALS[key]["carbs"] * portion if key else 0.0,
        "gi": MEALS[key]["gi"] if key else 60,
        "delay_min": int(out.get("delay_min") or 0),
        "walk_minutes": int(out.get("walk_minutes") or 0),
        "walk_start_after_meal": int(out.get("walk_start_after_meal") or 15),
        "walk_intensity": float(out.get("walk_intensity") or 0.7),
        "add_metformin": bool(out.get("add_metformin")),
        "sleep_hours": out.get("sleep_hours"),
        "summary": out.get("summary") or text[:60],
        "parser": out["parser"],
    }
    return plan


def _whatif_rules(text: str) -> dict:
    t = text.lower()
    key = None
    for k, m in MEALS.items():
        names = [k.replace("_", " ")] + [m["label"].split("(")[0].split("+")[0].strip().lower()]
        if any(n and n in t for n in names):
            key = k
            break
    walk = re.search(r"(\d+)\s*(?:-|\s)?min(?:ute)?s?\s*(?:brisk\s*)?walk", t) or re.search(r"walk\w*\s*(?:for\s*)?(\d+)", t)
    sleep = (re.search(r"(\d+(?:\.\d+)?)\s*h(?:ou)?rs?\s*(?:of\s*)?sleep", t)
             or re.search(r"(?:sleep|slept)\D{0,15}(\d+(?:\.\d+)?)\s*h", t))
    portion = 0.5 if "half" in t else 1.5 if "extra" in t or "double" in t else 1.0
    return {"meal_key": key, "portion": portion, "delay_min": 0,
            "walk_minutes": int(walk.group(1)) if walk else (20 if "walk" in t else 0),
            "walk_start_after_meal": 15, "walk_intensity": 0.8 if "brisk" in t else 0.6,
            "add_metformin": "metformin" in t, "sleep_hours": float(sleep.group(1)) if sleep else None,
            "summary": text[:60]}
