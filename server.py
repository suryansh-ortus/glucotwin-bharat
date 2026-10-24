"""FastAPI backend for the GlucoTwin doctor dashboard.

A replay clock plays each patient's synthetic week forward in "real time"
(default: 5 simulated minutes every 2 s). The React UI polls /api/state and
re-renders, so alerts appear the moment the twin predicts them.

Run:  uvicorn server:app --reload --port 8000
"""
from __future__ import annotations

import asyncio
import os
from contextlib import asynccontextmanager

import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

load_dotenv()

from twin.ehr import PERSONAS, bmi  # noqa: E402
from twin.engine import TwinEngine  # noqa: E402
from twin import chat as twinchat  # noqa: E402
from twin.experts.foundation import FoundationExpert  # noqa: E402
from twin.privacy import GATEWAY  # noqa: E402
from twin.experts.ppg import FS as PPG_FS, PPGExpert  # noqa: E402
from twin.llm import LLM  # noqa: E402
from twin.meals import MEALS  # noqa: E402
from twin.wearables import START, simulate_week  # noqa: E402

TICK_SECONDS = float(os.getenv("TICK_SECONDS", "2"))
END = 7 * 1440 - 125

# demo moments: (label, minute)
MOMENTS = {
    "P001": [("Day 2 09:00 - normal day", 1 * 1440 + 9 * 60),
             ("Day 4 22:35 - late biryani after 2 short nights", 3 * 1440 + 22 * 60 + 35),
             ("Day 6 20:00 - hidden viral illness (twin divergence)", 5 * 1440 + 20 * 60)],
    "P002": [("Day 2 13:05 - normal lunch", 1 * 1440 + 13 * 60 + 5),
             ("Day 4 08:30 - vrat fast on glimepiride", 3 * 1440 + 8 * 60 + 30),
             ("Day 4 10:45 - temple walk, no food", 3 * 1440 + 10 * 60 + 45)],
    "P003": [("Day 2 09:20 - dosa breakfast", 1 * 1440 + 9 * 60 + 20),
             ("Day 6 12:00 - release-week strain", 5 * 1440 + 12 * 60)],
}

STATE: dict = {}


def _engine(pid: str):
    if pid not in PERSONAS:
        raise HTTPException(404, f"unknown patient {pid}")
    if pid not in STATE:
        bundle = simulate_week(PERSONAS[pid], seed=1)
        STATE[pid] = {"engine": TwinEngine(bundle, llm=LLM_, foundation=FM_, ppg=PPG_, adapter_prior=PRIOR_),
                      "minute": MOMENTS[pid][0][1], "playing": False, "speed": 5,
                      "snap_cache": {}, "drift": {}, "brief": {}}
    return STATE[pid]


async def _ticker():
    while True:
        await asyncio.sleep(TICK_SECONDS)
        for st in STATE.values():
            if st["playing"]:
                st["minute"] = min(st["minute"] + st["speed"], END)
                if st["minute"] >= END:
                    st["playing"] = False


@asynccontextmanager
async def lifespan(_app):
    global FM_, LLM_, PPG_, PRIOR_
    if os.path.exists(PRIOR_PATH):
        PRIOR_ = np.load(PRIOR_PATH)
        print(f"[twin] federated adapter prior loaded ({PRIOR_.size} weights)")
    FM_ = FoundationExpert()
    LLM_ = LLM()
    PPG_ = PPGExpert()
    print(f"[twin] PPG: {PPG_.backend} {('(' + PPG_.error + ')') if PPG_.error else ''}")
    print(f"[twin] foundation: {FM_.backend} {('(' + FM_.error + ')') if FM_.error else ''}")
    print(f"[twin] LLM: {(('local ' + LLM_.base_url) if LLM_.local else 'DeepInfra') + ' ' + LLM_.model if LLM_.available else 'off (no DEEPINFRA_API_KEY) - rule fallbacks'} | privacy gateway on")
    for pid in PERSONAS:
        _engine(pid)
    task = asyncio.create_task(_ticker())
    yield
    task.cancel()


FM_: FoundationExpert | None = None
LLM_: LLM | None = None
PPG_: PPGExpert | None = None
PRIOR_ = None
PRIOR_PATH = os.path.join(os.path.dirname(__file__), "results", "federated_prior.npy")
app = FastAPI(title="GlucoTwin-Bharat", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


def _l(a):
    return [round(float(x), 1) for x in np.asarray(a)]


def _clean(df) -> list[dict]:
    """DataFrame -> JSON-safe records (NaN -> null)."""
    return [{k: (None if isinstance(v, float) and not np.isfinite(v) else v) for k, v in r.items()}
            for r in df.to_dict("records")]


def _ts(minute: int) -> str:
    return (START + np.timedelta64(int(minute), "m")).strftime("%a %d %b %H:%M")


@app.get("/api/patients")
def patients():
    out = []
    for pid, e in PERSONAS.items():
        d = e["demographics"]
        out.append({"id": pid, "name": d["name"], "age": d["age"], "sex": d["sex"], "city": d["city"],
                    "condition": e["conditions"][0]["name"], "story": e["story"],
                    "moments": [{"label": l, "minute": m} for l, m in MOMENTS[pid]]})
    return out


@app.get("/api/state")
def state(pid: str):
    st = _engine(pid)
    eng: TwinEngine = st["engine"]
    o = int(st["minute"] // 5 * 5)
    if o not in st["snap_cache"]:
        st["snap_cache"][o] = eng.snapshot(o)
    s = st["snap_cache"][o]
    b = eng.b
    i = o // 5
    lo = max(i - 143, 0)                     # last 12 h of history
    df = b["df"].iloc[lo: i + 1]
    # drift history every 2 h, computed incrementally
    for t in range(1440, o + 1, 120):
        if t not in st["drift"]:
            th, _, _, dgb = eng.mech.assimilate(t)
            st["drift"][t] = (th, dgb)
    nt = s["autonomic"]["table"]
    ehr = b["ehr"]
    return {
        "pid": pid, "minute": o, "time": _ts(o), "playing": st["playing"], "speed": st["speed"],
        "tick_seconds": TICK_SECONDS,
        "history": {"t": [int(x) for x in df["minute"]], "cgm": _l(df["cgm"]), "hr": _l(df["hr"]),
                    "steps": _l(df["steps"]), "sleep": list(df["sleep_stage"])},
        "forecast": {"t": [o + 5 * (k + 1) for k in range(24)], "median": _l(s["forecast"]["median"]),
                     "lo": _l(s["forecast"]["lo"]), "hi": _l(s["forecast"]["hi"]),
                     "experts": {k: _l(v) for k, v in s["forecast"]["experts"].items()}},
        "meals": [{"t": m["t"], "label": m["label"], "carbs": m["carbs_logged"], "gi": m["gi"]}
                  for m in b["meal_log"] if o - 720 <= m["t"] <= o],
        "current": round(s["current"]), "p_spike": s["p_spike"], "p_hypo": s["p_hypo"], "alerts": s["alerts"],
        "weights": s["weights"], "w_skill": s["w_skill"], "w_route": s["w_route"], "skill_mae": s["skill_mae"],
        "router": s["router"], "theta": s["theta"], "dgb": s["dgb"],
        "drift": [{"t": t, "label": _ts(t), "theta": v[0], "dgb": round(v[1], 1)} for t, v in sorted(st["drift"].items())],
        "cgm24": s["cgm"], "cgm_all": s["cgm_all"],
        "framingham_10y": s["framingham_10y"], "idrs": s["idrs"],
        "ppg": _ppg_payload(eng, o),
        "adapter": s["adapter"],
        "privacy": {"mode": "on-device LLM" if LLM_.local else "cloud LLM via privacy gateway",
                    "payloads": GATEWAY.stats["payloads"], "redactions": GATEWAY.stats["redactions"]},
        "autonomic": {"level": s["autonomic"]["level"], "message": s["autonomic"]["message"],
                      "hrv_source": s["autonomic"].get("hrv_source", "wearable"),
                      "nights": _clean(nt.drop(columns=["ill_hidden"], errors="ignore").round(2))},
        "ehr": {**ehr, "bmi": round(bmi(ehr), 1), "twin_notes": eng.mech.base.notes},
        "backend": {"foundation": s["foundation_backend"], "ppg": s.get("ppg_backend"), "llm": s["llm"]},
        "brief": st["brief"].get(o),
    }


def _ppg_payload(eng: TwinEngine, o: int) -> dict:
    """Last completed night's 10-s raw pulse waveform for the dashboard."""
    night = max(o // 1440 - 1, 0)          # the night that ended this morning
    x = eng.ppg_samples.get(night)
    if x is None:
        return {"backend": eng.ppg.backend, "wave": []}
    x = (x - x.mean()) / (x.std() + 1e-8)
    step = 2
    return {"backend": eng.ppg.backend, "night": night,
            "wave": [{"t": round(i * step / PPG_FS, 3), "v": round(float(v), 3)} for i, v in enumerate(x[::step][: 5 * PPG_FS // step])]}


class ClockReq(BaseModel):
    pid: str
    minute: int | None = None
    playing: bool | None = None
    speed: int | None = None


@app.post("/api/clock")
def clock(r: ClockReq):
    st = _engine(r.pid)
    if r.minute is not None:
        st["minute"] = int(min(max(r.minute, 1440 + 120), END))
    if r.playing is not None:
        st["playing"] = r.playing
    if r.speed is not None:
        st["speed"] = int(min(max(r.speed, 5), 60))
    return {"minute": st["minute"], "playing": st["playing"], "speed": st["speed"]}


class BriefReq(BaseModel):
    pid: str


@app.post("/api/brief")
def brief(r: BriefReq):
    from twin.llm import clinical_brief
    st = _engine(r.pid)
    o = int(st["minute"] // 5 * 5)
    eng = st["engine"]
    s = st["snap_cache"].get(o) or eng.snapshot(o)
    txt = clinical_brief(eng.llm, eng.brief_payload(s))
    st["brief"][o] = txt
    return {"brief": txt, "llm": eng.llm.available}


class WhatIfReq(BaseModel):
    pid: str
    text: str = ""
    plan_a: dict | None = None
    plan_b: dict | None = None


@app.post("/api/whatif")
def whatif(r: WhatIfReq):
    return _run_whatif(r)


def _run_whatif(r: WhatIfReq):
    st = _engine(r.pid)
    eng: TwinEngine = st["engine"]
    o = int(st["minute"] // 5 * 5)
    plan_b = r.plan_b or eng.parse(r.text)
    if not r.plan_b and not (plan_b.get("meal_key") or plan_b.get("walk_minutes") or plan_b.get("add_metformin")
                             or plan_b.get("sleep_hours") is not None):
        return {"error": "That doesn't describe a change the twin can simulate. Name a food, a walk, a drug or "
                         "sleep, e.g. \"swap to millet roti\", \"20 minute walk after the meal\", \"add metformin\"."}
    # If a meal was logged in the last hour, rewind to just before it: "what if he had eaten X instead?"
    recent = [m for m in eng.b["meal_log"] if o - 60 <= m["t"] <= o]
    if recent:
        ref, ref_kind = recent[-1], "Actual"
        o = int((ref["t"] - 5) // 5 * 5)
    else:  # otherwise use what this patient usually eats around this time of day
        tod = o % 1440
        hist = [m for m in eng.b["meal_log"] if abs((m["t"] % 1440) - tod) <= 90 and m["t"] < o - 60]
        ref, ref_kind = (hist[-1] if hist else None), "Usual"
    if not plan_b.get("meal_key") and ref and not r.plan_b:
        # no food named ("walk after dinner", "add metformin") -> run it on that meal
        plan_b.update(meal_key=ref["key"], carbs=float(ref["carbs_logged"]), gi=ref["gi"])
    if r.plan_a:
        plan_a = r.plan_a
    else:
        plan_a = {**plan_b, "walk_minutes": 0, "add_metformin": False, "sleep_hours": None}
        if ref:
            plan_a.update(meal_key=ref["key"], carbs=float(ref["carbs_logged"]), gi=ref["gi"])
        plan_a["summary"] = f"{ref_kind}: {ref['label']}" if ref else "No change"
    res = eng.whatif(o, plan_a, plan_b)
    return {
        "t": list(range(o + 1, o + 1 + len(res["A"]["curve"]), 1))[::5],
        "A": {**{k: v for k, v in res["A"].items() if k != "curve"}, "curve": _l(res["A"]["curve"][::5])},
        "B": {**{k: v for k, v in res["B"].items() if k != "curve"}, "curve": _l(res["B"]["curve"][::5])},
        "parser": plan_b.get("parser", "manual"), "from": _ts(o),
    }


class ChatReq(BaseModel):
    pid: str
    message: str
    history: list[dict] = []


@app.post("/api/chat")
def chat(r: ChatReq):
    st = _engine(r.pid)
    eng: TwinEngine = st["engine"]
    o = int(st["minute"] // 5 * 5)
    snap = st["snap_cache"].get(o) or eng.snapshot(o)
    st["snap_cache"][o] = snap

    def sim(text):
        return twinchat.summarise_whatif(_run_whatif(WhatIfReq(pid=r.pid, text=text)))

    return twinchat.answer(eng, snap, r.history, r.message, whatif_fn=sim)


@app.get("/api/privacy")
def privacy():
    return {"mode": "on-device LLM (nothing leaves this machine)" if LLM_.local else "cloud LLM via privacy gateway",
            "llm_endpoint": LLM_.base_url, "stats": GATEWAY.stats, "recent": GATEWAY.recent[::-1][:8],
            "on_device": ["Physiology twin + data assimilation", f"Forecaster: {FM_.backend}", f"PPG: {PPG_.backend}",
                          "Online adapter (RLS)", "Risk scores, alerts, what-if simulator"]}


@app.get("/api/learning")
def learning(pid: str):
    st = _engine(pid)
    fed = None
    p = os.path.join(os.path.dirname(__file__), "results", "federated.txt")
    if os.path.exists(p):
        fed = open(p).read()
    return {"adapter": st["engine"].adapter.summary(), "federated_prior": PRIOR_ is not None, "federated_report": fed}


@app.get("/api/meals")
def meals():
    return [{"key": k, **v} for k, v in MEALS.items() if k != "none"]


_dist = os.path.join(os.path.dirname(__file__), "web", "dist")
if os.path.isdir(_dist):
    app.mount("/", StaticFiles(directory=_dist, html=True), name="web")
