import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  Area, Bar, BarChart, CartesianGrid, ComposedChart, Legend, Line, LineChart, ReferenceArea,
  ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from "recharts";
import { api } from "./api";

const POLL_MS = 2000;
const C = {
  ink: "#14263B", muted: "#5E6E80", rule: "#D3DCE3", paper: "#EEF2F4", card: "#FFFFFF",
  range: "#1F8A7A", high: "#C7841C", low: "#C0392B", twin: "#5B4BC4", fm: "#2F7FC1", llm: "#B0478F",
};
const EXPERT = {
  mechanistic: { label: "Physiology twin", color: C.twin },
  foundation: { label: "Chronos foundation model", color: C.fm },
  llm: { label: "LLM forecaster", color: C.llm },
};

const hhmm = (m) => {
  const d = Math.floor(m / 1440) + 1;
  const h = Math.floor((m % 1440) / 60);
  const mm = m % 60;
  return `D${d} ${String(h).padStart(2, "0")}:${String(mm).padStart(2, "0")}`;
};
const zone = (g) => (g == null ? C.ink : g < 70 ? C.low : g > 180 ? C.high : C.range);
const pct = (x) => `${Math.round(x * 100)}%`;

export default function App() {
  const [patients, setPatients] = useState([]);
  const [pid, setPid] = useState("P001");
  const [s, setS] = useState(null);
  const [err, setErr] = useState(null);
  const [tab, setTab] = useState("chat");
  const [lastOk, setLastOk] = useState(null);
  const busy = useRef(false);

  useEffect(() => { api.patients().then(setPatients).catch((e) => setErr(e.message)); }, []);

  const refresh = useCallback(async () => {
    if (busy.current) return;
    busy.current = true;
    try {
      const d = await api.state(pid);
      setS(d); setErr(null); setLastOk(Date.now());
    } catch (e) {
      setErr(`Can't reach the twin server (${e.message}). Start it with: uvicorn server:app --port 8000`);
    } finally { busy.current = false; }
  }, [pid]);

  useEffect(() => {
    setS(null);
    refresh();
    const id = setInterval(() => { if (!document.hidden) refresh(); }, POLL_MS);
    return () => clearInterval(id);
  }, [refresh]);

  const clock = async (body) => { await api.clock({ pid, ...body }); refresh(); };
  const patient = patients.find((p) => p.id === pid);

  return (
    <div className="shell">
      <aside className="rail">
        <div className="brand">GlucoTwin<span>Bharat</span></div>
        <p className="tagline">A live virtual patient for type 2 diabetes.</p>
        <nav className="patients" aria-label="Patients">
          {patients.map((p) => (
            <button key={p.id} className={`patient ${p.id === pid ? "on" : ""}`} onClick={() => setPid(p.id)}>
              <strong>{p.name}</strong>
              <span>{p.age}{p.sex}, {p.city}</span>
              <span>{p.condition}</span>
            </button>
          ))}
        </nav>
        {patient && (
          <section className="moments">
            <h3>Jump to</h3>
            {patient.moments.map((m) => (
              <button key={m.minute} onClick={() => clock({ minute: m.minute, playing: false })}>{m.label}</button>
            ))}
            <p className="story">{patient.story}</p>
          </section>
        )}
      </aside>

      <main>
        {err && <div className="banner">{err}</div>}
        {!s ? <div className="loading">Building the twin…</div> : (
          <>
            <Monitor s={s} clock={clock} lastOk={lastOk} />
            <section className="grid">
              <div className="panel chart-panel">
                <LiveChart s={s} />
              </div>
              <div className="side">
                <Alerts s={s} />
                <Router s={s} />
              </div>
            </section>
            <div className="tabs" role="tablist">
              {[["chat", "Chat"], ["ask", "Ask the twin"], ["drift", "Twin vs record"], ["auto", "Sleep & heart"], ["record", "Health record"], ["brief", "Clinical brief"], ["learn", "Learning"], ["privacy", "Privacy & edge"]].map(([k, l]) => (
                <button key={k} role="tab" aria-selected={tab === k} className={tab === k ? "on" : ""} onClick={() => setTab(k)}>{l}</button>
              ))}
            </div>
            <div className="panel tabbody">
              {tab === "ask" && <AskTwin pid={pid} s={s} />}
              {tab === "drift" && <Drift s={s} />}
              {tab === "auto" && <Autonomic s={s} />}
              {tab === "record" && <Record s={s} />}
              {tab === "brief" && <Brief pid={pid} s={s} />}
              {tab === "chat" && <Chat key={pid} pid={pid} s={s} />}
              {tab === "learn" && <Learning pid={pid} s={s} />}
              {tab === "privacy" && <Privacy />}
            </div>
            <footer>
              Forecast model: {s.backend.foundation}. PPG: {s.backend.ppg}. LLM: {s.backend.llm.available ? `${s.backend.llm.model} (${s.backend.llm.calls} calls, $${s.backend.llm.cost_usd.toFixed(4)})` : "off — rule-based routing"}. Synthetic data only.
            </footer>
          </>
        )}
      </main>
    </div>
  );
}

function Monitor({ s, clock, lastOk }) {
  const h = s.history.cgm;
  const trend = h.length > 3 ? (h[h.length - 1] - h[h.length - 4]) / 15 : 0;
  const arrow = trend > 2 ? "⇈" : trend > 1 ? "↑" : trend > 0.5 ? "↗" : trend < -2 ? "⇊" : trend < -1 ? "↓" : trend < -0.5 ? "↘" : "→";
  const peak = Math.max(...s.forecast.median);
  const low = Math.min(...s.forecast.median);
  const [ago, setAgo] = useState(0);
  useEffect(() => { const id = setInterval(() => setAgo(lastOk ? Math.round((Date.now() - lastOk) / 1000) : 0), 1000); return () => clearInterval(id); }, [lastOk]);
  return (
    <header className="monitor">
      <div className="readout" style={{ color: zone(s.current) }} aria-live="polite">
        <span className="value">{s.current}</span>
        <span className="unit">mg/dL <b>{arrow}</b></span>
      </div>
      <dl className="stats">
        <div><dt>Next 2 h</dt><dd>{Math.round(low)}–{Math.round(peak)}</dd></div>
        <div><dt>Risk above 180</dt><dd style={{ color: s.p_spike > 0.5 ? C.high : C.ink }}>{pct(s.p_spike)}</dd></div>
        <div><dt>Risk below 70</dt><dd style={{ color: s.p_hypo > 0.3 ? C.low : C.ink }}>{pct(s.p_hypo)}</dd></div>
        <div><dt>Time in range, 24 h</dt><dd>{Math.round(s.cgm24.tir)}%</dd></div>
        <div><dt>GMI</dt><dd>{s.cgm24.gmi.toFixed(1)}%</dd></div>
      </dl>
      <div className="clock">
        <div className="now">{s.time}</div>
        <div className="controls">
          <button className="play" onClick={() => clock({ playing: !s.playing })}>{s.playing ? "Pause" : "Play week"}</button>
          <label>Speed
            <select value={s.speed} onChange={(e) => clock({ speed: +e.target.value })}>
              {[5, 15, 30, 60].map((v) => <option key={v} value={v}>{v} min / {s.tick_seconds}s</option>)}
            </select>
          </label>
          <button onClick={() => clock({ minute: s.minute - 60 })} aria-label="Back one hour">−1 h</button>
          <button onClick={() => clock({ minute: s.minute + 60 })} aria-label="Forward one hour">+1 h</button>
        </div>
        <div className={`live ${s.playing ? "on" : ""}`}>{s.playing ? "Live" : "Paused"} · updated {ago}s ago</div>
      </div>
    </header>
  );
}

function LiveChart({ s }) {
  const data = useMemo(() => {
    const rows = s.history.t.map((t, i) => ({ t, cgm: s.history.cgm[i] }));
    const last = rows[rows.length - 1];
    rows[rows.length - 1] = { ...last, median: last.cgm, band: [last.cgm, last.cgm] };
    s.forecast.t.forEach((t, k) => {
      const r = { t, median: s.forecast.median[k], band: [s.forecast.lo[k], s.forecast.hi[k]] };
      Object.entries(s.forecast.experts).forEach(([n, v]) => { r[n] = v[k]; });
      rows.push(r);
    });
    return rows;
  }, [s]);
  const ymax = Math.ceil((Math.max(260, ...data.map((d) => Math.max(d.cgm || 0, d.band ? d.band[1] : 0))) + 10) / 50) * 50;
  return (
    <>
      <div className="chart-head">
        <h2>Glucose, last 12 h and the twin's next 2 h</h2>
        <p>Shaded fan = calibrated 80% interval. Thin lines = individual experts before fusion.</p>
      </div>
      <ResponsiveContainer width="100%" height={340}>
        <ComposedChart data={data} margin={{ top: 10, right: 12, left: -8, bottom: 0 }}>
          <CartesianGrid stroke={C.rule} vertical={false} />
          <ReferenceArea y1={70} y2={180} fill={C.range} fillOpacity={0.07} />
          <ReferenceLine y={180} stroke={C.high} strokeDasharray="2 4" />
          <ReferenceLine y={70} stroke={C.low} strokeDasharray="2 4" />
          <XAxis dataKey="t" type="number" domain={["dataMin", "dataMax"]} tickFormatter={hhmm} stroke={C.muted} fontSize={11} tickCount={8} />
          <YAxis domain={[40, ymax]} ticks={[40, 70, 120, 180, 250, 300, 350, 400].filter((v) => v <= ymax)} stroke={C.muted} fontSize={11} />
          <Tooltip labelFormatter={hhmm} formatter={(v, n) => [Array.isArray(v) ? `${Math.round(v[0])}–${Math.round(v[1])}` : Math.round(v), EXPERT[n]?.label || { cgm: "CGM", median: "Twin forecast", band: "80% interval" }[n] || n]} />
          {s.meals.map((m) => (
            <ReferenceLine key={m.t} x={m.t} stroke={C.muted} strokeDasharray="1 3"
              label={{ value: `${m.label.split("(")[0].split("+")[0].trim()} · ${m.carbs}g`, position: "insideTopLeft", fontSize: 10, fill: C.muted, angle: -90, dx: 4, dy: 60 }} />
          ))}
          <ReferenceLine x={s.minute} stroke={C.ink} strokeWidth={1.5} label={{ value: "now", position: "top", fontSize: 11, fill: C.ink }} />
          <Area dataKey="band" stroke="none" fill={C.twin} fillOpacity={0.16} isAnimationActive={false} />
          {Object.keys(s.forecast.experts).map((n) => (
            <Line key={n} dataKey={n} stroke={EXPERT[n].color} strokeWidth={1} dot={false} strokeOpacity={0.7} isAnimationActive={false} />
          ))}
          <Line dataKey="median" stroke={C.twin} strokeWidth={2.5} strokeDasharray="6 3" dot={false} isAnimationActive={false} />
          <Line dataKey="cgm" stroke={C.ink} strokeWidth={2} dot={false} isAnimationActive={false} />
        </ComposedChart>
      </ResponsiveContainer>
    </>
  );
}

function Alerts({ s }) {
  return (
    <div className="panel alerts">
      <h2>Alerts</h2>
      {s.alerts.length === 0 && <p className="quiet">Nothing predicted in the next 2 hours.</p>}
      {s.alerts.map((a, i) => (
        <div key={i} className={`alert ${a.severity} ${a.type}`}>
          <strong>{{ spike: "Spike coming", hypo: "Hypoglycaemia risk", high: "High now", drift: "Twin divergence", autonomic: "Heart & sleep strain" }[a.type]}</strong>
          <p>{a.message}</p>
        </div>
      ))}
    </div>
  );
}

function Router({ s }) {
  const names = Object.keys(s.weights);
  return (
    <div className="panel router">
      <h2>Who the twin trusts right now</h2>
      {names.map((n) => (
        <div key={n} className="wrow">
          <div className="wlabel"><span className="dot" style={{ background: EXPERT[n].color }} />{EXPERT[n].label}<b>{pct(s.weights[n])}</b></div>
          <div className="wbar"><i style={{ width: pct(s.weights[n]), background: EXPERT[n].color }} /></div>
          <div className="wmeta">recent error {s.skill_mae[n].toFixed(1)} mg/dL · skill {pct(s.w_skill[n])} · router {pct(s.w_route[n])}</div>
        </div>
      ))}
      <p className="rationale"><span>{s.router.source}:</span> {s.router.rationale}</p>
    </div>
  );
}

const SUGGEST = [
  "What if he has biryani instead?",
  "Swap to millet roti and walk 20 minutes after",
  "20 minute brisk walk after the meal",
  "Add metformin",
  "What if she slept only 4 hours?",
  "Half portion of rice dal",
];

function AskTwin({ pid, s }) {
  const [q, setQ] = useState("");
  const [r, setR] = useState(null);
  const [busy, setBusy] = useState(false);
  const [e, setE] = useState(null);
  const ask = async (text) => {
    setBusy(true); setE(null); setQ(text);
    try {
      const res = await api.whatif({ pid, text });
      if (res.error) { setR(null); setE(res.error); } else setR(res);
    } catch (x) { setE(x.message); } finally { setBusy(false); }
  };
  const data = r ? r.t.map((t, i) => ({ t, A: r.A.curve[i], B: r.B.curve[i] })) : [];
  const ymax = r ? Math.ceil((Math.max(200, r.A.peak, r.B.peak) + 10) / 50) * 50 : 250;
  return (
    <div className="ask">
      <div className="askbar">
        <form onSubmit={(ev) => { ev.preventDefault(); if (q.trim()) ask(q); }}>
          <input value={q} onChange={(ev) => setQ(ev.target.value)} placeholder="Ask a what-if about the next 4 hours, from now" aria-label="What-if question" />
          <button disabled={busy}>{busy ? "Simulating…" : "Simulate"}</button>
        </form>
        <div className="chips">{SUGGEST.map((x) => <button key={x} onClick={() => ask(x)}>{x}</button>)}</div>
      </div>
      {e && <p className="banner">{e}</p>}
      {!r && !e && <p className="quiet">The physiology twin replays the next 4 hours under two plans, from {s.time}. A statistical model can't do this — it has never seen the patient eat millet instead of rice.</p>}
      {r && (
        <div className="askres">
          <ResponsiveContainer width="100%" height={260}>
            <LineChart data={data} margin={{ top: 10, right: 12, left: -8, bottom: 0 }}>
              <CartesianGrid stroke={C.rule} vertical={false} />
              <ReferenceArea y1={70} y2={180} fill={C.range} fillOpacity={0.07} />
              <XAxis dataKey="t" type="number" domain={["dataMin", "dataMax"]} tickFormatter={hhmm} fontSize={11} stroke={C.muted} />
              <YAxis domain={[40, ymax]} ticks={[40, 70, 120, 180, 250, 300, 350, 400].filter((v) => v <= ymax)} fontSize={11} stroke={C.muted} />
              <Tooltip labelFormatter={hhmm} formatter={(v) => Math.round(v)} />
              <Legend />
              <Line dataKey="A" name={r.A.plan.summary} stroke={C.muted} strokeWidth={2} dot={false} />
              <Line dataKey="B" name={r.B.plan.summary} stroke={C.twin} strokeWidth={2.5} dot={false} />
            </LineChart>
          </ResponsiveContainer>
          <table className="cmp">
            <thead><tr><th></th><th>{r.A.plan.summary}</th><th>{r.B.plan.summary}</th></tr></thead>
            <tbody>
              <tr><td>Peak</td><td>{Math.round(r.A.peak)}</td><td>{Math.round(r.B.peak)}</td></tr>
              <tr><td>Minutes above 180</td><td>{r.A.min_above_180}</td><td>{r.B.min_above_180}</td></tr>
              <tr><td>Minutes below 70</td><td>{r.A.min_below_70}</td><td>{r.B.min_below_70}</td></tr>
              <tr><td>Glucose load (iAUC, mg/dL·h)</td><td>{Math.round(r.A.iauc)}</td><td>{Math.round(r.B.iauc)}</td></tr>
            </tbody>
          </table>
          <p className="quiet">Simulated from {r.from}. Question read by {r.parser === "LLM" ? "the LLM" : "keyword rules"}.</p>
        </div>
      )}
    </div>
  );
}

function Drift({ s }) {
  return (
    <div>
      <p className="lead">Every two hours the twin re-fits itself to the CGM. If the patient drifts away from what their record says — illness, missed doses, steroids — these lines move before HbA1c ever would.</p>
      <ResponsiveContainer width="100%" height={260}>
        <LineChart data={s.drift} margin={{ top: 10, right: 12, left: -8, bottom: 0 }}>
          <CartesianGrid stroke={C.rule} vertical={false} />
          <XAxis dataKey="t" type="number" domain={["dataMin", "dataMax"]} tickFormatter={hhmm} fontSize={11} stroke={C.muted} />
          <YAxis yAxisId="g" fontSize={11} stroke={C.muted} />
          <YAxis yAxisId="s" orientation="right" domain={[0.4, 1.6]} fontSize={11} stroke={C.muted} />
          <ReferenceLine yAxisId="g" y={15} stroke={C.high} strokeDasharray="3 3" label={{ value: "divergence", fontSize: 10, fill: C.high, position: "insideTopRight" }} />
          <Tooltip labelFormatter={hhmm} />
          <Legend />
          <Line yAxisId="g" dataKey="dgb" name="Basal glucose vs record (mg/dL)" stroke={C.high} strokeWidth={2} dot={false} />
          <Line yAxisId="s" dataKey="theta" name="Insulin sensitivity vs record (×)" stroke={C.twin} strokeWidth={2} dot={false} type="stepAfter" />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

function Autonomic({ s }) {
  const n = s.autonomic.nights;
  const hasMorph = n.some((x) => x.ppg_embedding_drift != null);
  return (
    <div>
      <p className={`lead lvl-${s.autonomic.level}`}>{s.autonomic.message}</p>
      <div className="two">
        <ResponsiveContainer width="100%" height={220}>
          <LineChart data={n} margin={{ top: 10, right: 12, left: -8, bottom: 0 }}>
            <CartesianGrid stroke={C.rule} vertical={false} />
            <XAxis dataKey="date" fontSize={11} stroke={C.muted} />
            <YAxis yAxisId="a" fontSize={11} stroke={C.muted} />
            <YAxis yAxisId="b" orientation="right" fontSize={11} stroke={C.muted} />
            <Tooltip /><Legend />
            <Line yAxisId="a" dataKey="ppg_rmssd_ms" name="HRV from raw PPG (ms)" stroke={C.range} strokeWidth={2} />
            <Line yAxisId="a" dataKey="rmssd_ms" name="HRV reported by watch (ms)" stroke={C.range} strokeDasharray="4 3" strokeWidth={1} dot={false} />
            <Line yAxisId="b" dataKey="resting_hr" name="Resting HR (bpm)" stroke={C.low} strokeWidth={2} />
          </LineChart>
        </ResponsiveContainer>
        <ResponsiveContainer width="100%" height={220}>
          <BarChart data={n} margin={{ top: 10, right: 12, left: -8, bottom: 0 }}>
            <CartesianGrid stroke={C.rule} vertical={false} />
            <XAxis dataKey="date" fontSize={11} stroke={C.muted} />
            <YAxis fontSize={11} stroke={C.muted} domain={[0, 9]} />
            <ReferenceLine y={7} stroke={C.muted} strokeDasharray="3 3" />
            <Tooltip /><Legend />
            <Bar dataKey="sleep_hours" name="Sleep (h)" fill={C.twin} />
          </BarChart>
        </ResponsiveContainer>
      </div>
      <h4>Pulse waveform</h4>
      <p className="quiet">Read by {s.ppg.backend}.</p>
      <div className="two">
        <div>
          <ResponsiveContainer width="100%" height={180}>
            <LineChart data={s.ppg.wave} margin={{ top: 10, right: 12, left: -8, bottom: 0 }}>
              <CartesianGrid stroke={C.rule} vertical={false} />
              <XAxis dataKey="t" type="number" domain={[0, 5]} tickFormatter={(v) => `${v}s`} fontSize={11} stroke={C.muted} />
              <YAxis hide />
              <Line dataKey="v" stroke={C.low} strokeWidth={1.5} dot={false} isAnimationActive={false} />
            </LineChart>
          </ResponsiveContainer>
          <p className="quiet">Five seconds of raw PPG from last night. Each beat has a forward wave and a reflected wave; stiffer, tighter arteries bring the reflection earlier and higher.</p>
        </div>
        {hasMorph ? (
          <div>
            <ResponsiveContainer width="100%" height={180}>
              <BarChart data={n} margin={{ top: 10, right: 12, left: -8, bottom: 0 }}>
                <CartesianGrid stroke={C.rule} vertical={false} />
                <XAxis dataKey="date" fontSize={11} stroke={C.muted} />
                <YAxis fontSize={11} stroke={C.muted} />
                <ReferenceLine y={3} stroke={C.high} strokeDasharray="3 3" label={{ value: "flag", fontSize: 10, fill: C.high, position: "insideTopRight" }} />
                <Tooltip />
                <Bar dataKey="ppg_embedding_drift" name="Waveform-shape drift (SD from baseline)" fill={C.low} />
              </BarChart>
            </ResponsiveContainer>
            <p className="quiet">PaPaGei turns each night's pulse into a 512-number fingerprint. Bars show how far tonight's fingerprint has moved from this patient's first three nights.</p>
          </div>
        ) : (
          <p className="quiet">Add the PaPaGei weights (scripts/get_papagei.sh) to see nightly waveform-shape drift. HR and HRV above are already computed from the raw pulse.</p>
        )}
      </div>
    </div>
  );
}

function Record({ s }) {
  const e = s.ehr;
  return (
    <div className="record">
      <div>
        <h3>{e.demographics.name}, {e.demographics.age}{e.demographics.sex}</h3>
        <p>BMI {e.bmi} · waist {e.demographics.waist_cm} cm · {e.lifestyle.smoker ? "smoker" : "non-smoker"} · {e.lifestyle.physical_activity}</p>
        <ul>{e.conditions.map((c) => <li key={c.code}>{c.name} <span className="code">{c.code}</span>, since {c.since}</li>)}</ul>
        <h4>Medication</h4>
        <ul>{e.medications.length ? e.medications.map((m) => <li key={m.name}>{m.name} {m.dose}</li>) : <li>None</li>}</ul>
        <h4>Genetics</h4>
        <p>TCF7L2 rs7903146 {e.genetics.TCF7L2_rs7903146} · T2D polygenic score {e.genetics.t2d_prs_percentile}th percentile</p>
      </div>
      <div>
        <h4>Labs</h4>
        <table className="cmp"><tbody>
          {[["HbA1c", `${e.labs.hba1c}%`], ["Fasting glucose", e.labs.fasting_glucose], ["Fasting insulin", e.labs.fasting_insulin], ["LDL / HDL", `${e.labs.ldl} / ${e.labs.hdl}`], ["Triglycerides", e.labs.triglycerides], ["eGFR", e.labs.egfr], ["BP", `${e.vitals.sbp}/${e.vitals.dbp}`]].map(([k, v]) => <tr key={k}><td>{k}</td><td>{v}</td></tr>)}
        </tbody></table>
        <h4>Risk</h4>
        <p>10-year cardiovascular risk (Framingham): <b>{pct(s.framingham_10y)}</b><br />Indian Diabetes Risk Score: <b>{s.idrs}</b>{s.idrs >= 60 ? " (high)" : ""}</p>
      </div>
      <div>
        <h4>How the record built the twin</h4>
        <ul>{e.twin_notes.map((n) => <li key={n}>{n}</li>)}</ul>
        <h4>Week so far</h4>
        <p>Time in range {Math.round(s.cgm_all.tir)}% · above 180 {Math.round(s.cgm_all.tar)}% · below 70 {s.cgm_all.tbr.toFixed(1)}% · CV {Math.round(s.cgm_all.cv)}%</p>
      </div>
    </div>
  );
}

function Brief({ pid, s }) {
  const [txt, setTxt] = useState(s.brief);
  const [busy, setBusy] = useState(false);
  useEffect(() => { setTxt(s.brief); }, [s.minute]); // eslint-disable-line
  const go = async () => { setBusy(true); try { setTxt((await api.brief(pid)).brief); } finally { setBusy(false); } };
  return (
    <div>
      <button className="play" onClick={go} disabled={busy}>{busy ? "Writing…" : `Write brief for ${s.time}`}</button>
      {txt ? <pre className="brief">{txt}</pre> : <p className="quiet">The LLM reads every expert's output and writes a note a doctor can act on.</p>}
    </div>
  );
}

const CHAT_SUGGEST = [
  "How is the patient doing right now?",
  "Why did the twin raise this alert?",
  "What if they had millet roti instead?",
  "How has sleep and HRV changed this week?",
  "Which of the medicines explains the low?",
];

function Chat({ pid, s }) {
  const [msgs, setMsgs] = useState([]);
  const [q, setQ] = useState("");
  const [busy, setBusy] = useState(false);
  const end = useRef(null);
  useEffect(() => { end.current?.scrollIntoView({ block: "nearest" }); }, [msgs]);
  const send = async (text) => {
    if (!text.trim() || busy) return;
    const hist = msgs.map(({ role, content }) => ({ role, content }));
    setMsgs((m) => [...m, { role: "user", content: text }]);
    setQ(""); setBusy(true);
    try {
      const r = await api.chat(pid, text, hist);
      setMsgs((m) => [...m, { role: "assistant", content: r.reply, sim: r.simulation, llm: r.llm }]);
    } catch (e) {
      setMsgs((m) => [...m, { role: "assistant", content: `The twin server didn't answer (${e.message}).` }]);
    } finally { setBusy(false); }
  };
  return (
    <div className="chat">
      <p className="quiet">Ask anything about {s.ehr.demographics.name} as of {s.time}. Answers use the live twin: record, glucose, forecast, alerts, sleep, PPG and meals. Names and IDs are replaced before anything reaches the language model.</p>
      <div className="msgs" aria-live="polite">
        {msgs.length === 0 && <div className="chips">{CHAT_SUGGEST.map((x) => <button key={x} onClick={() => send(x)}>{x}</button>)}</div>}
        {msgs.map((m, i) => (
          <div key={i} className={`msg ${m.role}`}>
            <div className="bubble">{m.content}</div>
            {m.sim && (
              <div className="simcard">
                <span>{m.sim.A}: peak <b>{m.sim.A_peak}</b>, {m.sim.A_min_above_180} min above 180</span>
                <span>{m.sim.B}: peak <b>{m.sim.B_peak}</b>, {m.sim.B_min_above_180} min above 180</span>
              </div>
            )}
          </div>
        ))}
        {busy && <div className="msg assistant"><div className="bubble quiet">Reading the twin…</div></div>}
        <div ref={end} />
      </div>
      <form className="askbar-row" onSubmit={(e) => { e.preventDefault(); send(q); }}>
        <input value={q} onChange={(e) => setQ(e.target.value)} placeholder="Ask about this patient" aria-label="Message" />
        <button disabled={busy}>Send</button>
      </form>
    </div>
  );
}

function Learning({ pid, s }) {
  const [d, setD] = useState(null);
  useEffect(() => { api.learning(pid).then(setD).catch(() => {}); }, [pid, s.minute]);
  const a = s.adapter;
  return (
    <div className="learn">
      <div>
        <h3>On-device online learning</h3>
        <p className="lead">A 192-weight adapter sits on top of the frozen models and learns this patient's own forecast error. Every forecast that matures (its real glucose arrives 2 h later) updates it once, on the device, by recursive least squares. If it stops helping, it switches itself off.</p>
        <table className="cmp"><tbody>
          <tr><td>Updates so far</td><td>{a.updates}</td></tr>
          <tr><td>Status</td><td>{a.active ? "Correcting forecasts" : a.updates < 36 ? "Warming up" : "Paused — not helping right now"}</td></tr>
          <tr><td>Recent error without adapter</td><td>{a.mae_without != null ? `${a.mae_without.toFixed(1)} mg/dL` : "—"}</td></tr>
          <tr><td>Recent error with adapter</td><td>{a.mae_with != null ? `${a.mae_with.toFixed(1)} mg/dL` : "—"}</td></tr>
          <tr><td>Started from federated prior</td><td>{a.from_federated_prior ? "Yes" : "No"}</td></tr>
          <tr><td>Learned mostly from</td><td>{a.top_features.join(", ") || "—"}</td></tr>
        </tbody></table>
      </div>
      <div>
        <h3>Federated learning</h3>
        <p className="lead">Each phone shares only its 192 adapter weights — never glucose, never identity. The server averages them (FedAvg) with clipping and noise for differential privacy, and new patients start from that shared prior.</p>
        {d?.federated_report ? <pre className="brief">{d.federated_report}</pre> : <p className="quiet">Run <code>python scripts/federated_demo.py</code> to simulate 15 devices and create the shared prior.</p>}
      </div>
    </div>
  );
}

function Privacy() {
  const [d, setD] = useState(null);
  useEffect(() => {
    const go = () => api.privacy().then(setD).catch(() => {});
    go(); const id = setInterval(go, 4000); return () => clearInterval(id);
  }, []);
  if (!d) return <p className="quiet">Loading…</p>;
  return (
    <div className="learn">
      <div>
        <h3>{d.mode}</h3>
        <p className="lead">Everything below runs on this machine. Only the language model can be remote, and it only ever sees de-identified text.</p>
        <ul className="plain">{d.on_device.map((x) => <li key={x}>{x}</li>)}</ul>
        <table className="cmp"><tbody>
          <tr><td>LLM endpoint</td><td className="mono">{d.llm_endpoint}</td></tr>
          <tr><td>Payloads sent</td><td>{d.stats.payloads}</td></tr>
          <tr><td>Identifiers removed</td><td>{d.stats.redactions}</td></tr>
        </tbody></table>
        <p className="quiet">Set LLM_BASE_URL to a local Ollama or llama.cpp server and nothing leaves the device at all.</p>
      </div>
      <div>
        <h3>What actually left the device</h3>
        {d.recent.length === 0 && <p className="quiet">No LLM calls yet. Ask the chat something and the exact outgoing text appears here.</p>}
        {d.recent.map((r, i) => (
          <details key={i} className="egress">
            <summary><b>{r.purpose}</b> · {r.ts} · removed: {r.removed.length ? r.removed.join(", ") : "nothing to remove"}</summary>
            <pre>{r.payload.map((m) => `[${m.role}] ${m.content}`).join("\n\n").slice(0, 2500)}</pre>
          </details>
        ))}
      </div>
    </div>
  );
}
