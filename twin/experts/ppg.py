"""Expert 6 - PPG (photoplethysmography) expert using PaPaGei open pretrained weights.

PaPaGei (Pillai et al., ICLR 2025, Nokia Bell Labs) is the first open foundation model
for PPG: a 5.8M-parameter 1-D ResNet with mixture-of-experts heads, pretrained on
57,000 h / 20M PPG segments from public datasets, and benchmarked for robustness across
skin tones. Weights (BSD-3-Clause-Clear): https://zenodo.org/records/13983110
Download with:  bash scripts/get_papagei.sh   ->  weights/papagei_s.pt

What we use, with NO training / no linear probes:
  * 512-d embedding per 10-s segment -> nightly "pulse-morphology drift":
    cosine distance of tonight's mean embedding from the patient's own first 3 nights.
    The waveform shape changes with vascular tone, stiffness, sympathetic drive,
    infection and dehydration, before resting HR does.
  * the two pretrained regression heads: IPA (inflection-point-area ratio, a vascular
    tone marker) and SQI (signal quality) - PaPaGei-S was trained to predict both.
Classical beat detection on the same raw signal gives HR and RMSSD (HRV) - this replaces
the wearable vendor's black-box HRV number with one we compute from the raw waveform.

Raw PPG here is synthetic (two-wave pulse model: forward + reflected wave). Arterial
stiffness moves the reflected wave earlier and higher, as in real physiology. Swap
`synth_ppg` for real watch / ring PPG exports without touching anything else.

Without torch or the weight file, the classical HR/HRV path still runs.
"""
from __future__ import annotations

import os

import numpy as np
from scipy.signal import butter, filtfilt, find_peaks, resample
from scipy.stats import skew

FS = 125            # PaPaGei input rate (Hz)
SEG_S = 10          # PaPaGei segment length (s)
WIN_S = 60          # length of each simulated overnight recording window (s)
WINDOWS_PER_NIGHT = 6


# ------------------------------------------------------------------ synthetic raw PPG
def synth_ppg(hr: float, rmssd_ms: float, stiffness: float, seconds: int, rng: np.random.Generator,
              fs: int = FS, motion: float = 0.0) -> np.ndarray:
    """Two-Gaussian pulse model. stiffness 0..1: higher -> earlier, larger reflected wave."""
    rr_mean = 60.0 / hr
    s = rmssd_ms / 1000.0 / np.sqrt(2)
    n_beats = int(seconds / rr_mean) + 3
    rr = rr_mean + rng.normal(0, s, n_beats) + 0.006 * np.sin(2 * np.pi * 0.25 * np.cumsum(np.full(n_beats, rr_mean)))
    rr = np.clip(rr, 0.3, 2.0)
    beats = np.cumsum(rr) - rr[0]
    t = np.arange(int(seconds * fs)) / fs
    x = np.zeros_like(t)
    refl_delay = 0.38 - 0.16 * stiffness           # s after systolic peak
    refl_amp = 0.35 + 0.45 * stiffness
    for b, r in zip(beats, rr):
        tp = b + 0.18 * rr_mean
        x += np.exp(-((t - tp) ** 2) / (2 * (0.055 * r) ** 2))
        x += refl_amp * np.exp(-((t - tp - refl_delay * min(1.0, r / 0.85)) ** 2) / (2 * (0.08 * r) ** 2))
    x += 0.15 * np.sin(2 * np.pi * 0.25 * t + rng.uniform(0, 6.28))           # respiration
    x += rng.normal(0, 0.02, len(t))
    if motion > 0:                                                              # motion artefact burst
        k = rng.integers(0, len(t) - fs * 3)
        x[k: k + fs * 3] += motion * rng.normal(0, 1, fs * 3)
    return x


def _bandpass(x, fs=FS, lo=0.5, hi=12.0, order=4):
    b, a = butter(order, [lo / (fs / 2), hi / (fs / 2)], btype="band")
    return filtfilt(b, a, x)


# ------------------------------------------------------------------ classical HR / HRV
def beats_hrv(x: np.ndarray, fs: int = FS) -> dict:
    y = _bandpass(x, fs)
    # dominant pulse frequency (0.7-3 Hz) sets the minimum peak spacing, so the
    # reflected / dicrotic wave is not counted as a second beat
    f = np.fft.rfftfreq(len(y), 1 / fs)
    p = np.abs(np.fft.rfft(y - y.mean())) ** 2
    band = (f >= 0.6) & (f <= 3.6)
    fb, pb = f[band], p[band]
    loc, _ = find_peaks(pb, height=0.12 * pb.max())
    f0 = fb[loc[0]] if len(loc) else fb[np.argmax(pb)]          # lowest strong peak = fundamental
    peaks, _ = find_peaks(y, distance=int(0.65 * fs / f0), prominence=0.4 * np.std(y))
    rr = np.diff(peaks) / fs
    rr = rr[(rr > 0.33) & (rr < 1.6)]
    if len(rr) < 5:
        return {"hr": np.nan, "rmssd": np.nan, "sqi": float(skew(y))}
    med = np.median(rr)
    rr = rr[np.abs(rr - med) < 0.25 * med]                                     # ectopic / artefact filter
    d = np.diff(rr)
    return {"hr": float(60.0 / rr.mean()), "rmssd": float(np.sqrt(np.mean(d ** 2)) * 1000), "sqi": float(skew(y))}


# ------------------------------------------------------------------ PaPaGei
class PPGExpert:
    name = "ppg"

    def __init__(self, weights: str | None = None):
        self.path = weights or os.getenv("PPG_MODEL_PATH") or os.path.join(
            os.path.dirname(__file__), "..", "..", "weights", "papagei_s.pt")
        self.path = os.path.abspath(self.path)
        self.model = None
        self.backend = "classical beat detection (PaPaGei weights not found)"
        self.error = None
        if os.getenv("DISABLE_PAPAGEI") == "1":
            self.error = "disabled via DISABLE_PAPAGEI=1"
            return
        if not os.path.exists(self.path):
            self.error = f"no weights at {self.path} - run scripts/get_papagei.sh"
            return
        try:
            import torch
            from .papagei_net import ResNet1DMoE

            m = ResNet1DMoE(in_channels=1, base_filters=32, kernel_size=3, stride=2, groups=1,
                            n_block=18, n_classes=512, n_experts=3)
            sd = torch.load(self.path, map_location="cpu", weights_only=True)
            if isinstance(sd, dict) and "state_dict" in sd:
                sd = sd["state_dict"]
            sd = {(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()}
            m.load_state_dict(sd)
            self.model = m.eval()
            self._torch = torch
            self.backend = "PaPaGei-S (Nokia Bell Labs, ICLR 2025) + classical beat detection"
        except Exception as e:  # noqa: BLE001
            self.error = f"{type(e).__name__}: {e}"

    @property
    def available(self) -> bool:
        return self.model is not None

    def embed(self, x: np.ndarray, fs: int = FS):
        """x: raw PPG at fs. Returns (embeddings [n_seg, 512], ipa [n_seg], sqi [n_seg])."""
        y = _bandpass(x, fs)
        seg = fs * SEG_S
        n = len(y) // seg
        segs = y[: n * seg].reshape(n, seg)
        if fs != FS:
            segs = resample(segs, FS * SEG_S, axis=-1)
        segs = (segs - segs.mean(1, keepdims=True)) / (segs.std(1, keepdims=True) + 1e-8)
        torch = self._torch
        with torch.inference_mode():
            emb, ipa, sqi, _ = self.model(torch.tensor(segs, dtype=torch.float32).unsqueeze(1))
        return emb.numpy(), ipa.numpy().ravel(), sqi.numpy().ravel()

    # ------------------------------------------------------------------
    def analyse_week(self, nightly, ehr: dict, seed: int = 0):
        """Simulate overnight raw PPG for each night, run the expert, add ppg_* columns.
        Returns (nightly_with_ppg, sample_waveform_from_last_night)."""
        rng = np.random.default_rng(seed + 77)
        nt = nightly.copy()
        age = ehr["demographics"]["age"]
        rhr0 = float(nt["resting_hr"].iloc[0])
        stiff_base = float(np.clip(0.15 + 0.008 * (age - 30), 0.1, 0.7))
        cols = {k: [] for k in ("ppg_hr", "ppg_rmssd_ms", "ppg_sqi", "ppg_ipa", "ppg_embedding_drift")}
        embs, samples = [], {}
        for _, r in nt.iterrows():
            # hidden vascular state: illness / sympathetic load stiffens and constricts vessels
            stiff = stiff_base + 0.015 * max(r["resting_hr"] - rhr0, 0) + (0.2 if r.get("ill_hidden") else 0.0)
            stiff = float(np.clip(stiff, 0.05, 0.95))
            xs = [synth_ppg(r["resting_hr"] - 4, r["rmssd_ms"], stiff, WIN_S, rng,
                            motion=1.5 if rng.random() < 0.15 else 0.0) for _ in range(WINDOWS_PER_NIGHT)]
            m = [beats_hrv(x) for x in xs]
            cols["ppg_hr"].append(round(float(np.nanmedian([a["hr"] for a in m])), 1))
            cols["ppg_rmssd_ms"].append(round(float(np.nanmedian([a["rmssd"] for a in m])), 1))
            samples[int(r["day"])] = xs[0][: FS * SEG_S]
            if self.available:
                e, ipa, sqi = zip(*(self.embed(x) for x in xs))
                e, ipa, sqi = np.concatenate(e), np.concatenate(ipa), np.concatenate(sqi)
                good = sqi >= np.quantile(sqi, 0.2)                        # drop worst-quality segments
                embs.append(e[good].mean(0))
                cols["ppg_sqi"].append(round(float(sqi.mean()), 3))
                cols["ppg_ipa"].append(round(float(ipa[good].mean()), 3))
            else:
                cols["ppg_sqi"].append(round(float(np.mean([a["sqi"] for a in m])), 3))
                cols["ppg_ipa"].append(np.nan)
        if embs:
            E = np.array(embs)
            E = E / np.linalg.norm(E, axis=1, keepdims=True)
            base = E[:3].mean(0)
            base /= np.linalg.norm(base)
            dist = 1 - E @ base
            spread = max(float(np.std(dist[:3])), 1e-3)
            cols["ppg_embedding_drift"] = list(np.round((dist - dist[:3].mean()) / spread, 2))
        else:
            cols["ppg_embedding_drift"] = [np.nan] * len(nt)
        for k, v in cols.items():
            nt[k] = v
        return nt, samples
