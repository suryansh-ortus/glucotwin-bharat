"""On-device online learning + federated learning, without touching the big models.

The frozen experts (Chronos, PaPaGei, physiology) never change. On top of the fused
forecast sits a tiny per-patient ADAPTER (24 horizons x 8 weights = 192 numbers) that
learns this patient's systematic error:

    corrected_k = fused_k + w_k . x_k        k = 5, 10, ... 120 min ahead

ONLINE LEARNING  - recursive least squares (RLS) with a forgetting factor. Every time a
                   2-h forecast "matures" (the real CGM arrives), the adapter is updated
                   once, on the device, in microseconds. Predict-then-update, so every
                   error it reports is out-of-sample (prequential evaluation).
FEDERATED        - each device shares ONLY its 192 adapter weights (never glucose, never
                   identity). The server does FedAvg (McMahan et al. 2017) with update
                   clipping + Gaussian noise (differential privacy, Abadi et al. 2016) and
                   sends back a global prior. A new patient starts from that prior instead
                   of zero, so the twin is useful from day one.

Same pattern scales to the big models later: LoRA adapters on Chronos-Bolt / PaPaGei,
averaged with FedAvg, while the base weights stay frozen on every device.
"""
from __future__ import annotations

import numpy as np

H = 24
F = 8
FEATURES = ["bias", "glucose level", "trend", "meal in last hour", "recent carbs",
            "forecast rise", "physiology vs foundation gap", "night"]


def features(cur, trend, minutes_since_meal, carbs_recent, fused_row, mech_row, fm_row, hour) -> np.ndarray:
    """Feature matrix [H, F] for one forecast origin."""
    meal = np.exp(-(minutes_since_meal if minutes_since_meal is not None else 600) / 60.0)
    base = np.array([1.0, (cur - 150) / 50.0, trend / 2.0, meal, carbs_recent / 60.0, 0.0, 0.0,
                     1.0 if (hour < 6 or hour >= 23) else 0.0])
    X = np.tile(base, (H, 1))
    X[:, 5] = (np.asarray(fused_row) - cur) / 50.0
    X[:, 6] = (np.asarray(mech_row) - np.asarray(fm_row)) / 50.0
    return X


class OnlineAdapter:
    def __init__(self, prior: np.ndarray | None = None, lam: float = 0.995, delta: float = 10.0,
                 warmup: int = 36, max_corr: float = 40.0):
        self.W = np.zeros((H, F)) if prior is None else prior.copy()
        self.prior = self.W.copy()
        self.P = np.stack([np.eye(F) * delta for _ in range(H)])
        self.lam, self.warmup, self.max_corr = lam, warmup, max_corr
        self.n = 0
        self.seen: set[int] = set()
        self.err_before: list[float] = []   # |actual - fused| (prequential)
        self.err_after: list[float] = []    # |actual - corrected|
        self.local_data: list[tuple] = []   # stays on the device; used only for local federated rounds
        if prior is not None:               # trust the federated prior: smaller P, no warm-up
            self.P /= 5.0
            self.warmup = 0

    def raw_correction(self, X: np.ndarray) -> np.ndarray:
        return np.clip(np.einsum("hf,hf->h", self.W, X), -self.max_corr, self.max_corr)

    @property
    def helping(self) -> bool:
        """Self-gating: serve corrections only while the adapter's own shadow error over the
        last 24 matured forecasts beats the un-adapted twin."""
        if len(self.err_after) < 24:
            return True
        return float(np.mean(self.err_after[-24:])) < float(np.mean(self.err_before[-24:]))

    def correction(self, X: np.ndarray) -> np.ndarray:
        if self.n < self.warmup or not self.helping:
            return np.zeros(H)
        return self.raw_correction(X)

    def update(self, X: np.ndarray, fused: np.ndarray, actual: np.ndarray):
        corr = self.raw_correction(X) if self.n >= self.warmup else np.zeros(H)   # shadow prediction
        self.err_before.append(float(np.mean(np.abs(actual - fused))))
        self.err_after.append(float(np.mean(np.abs(actual - fused - corr))))
        self.local_data.append((X, fused, actual))
        r = np.clip(actual - fused, -120, 120)
        for k in range(H):
            x, P = X[k], self.P[k]
            Px = P @ x
            g = Px / (self.lam + x @ Px)
            self.W[k] += g * (r[k] - self.W[k] @ x)
            self.P[k] = (P - np.outer(g, Px)) / self.lam
        self.n += 1

    def summary(self, last: int = 24) -> dict:
        b, a = self.err_before[-last:], self.err_after[-last:]
        return {"updates": self.n, "active": self.n >= self.warmup and self.helping,
                "mae_without": round(float(np.mean(b)), 2) if b else None, "mae_with": round(float(np.mean(a)), 2) if a else None,
                "n_params": int(self.W.size), "from_federated_prior": bool(np.any(self.prior)),
                "top_features": _top(self.W)}


def _top(W):
    imp = np.abs(W[:, 1:]).mean(0)
    order = np.argsort(-imp)[:3]
    return [FEATURES[i + 1] for i in order if imp[i] > 1e-6]


# ------------------------------------------------------------------ federated averaging
def fedavg(client_weights: list[np.ndarray], client_sizes: list[int], global_w: np.ndarray | None = None,
           clip: float = 5.0, noise_mult: float = 0.0, rng: np.random.Generator | None = None) -> np.ndarray:
    """FedAvg with per-client update clipping and optional Gaussian (DP) noise.
    Only adapter weights are exchanged - never raw data."""
    rng = rng or np.random.default_rng(0)
    g = np.zeros_like(client_weights[0]) if global_w is None else global_w
    deltas = []
    for w in client_weights:
        d = w - g
        nrm = np.linalg.norm(d)
        deltas.append(d * min(1.0, clip / (nrm + 1e-12)))
    sizes = np.asarray(client_sizes, dtype=float)
    avg = sum(s * d for s, d in zip(sizes, deltas)) / sizes.sum()
    if noise_mult > 0:
        avg = avg + rng.normal(0, noise_mult * clip / len(deltas), size=avg.shape)
    return g + avg


def replay(data: list[tuple], prior: np.ndarray | None = None) -> tuple[OnlineAdapter, float, float]:
    """Run an adapter over a stream of (X, fused, actual); returns (adapter, MAE fused, MAE adapted)."""
    ad = OnlineAdapter(prior)
    for X, f, y in data:
        ad.update(X, f, y)
    return ad, float(np.mean(ad.err_before)), float(np.mean(ad.err_after))
