"""Expert 2 - Time-series foundation model (zero-shot, pretrained weights, no training).

Default: Amazon Chronos-Bolt (Ansari et al., 2024, "Chronos: Learning the Language of
Time Series"), weights from Hugging Face `amazon/chronos-bolt-small`.
GlucoFM-Bench (2026) reports Chronos-2 / TimesFM zero-shot CGM forecasting within ~5% of
the best full-shot supervised model, which is why we use it untrained.

If `chronos-forecasting`/torch or the weights are unavailable, a damped-trend
statistical forecaster is used so the app still runs.
"""
from __future__ import annotations

import os
import numpy as np

QUANTILES = [0.1, 0.5, 0.9]


class FoundationExpert:
    name = "foundation"

    def __init__(self, model_id: str | None = None):
        self.model_id = model_id or os.getenv("CHRONOS_MODEL", "amazon/chronos-bolt-small")
        self.pipe = None
        self.backend = "damped-trend (fallback)"
        self.error = None
        if os.getenv("DISABLE_CHRONOS") == "1":
            self.error = "disabled via DISABLE_CHRONOS=1"
            return
        try:
            import torch
            from chronos import BaseChronosPipeline

            self._torch = torch
            self.pipe = BaseChronosPipeline.from_pretrained(
                self.model_id, device_map="cpu", torch_dtype=torch.float32
            )
            self.backend = self.model_id
        except Exception as e:  # noqa: BLE001
            self.error = f"{type(e).__name__}: {e}"

    @property
    def available(self) -> bool:
        return self.pipe is not None

    def forecast(self, contexts: list[np.ndarray], horizon: int = 24):
        """contexts: list of 1-D CGM arrays (5-min). Returns (q10, q50, q90) each [B, horizon]."""
        if self.pipe is not None:
            torch = self._torch
            ctx = [torch.tensor(c[-512:], dtype=torch.float32) for c in contexts]
            q, _mean = self.pipe.predict_quantiles(
                inputs=ctx, prediction_length=horizon, quantile_levels=QUANTILES
            ) if _accepts_inputs(self.pipe) else self.pipe.predict_quantiles(
                context=ctx, prediction_length=horizon, quantile_levels=QUANTILES
            )
            q = q.detach().cpu().numpy()  # [B, H, 3]
            return q[..., 0], q[..., 1], q[..., 2]
        return _damped_trend(contexts, horizon)


def _accepts_inputs(pipe) -> bool:
    import inspect

    try:
        return "inputs" in inspect.signature(pipe.predict_quantiles).parameters
    except (TypeError, ValueError):
        return False


def _damped_trend(contexts, horizon):
    q10, q50, q90 = [], [], []
    k = np.arange(1, horizon + 1)
    for c in contexts:
        c = np.asarray(c, dtype=float)
        last = c[-1]
        tail = c[-6:]
        slope = np.polyfit(np.arange(len(tail)), tail, 1)[0] if len(tail) >= 3 else 0.0
        phi = 0.88
        trend = slope * phi * (1 - phi ** k) / (1 - phi)
        anchor = np.median(c[-288:])
        revert = (anchor - last) * (1 - np.exp(-k / 60.0))
        med = last + trend + revert
        resid = np.std(np.diff(c[-48:])) if len(c) > 10 else 5.0
        sd = resid * np.sqrt(k) * 1.2
        q10.append(med - 1.28 * sd); q50.append(med); q90.append(med + 1.28 * sd)
    return np.array(q10), np.array(q50), np.array(q90)
