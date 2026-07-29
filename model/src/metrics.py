"""RIADD (ISBI-2021) challenge metrics for RFMiD.

The official score has two halves:

* **Score A** — screening: ROC-AUC of ``Disease_Risk``.
* **Score B** — multi-disease over 28 categories (27 named + ``OTHER``):
  ``(mAP + mAUC) / 2``, macro-averaged.

Final score = ``(A + B) / 2``. Leaderboard reference: A = 0.9636,
B = 0.7873, final = 0.8754.

Classes with no positive in the evaluated split are skipped from the macro
averages (AUC/AP are undefined there) and reported in ``n_skipped`` so the
numbers stay comparable and honest.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from .labels import ALL_LABELS, DISEASE_LABELS, RIADD_ML_LABELS


def _rare_slice() -> slice:
    """Columns of the 18 rare diseases inside the 46-wide array."""
    start = 1 + 27  # skip Disease_Risk + the 27 scored diseases
    return slice(start, 1 + len(DISEASE_LABELS))


def to_riadd_view(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split a ``(N, 46)`` array into ``(binary (N,), multi-disease (N, 28))``.

    ``OTHER`` is the max over the 18 rare columns — the natural reduction for
    both a 0/1 target (logical OR) and a probability (most confident rare hit).
    """
    arr = np.asarray(arr, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != len(ALL_LABELS):
        raise ValueError(f"expected (N, {len(ALL_LABELS)}), got {arr.shape}")
    binary = arr[:, 0]
    scored = arr[:, 1:1 + 27]
    other = arr[:, _rare_slice()].max(axis=1, keepdims=True)
    return binary, np.concatenate([scored, other], axis=1)


def _macro(y: np.ndarray, p: np.ndarray, fn) -> tuple[float, dict[str, float], int]:
    per_class: dict[str, float] = {}
    for i, name in enumerate(RIADD_ML_LABELS):
        col = y[:, i]
        if col.sum() == 0 or col.sum() == len(col):
            continue
        per_class[name] = float(fn(col, p[:, i]))
    values = list(per_class.values())
    n_skipped = len(RIADD_ML_LABELS) - len(values)
    return (float(np.mean(values)) if values else float("nan")), per_class, n_skipped


def riadd_score(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    """Full metric bundle from ``(N, 46)`` targets and probabilities."""
    y_bin, y_ml = to_riadd_view(y_true)
    p_bin, p_ml = to_riadd_view(y_prob)

    auc_binary = (
        float(roc_auc_score(y_bin, p_bin))
        if 0 < y_bin.sum() < len(y_bin)
        else float("nan")
    )
    m_auc, auc_per_class, n_skipped = _macro(y_ml, p_ml, roc_auc_score)
    m_ap, ap_per_class, _ = _macro(y_ml, p_ml, average_precision_score)

    score_b = (m_ap + m_auc) / 2
    return {
        "auc_disease_risk": auc_binary,
        "map_multi": m_ap,
        "mauc_multi": m_auc,
        "score_a": auc_binary,
        "score_b": score_b,
        "final_score": (auc_binary + score_b) / 2,
        "n_scored_classes": len(RIADD_ML_LABELS) - n_skipped,
        "n_skipped": n_skipped,
        "auc_per_class": auc_per_class,
        "ap_per_class": ap_per_class,
    }


def format_report(m: dict, title: str = "RIADD evaluation") -> str:
    lines = [
        f"== {title} ==",
        f"  Score A  Disease_Risk AUC : {m['auc_disease_risk']:.4f}   (leaderboard 0.9636)",
        f"  Score B  (mAP+mAUC)/2     : {m['score_b']:.4f}   (leaderboard 0.7873)",
        f"           mAUC             : {m['mauc_multi']:.4f}",
        f"           mAP              : {m['map_multi']:.4f}",
        f"  FINAL    (A+B)/2          : {m['final_score']:.4f}   (leaderboard 0.8754)",
        f"  scored {m['n_scored_classes']}/28 classes ({m['n_skipped']} had no positives in this split)",
    ]
    if m.get("auc_per_class"):
        ranked = sorted(m["auc_per_class"].items(), key=lambda kv: -kv[1])
        lines.append("  per-class AUC (best -> worst):")
        for name, v in ranked:
            ap = m["ap_per_class"].get(name, float("nan"))
            lines.append(f"    {name:<6} AUC {v:.4f}  AP {ap:.4f}")
    return "\n".join(lines)


def best_thresholds(y_true: np.ndarray, y_prob: np.ndarray,
                    grid: np.ndarray | None = None) -> np.ndarray:
    """Per-class threshold maximising F1 on the given split (46 values)."""
    from sklearn.metrics import f1_score

    grid = np.linspace(0.05, 0.95, 19) if grid is None else grid
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    out = np.full(y_true.shape[1], 0.5)
    for i in range(y_true.shape[1]):
        if y_true[:, i].sum() == 0:
            continue
        scores = [f1_score(y_true[:, i], (y_prob[:, i] >= t).astype(int),
                           zero_division=0) for t in grid]
        out[i] = float(grid[int(np.argmax(scores))])
    return out
