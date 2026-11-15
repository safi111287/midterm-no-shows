"""Cost-based decisions, training threshold selection, and final evaluation."""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

CALL_COST = 8.0
MISS_COST = 35.0
THRESHOLDS = tuple(i / 100 for i in range(5, 96))


def _binary(values, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional array.")
    if not np.isin(array, [0, 1]).all():
        raise ValueError(f"{name} must contain only 0 and 1.")
    return array.astype(int)


def _probabilities(values) -> np.ndarray:
    try:
        array = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "probabilities must be finite numbers between 0 and 1."
        ) from exc
    if (
        array.ndim != 1
        or array.size == 0
        or not np.isfinite(array).all()
        or not ((array >= 0) & (array <= 1)).all()
    ):
        raise ValueError("probabilities must be finite numbers between 0 and 1.")
    return array


def _costs(call_cost: float, miss_cost: float) -> tuple[float, float]:
    try:
        call_cost, miss_cost = float(call_cost), float(miss_cost)
    except (TypeError, ValueError) as exc:
        raise ValueError("Costs must be finite non-negative numbers.") from exc
    if not np.isfinite([call_cost, miss_cost]).all() or min(call_cost, miss_cost) < 0:
        raise ValueError("Costs must be finite non-negative numbers.")
    return call_cost, miss_cost


def total_cost(
    y_true,
    flagged,
    *,
    call_cost: float = CALL_COST,
    miss_cost: float = MISS_COST,
) -> float:
    """Charge for every call and each uncalled actual no-show."""
    truth = _binary(y_true, "y_true")
    flags = _binary(flagged, "flagged")
    if truth.shape != flags.shape:
        raise ValueError("y_true and flagged must have the same length.")
    call_cost, miss_cost = _costs(call_cost, miss_cost)
    return float(
        call_cost * flags.sum() + miss_cost * ((truth == 1) & (flags == 0)).sum()
    )


def threshold_costs(
    y_true,
    probabilities,
    *,
    call_cost: float = CALL_COST,
    miss_cost: float = MISS_COST,
) -> pd.DataFrame:
    """Evaluate the fixed 0.05-0.95 grid using training OOF probabilities."""
    truth = _binary(y_true, "y_true")
    probability = _probabilities(probabilities)
    if truth.shape != probability.shape:
        raise ValueError("y_true and probabilities must have the same length.")
    rows = []
    for threshold in THRESHOLDS:
        flagged = probability >= threshold
        rows.append(
            {
                "threshold": threshold,
                "cost": total_cost(
                    truth, flagged, call_cost=call_cost, miss_cost=miss_cost
                ),
                "calls": int(flagged.sum()),
            }
        )
    return pd.DataFrame(rows)


def choose_threshold(
    y_true,
    probabilities,
    *,
    call_cost: float = CALL_COST,
    miss_cost: float = MISS_COST,
) -> float:
    """Minimize OOF cost; ties prefer fewer calls, then the higher threshold."""
    table = threshold_costs(
        y_true,
        probabilities,
        call_cost=call_cost,
        miss_cost=miss_cost,
    )
    best = table.sort_values(
        ["cost", "calls", "threshold"],
        ascending=[True, True, False],
    ).iloc[0]
    return float(best["threshold"])


def summarize_predictions(y_true, probabilities, threshold: float) -> dict:
    """Evaluate one fixed model/threshold on the held-out test probabilities."""
    truth = _binary(y_true, "y_true")
    probability = _probabilities(probabilities)
    if truth.shape != probability.shape:
        raise ValueError("y_true and probabilities must have the same length.")
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1.")
    flags = probability >= threshold
    none_cost = total_cost(truth, np.zeros(truth.size, dtype=int))
    all_cost = total_cost(truth, np.ones(truth.size, dtype=int))
    model_cost = total_cost(truth, flags)
    return {
        "n_test": int(truth.size),
        "no_shows": int(truth.sum()),
        "no_show_rate": float(truth.mean()),
        "calls": int(flags.sum()),
        "accuracy": float(accuracy_score(truth, flags)),
        "precision": float(precision_score(truth, flags, zero_division=0)),
        "recall": float(recall_score(truth, flags, zero_division=0)),
        "f1": float(f1_score(truth, flags, zero_division=0)),
        "roc_auc": float(roc_auc_score(truth, probability)),
        "confusion_matrix": confusion_matrix(truth, flags, labels=[0, 1]).tolist(),
        "costs": {
            "call_nobody": none_cost,
            "call_everybody": all_cost,
            "model": model_cost,
        },
        "savings_vs_nobody": none_cost - model_cost,
        "savings_vs_everybody": all_cost - model_cost,
        "beats_both": bool(model_cost < min(none_cost, all_cost)),
    }


def savings_intervals(y_true, probabilities, threshold: float, *, seed: int) -> dict:
    """Paired percentile bootstrap intervals for fixed-policy test savings.

    No models are refitted and the threshold is never changed here.
    """
    truth = _binary(y_true, "y_true")
    probability = _probabilities(probabilities)
    if truth.shape != probability.shape:
        raise ValueError("y_true and probabilities must have the same length.")
    flags = probability >= threshold
    per_patient = CALL_COST * flags + MISS_COST * truth * ~flags
    differences = {
        "vs_nobody": MISS_COST * truth - per_patient,
        "vs_everybody": CALL_COST - per_patient,
    }
    rng = np.random.default_rng(seed)
    samples = {name: [] for name in differences}
    for _ in range(2000):
        indices = rng.integers(0, truth.size, truth.size)
        for name, difference in differences.items():
            samples[name].append(float(difference[indices].sum()))
    return {
        name: [float(value) for value in np.percentile(values, [2.5, 97.5])]
        for name, values in samples.items()
    }
