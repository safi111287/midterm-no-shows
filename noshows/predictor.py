"""Saved-model prediction interface; implementation owned by Kawsarul."""

import json
from math import isfinite
from pathlib import Path
from typing import TypedDict

import joblib

from noshows.appointment import FEATURE_NAMES, Appointment

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "no_show.joblib"
INFO_PATH = PROJECT_ROOT / "models" / "model_info.json"


class PredictionResult(TypedDict):
    """Output contract shared with the user interface."""

    no_show_probability: float
    attend_probability: float
    call_patient: bool
    threshold: float
    warnings: tuple[str, ...]


class NoShowPredictor:
    """Load once per instance; predict with the saved fitted pipeline."""

    def __init__(
        self,
        model_path: str | Path = MODEL_PATH,
        info_path: str | Path = INFO_PATH,
    ) -> None:
        self.pipeline = joblib.load(model_path)
        with Path(info_path).open(encoding="utf-8") as handle:
            info = json.load(handle)

        expected = list(FEATURE_NAMES)
        feature_keys = ("feature_names", "features", "feature_columns")
        present = [key for key in feature_keys if key in info]
        if not present:
            raise ValueError("model_info.json: missing feature list")
        for key in present:
            if not isinstance(info[key], list) or info[key] != expected:
                raise ValueError(f"model_info.json: {key} does not match FEATURE_NAMES")
        for key in ("numeric_features", "categorical_features"):
            if key in info:
                from noshows.appointment import CATEGORY_RULES, NUMERIC_RULES

                expected_subset = list(
                    NUMERIC_RULES if key == "numeric_features" else CATEGORY_RULES
                )
                if not isinstance(info[key], list) or info[key] != expected_subset:
                    raise ValueError(
                        f"model_info.json: {key} does not match shared rules"
                    )
        if (
            hasattr(self.pipeline, "feature_names_in_")
            and list(self.pipeline.feature_names_in_) != expected
        ):
            raise ValueError("saved pipeline feature names do not match FEATURE_NAMES")

        try:
            raw = info["threshold"]
            if isinstance(raw, bool):
                raise TypeError("boolean threshold")
            value = float(raw)
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise ValueError("threshold: invalid model metadata") from exc
        if not isfinite(value) or not 0 <= value <= 1:
            raise ValueError("threshold: must be finite and within [0, 1]")
        self.threshold = value

    def should_call(self, probability: float) -> bool:
        """Use the saved threshold, including the equality boundary."""
        if isinstance(probability, bool):
            raise ValueError("probability: booleans are not probabilities")
        try:
            value = float(probability)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("probability: expected a number") from exc
        if not isfinite(value) or not 0 <= value <= 1:
            raise ValueError("probability: must be finite and within [0, 1]")
        return bool(value >= self.threshold)

    def predict(self, appointment: Appointment) -> PredictionResult:
        """Find the label-1 probability without assuming class order."""
        if not isinstance(appointment, Appointment):
            raise TypeError("appointment must be an Appointment")
        classes = list(self.pipeline.classes_)
        if classes.count(1) != 1:
            raise ValueError("saved pipeline must contain exactly one class labeled 1")
        scores = self.pipeline.predict_proba(appointment.to_frame())
        probability = float(scores[0][classes.index(1)])
        decision = self.should_call(probability)
        return {
            "no_show_probability": probability,
            "attend_probability": float(1 - probability),
            "call_patient": decision,
            "threshold": float(self.threshold),
            "warnings": appointment.warnings,
        }

    def __repr__(self) -> str:
        return f"NoShowPredictor(threshold={self.threshold!r})"
