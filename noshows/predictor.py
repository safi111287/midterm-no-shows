"""Saved-model prediction interface; implementation owned by Kawsarul."""

from pathlib import Path
from typing import TypedDict

from noshows.appointment import Appointment

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "no_show.joblib"
INFO_PATH = PROJECT_ROOT / "models" / "model_info.json"


class PredictionResult(TypedDict):
    """Stable output shared with the page."""

    no_show_probability: float
    attend_probability: float
    call_patient: bool
    threshold: float
    warnings: tuple[str, ...]


class NoShowPredictor:
    """Load one pipeline and its metadata per instance, never per prediction."""

    def __init__(
        self,
        model_path: str | Path = MODEL_PATH,
        info_path: str | Path = INFO_PATH,
    ) -> None:
        """Load with joblib.load once, read JSON once, retain self.threshold.

        Check saved feature lists against FEATURE_NAMES and validate the saved
        threshold as a finite number in [0, 1]. Do not fit or retrain here.
        """
        raise NotImplementedError("Implement NoShowPredictor loading.")

    def should_call(self, probability: float) -> bool:
        """Validate probability, then return bool(probability >= self.threshold)."""
        raise NotImplementedError("Implement NoShowPredictor.should_call.")

    def predict(self, appointment: Appointment) -> PredictionResult:
        """Use to_frame and predict_proba; locate class 1 using model.classes_.

        Return exactly the five PredictionResult keys, including appointment.warnings.
        Preserve the booking and do not reload the model.
        """
        raise NotImplementedError("Implement NoShowPredictor.predict.")

    def __repr__(self) -> str:
        """Include the class name and the saved threshold in readable text."""
        raise NotImplementedError("Implement NoShowPredictor.__repr__.")
