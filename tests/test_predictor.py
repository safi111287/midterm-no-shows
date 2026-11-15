"""Predictor tests to be implemented by Kawsarul."""

import json
from pathlib import Path

import joblib
import numpy as np
import pytest

from noshows.appointment import FEATURE_NAMES, Appointment
from noshows.predictor import NoShowPredictor


@pytest.fixture
def booking():
    return {
        "lead_days": 2,
        "age": 65,
        "prior_no_shows": 0,
        "hour": 11,
        "first_visit": 0,
        "distance_km": 2.0,
        "weekday": "wed",
        "booking": "front_desk",
        "insurance": "provincial",
        "treatment": "acupuncture",
    }


@pytest.fixture
def predictor():
    return NoShowPredictor()


def test_saved_probability_schema_and_types(predictor, booking):
    result = predictor.predict(Appointment(**booking))
    assert set(result) == {
        "no_show_probability",
        "attend_probability",
        "call_patient",
        "threshold",
        "warnings",
    }
    assert type(result["no_show_probability"]) is float
    assert type(result["attend_probability"]) is float
    assert type(result["threshold"]) is float
    assert type(result["call_patient"]) is bool
    assert type(result["warnings"]) is tuple
    assert 0 <= result["no_show_probability"] <= 1
    assert 0 <= result["attend_probability"] <= 1
    total = result["no_show_probability"] + result["attend_probability"]
    assert total == pytest.approx(1)


def test_riskier_booking_scores_higher(predictor, booking):
    safe = predictor.predict(Appointment(**booking))
    risky = predictor.predict(
        Appointment(
            **{
                **booking,
                "lead_days": 45,
                "age": 22,
                "prior_no_shows": 4,
                "hour": 17,
                "weekday": "mon",
                "booking": "online",
                "insurance": "self_pay",
                "first_visit": 1,
                "distance_km": 35.0,
                "treatment": "physio",
            }
        )
    )
    assert risky["no_show_probability"] > safe["no_show_probability"]


@pytest.mark.parametrize("delta,expected", [(-1e-9, False), (0, True), (1e-9, True)])
def test_threshold_boundary(predictor, delta, expected):
    assert predictor.should_call(predictor.threshold + delta) is expected


@pytest.mark.parametrize(
    "value",
    [True, False, None, "oops", -0.01, 1.01, float("nan"), float("inf"), -float("inf")],
)
def test_should_call_rejects_bad_probability(predictor, value):
    with pytest.raises(ValueError, match="probability"):
        predictor.should_call(value)


def test_threshold_and_repr_use_metadata(predictor):
    info_path = Path(__file__).resolve().parents[1] / "models" / "model_info.json"
    with info_path.open(encoding="utf-8") as handle:
        metadata = json.load(handle)
    assert predictor.threshold == metadata["threshold"]
    assert "NoShowPredictor" in repr(predictor)
    assert str(predictor.threshold) in repr(predictor)


def test_missing_distance_warning_is_tuple(predictor, booking):
    result = predictor.predict(Appointment(**{**booking, "distance_km": None}))
    assert len(result["warnings"]) == 1


def test_one_load_for_repeated_predictions(monkeypatch, booking):
    calls = []
    original = joblib.load

    def counted(path):
        calls.append(path)
        return original(path)

    monkeypatch.setattr(joblib, "load", counted)
    predictor = NoShowPredictor()
    appointment = Appointment(**booking)
    predictor.predict(appointment)
    predictor.predict(appointment)
    assert len(calls) == 1


def test_reversed_classes_are_not_assumed(monkeypatch, predictor, booking):
    class Reversed:
        classes_ = np.array([1, 0])

        def predict_proba(self, frame):
            assert frame.shape[0] == 1
            return np.array([[0.8, 0.2]])

    predictor.pipeline = Reversed()
    result = predictor.predict(Appointment(**booking))
    assert result["no_show_probability"] == pytest.approx(0.8)
    assert result["attend_probability"] == pytest.approx(0.2)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -0.1, 1.1, True])
def test_bad_metadata_threshold_rejected(tmp_path, monkeypatch, bad):
    info = {
        "features": list(FEATURE_NAMES),
        "threshold": bad,
    }
    location = tmp_path / "model_info.json"
    location.write_text(json.dumps(info))
    monkeypatch.setattr(joblib, "load", lambda _: object())
    with pytest.raises(ValueError, match="threshold"):
        NoShowPredictor(info_path=location)


def test_mismatched_metadata_features_rejected(tmp_path, monkeypatch):
    location = tmp_path / "model_info.json"
    location.write_text(json.dumps({"features": ["wrong"], "threshold": 0.5}))
    monkeypatch.setattr(joblib, "load", lambda _: object())
    with pytest.raises(ValueError, match="features"):
        NoShowPredictor(info_path=location)


def test_bad_model_probability_rejected(predictor, booking):
    class BadModel:
        classes_ = [0, 1]

        def predict_proba(self, frame):
            return [[0, float("nan")]]

    predictor.pipeline = BadModel()
    with pytest.raises(ValueError, match="probability"):
        predictor.predict(Appointment(**booking))
