"""Predictor tests to be implemented by Kawsarul."""

import pytest


def test_probability_range_and_output_types(valid_booking):
    """Check both probabilities, sum, bool decision, saved threshold and warnings."""
    raise NotImplementedError("Implement probability test.")


def test_riskier_booking_scores_higher(valid_booking):
    """Compare long lead, previous misses, online/first visit with a safer booking."""
    raise NotImplementedError("Implement risk comparison.")


@pytest.mark.parametrize(
    ("offset", "expected"), [(-1e-6, False), (0, True), (1e-6, True)]
)
def test_decision_at_saved_threshold(offset, expected):
    """Use the predictor's saved threshold plus offset, not a typed-in threshold."""
    raise NotImplementedError("Implement saved-threshold boundary test.")


def test_model_loads_once_per_predictor(monkeypatch, valid_booking):
    """Count predictor module joblib.load: one construction and two predictions."""
    raise NotImplementedError("Implement one-load test.")


def test_predictor_repr_is_readable():
    """Include NoShowPredictor and its threshold in repr(instance)."""
    raise NotImplementedError("Implement repr test.")
