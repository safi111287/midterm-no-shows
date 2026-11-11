"""Appointment tests to be implemented by Kawsarul."""

import pytest


def test_valid_booking_to_frame(valid_booking):
    """Check one row, FEATURE_NAMES order, values, and no warnings."""
    raise NotImplementedError("Implement valid-booking test.")


@pytest.mark.parametrize("distance", [None, "", "  ", float("nan")])
def test_missing_distance_is_accepted_with_warning(valid_booking, distance):
    """Normalize to NaN; expose MISSING_DISTANCE_WARNING; prediction must work."""
    raise NotImplementedError("Implement missing-distance test.")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("lead_days", -1),
        ("lead_days", 121),
        ("lead_days", 1.5),
        ("age", 11),
        ("age", 111),
        ("prior_no_shows", -1),
        ("prior_no_shows", 31),
        ("hour", 7),
        ("hour", 18),
        ("hour", 9.5),
        ("first_visit", 2),
        ("distance_km", 0),
        ("distance_km", 151),
        ("distance_km", float("inf")),
    ],
)
def test_impossible_numbers_name_field(valid_booking, field, value):
    """Expect ValueError whose message contains the changed field name."""
    raise NotImplementedError("Implement invalid-number tests.")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("weekday", "Mon"),
        ("weekday", "sat"),
        ("booking", "email"),
        ("insurance", "unknown"),
        ("treatment", "surgery"),
    ],
)
def test_unknown_categories_name_field(valid_booking, field, value):
    """Reject unrecognized spellings before calling the encoder."""
    raise NotImplementedError("Implement invalid-category tests.")


def test_booking_cannot_be_changed(valid_booking):
    """Setting a field after construction must raise FrozenInstanceError."""
    raise NotImplementedError("Implement immutability test.")


def test_required_values_and_booleans_are_rejected(valid_booking):
    """Check None/blank/NaN/infinity/bool for required numeric fields."""
    raise NotImplementedError("Implement required-value tests.")
