"""Appointment tests to be implemented by Kawsarul."""

from dataclasses import FrozenInstanceError
from math import isnan

import pandas as pd
import pytest

from noshows.appointment import (
    FEATURE_NAMES,
    MISSING_DISTANCE_WARNING,
    Appointment,
)


@pytest.fixture
def booking():
    return {
        "lead_days": 4,
        "age": 42,
        "prior_no_shows": 1,
        "hour": 10,
        "first_visit": 0,
        "distance_km": 8.5,
        "weekday": "mon",
        "booking": "online",
        "insurance": "private",
        "treatment": "physio",
    }


def test_valid_row_and_feature_order(booking):
    appointment = Appointment(**booking)
    frame = appointment.to_frame()
    assert isinstance(frame, pd.DataFrame)
    assert frame.shape == (1, len(FEATURE_NAMES))
    assert frame.columns.tolist() == list(FEATURE_NAMES)
    assert frame.loc[0, "distance_km"] == 8.5
    assert appointment.warnings == ()


@pytest.mark.parametrize("missing", [None, "", "  \t ", float("nan"), "NaN"])
def test_missing_distance_is_nan_and_warns(booking, missing):
    appointment = Appointment(**{**booking, "distance_km": missing})
    assert isnan(appointment.distance_km)
    assert appointment.to_frame()["distance_km"].isna().all()
    assert appointment.warnings == (MISSING_DISTANCE_WARNING,)


@pytest.mark.parametrize(
    "field,value,expected",
    [
        ("lead_days", "120", 120),
        ("age", "12.0", 12),
        ("prior_no_shows", "30", 30),
        ("hour", "17", 17),
        ("first_visit", "1", 1),
        ("distance_km", "0.1", 0.1),
        ("distance_km", "150", 150.0),
    ],
)
def test_numeric_strings_normalize(booking, field, value, expected):
    appointment = Appointment(**{**booking, field: value})
    assert getattr(appointment, field) == expected
    assert not isinstance(getattr(appointment, field), str)


@pytest.mark.parametrize(
    "field,value",
    [
        ("lead_days", -1),
        ("lead_days", 121),
        ("age", 11),
        ("age", 111),
        ("prior_no_shows", 31),
        ("hour", 7),
        ("hour", 18),
        ("first_visit", 2),
        ("distance_km", 0),
        ("distance_km", 151),
        ("age", 40.5),
        ("lead_days", 1.5),
        ("prior_no_shows", 1.2),
        ("hour", 9.8),
        ("first_visit", 0.5),
        ("hour", None),
        ("lead_days", ""),
        ("age", "not a number"),
        ("age", float("nan")),
        ("hour", float("inf")),
        ("distance_km", float("inf")),
        ("distance_km", "-inf"),
        ("first_visit", True),
        ("distance_km", False),
        ("weekday", "Mon"),
        ("weekday", "sun"),
        ("booking", "email"),
        ("insurance", "other"),
        ("treatment", "unknown"),
        ("treatment", None),
    ],
)
def test_invalid_field_is_named(booking, field, value):
    with pytest.raises(ValueError, match=field):
        Appointment(**{**booking, field: value})


def test_immutable_after_construction(booking):
    appointment = Appointment(**booking)
    with pytest.raises(FrozenInstanceError):
        appointment.age = 43


def test_required_fields_cannot_be_omitted(booking):
    for field in booking:
        with pytest.raises(TypeError):
            Appointment(**{key: val for key, val in booking.items() if key != field})
