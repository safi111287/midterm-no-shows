"""Common valid booking values for class and interface tests."""

import pytest


@pytest.fixture
def valid_booking():
    return {
        "lead_days": 7,
        "age": 46,
        "prior_no_shows": 0,
        "hour": 11,
        "first_visit": 0,
        "distance_km": 5.0,
        "weekday": "wed",
        "booking": "phone",
        "insurance": "private",
        "treatment": "physio",
    }
