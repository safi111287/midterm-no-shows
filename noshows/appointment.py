"""Immutable appointment interface; implementation owned by Kawsarul."""

from dataclasses import dataclass

import pandas as pd

from noshows.pipeline import CATEGORY_RULES as CATEGORY_RULES
from noshows.pipeline import FEATURE_NAMES as FEATURE_NAMES
from noshows.pipeline import MISSING_DISTANCE_WARNING as MISSING_DISTANCE_WARNING
from noshows.pipeline import NUMERIC_RULES as NUMERIC_RULES


@dataclass(frozen=True)
class Appointment:
    """One validated booking, using the shared feature names and rules.

    Convert numeric strings to numbers, reject booleans/nonfinite values,
    enforce whole numbers where required, and name invalid fields in ValueError.
    Only distance_km accepts None, an empty string, or NaN as missing.
    Normalize accepted missing distance to float('nan'). Reject infinity.
    Use object.__setattr__ only during __post_init__ normalization.
    """

    lead_days: int
    age: int
    prior_no_shows: int
    hour: int
    first_visit: int
    distance_km: float | None
    weekday: str
    booking: str
    insurance: str
    treatment: str

    def __post_init__(self) -> None:
        """Validate and normalize every field before the object is used."""
        raise NotImplementedError("Implement Appointment validation.")

    def to_frame(self) -> pd.DataFrame:
        """Return one row, with columns in FEATURE_NAMES order and numeric dtypes."""
        raise NotImplementedError("Implement Appointment.to_frame.")

    @property
    def warnings(self) -> tuple[str, ...]:
        """Return (MISSING_DISTANCE_WARNING,) if distance is missing, else ()."""
        raise NotImplementedError("Implement Appointment.warnings.")
