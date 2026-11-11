"""Immutable appointment interface; implementation owned by Kawsarul."""

from dataclasses import dataclass
from math import isfinite, isnan

import pandas as pd

from noshows import pipeline as contract

# Prefer the current team contract; support the older shared package's names.
NUMERIC_RULES = getattr(contract, "NUMERIC_RULES", None)
if NUMERIC_RULES is None:
    NUMERIC_RULES = contract.BOUNDS
CATEGORY_RULES = getattr(contract, "CATEGORY_RULES", None)
if CATEGORY_RULES is None:
    CATEGORY_RULES = contract.CATEGORIES
FEATURE_NAMES = getattr(contract, "FEATURE_NAMES", None)
if FEATURE_NAMES is None:
    FEATURE_NAMES = contract.FEATURES
MISSING_DISTANCE_WARNING = getattr(
    contract,
    "MISSING_DISTANCE_WARNING",
    "distance_km is missing; the model uses its training median.",
)


@dataclass(frozen=True)
class Appointment:
    """One validated booking, preserving exactly the training features."""

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
        """Normalize numeric strings and reject invalid fields by name."""
        for name, (lower, upper, whole) in NUMERIC_RULES.items():
            raw = getattr(self, name)
            if name == "distance_km" and (
                raw is None or isinstance(raw, str) and not raw.strip()
            ):
                object.__setattr__(self, name, float("nan"))
                continue
            if isinstance(raw, bool):
                raise ValueError(f"{name}: booleans are not numeric values")
            try:
                value = float(raw)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(f"{name}: expected a number") from exc
            if name == "distance_km" and isnan(value):
                object.__setattr__(self, name, float("nan"))
                continue
            if not isfinite(value):
                raise ValueError(f"{name}: expected a finite number")
            if not lower <= value <= upper:
                raise ValueError(f"{name}: must be between {lower} and {upper}")
            if whole and not value.is_integer():
                raise ValueError(f"{name}: must be a whole number")
            object.__setattr__(self, name, int(value) if whole else value)

        for name, choices in CATEGORY_RULES.items():
            raw = getattr(self, name)
            if not isinstance(raw, str) or raw not in choices:
                raise ValueError(f"{name}: must be one of {', '.join(choices)}")

    def to_frame(self) -> pd.DataFrame:
        """Return one row with the exact trained feature order and numeric dtypes."""
        return pd.DataFrame(
            [{name: getattr(self, name) for name in FEATURE_NAMES}],
            columns=list(FEATURE_NAMES),
        )

    @property
    def warnings(self) -> tuple[str, ...]:
        """Warn if the pipeline must impute the optional distance feature."""
        return (MISSING_DISTANCE_WARNING,) if isnan(self.distance_km) else ()
