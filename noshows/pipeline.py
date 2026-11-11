"""Shared feature definitions, validation rules, split, and preprocessing."""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SEED = 11
TARGET = "no_show"
TEST_SIZE = 0.25
CV_FOLDS = 5
NUMERIC_FEATURES = (
    "lead_days",
    "age",
    "prior_no_shows",
    "hour",
    "first_visit",
    "distance_km",
)
CATEGORICAL_FEATURES = ("weekday", "booking", "insurance", "treatment")
FEATURE_NAMES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
EXCLUDED_COLUMNS = ("appointment_id", "fee_charged")
# Each numeric rule contains minimum, maximum, and whole-number requirement.
NUMERIC_RULES = {
    "lead_days": (0, 120, True),
    "age": (12, 110, True),
    "prior_no_shows": (0, 30, True),
    "hour": (8, 17, True),
    "first_visit": (0, 1, True),
    "distance_km": (0.1, 150, False),
}
CATEGORY_RULES = {
    "weekday": ("mon", "tue", "wed", "thu", "fri"),
    "booking": ("online", "phone", "front_desk"),
    "insurance": ("private", "provincial", "self_pay"),
    "treatment": ("physio", "massage", "acupuncture"),
}
MISSING_DISTANCE_WARNING = (
    "Distance is missing; the model will use the training median distance."
)


def split_appointments(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return full training/test tables in a stratified 75/25 split."""
    return train_test_split(
        frame,
        test_size=TEST_SIZE,
        random_state=SEED,
        stratify=frame[TARGET],
    )


def make_cv() -> StratifiedKFold:
    """Use the same five reproducible folds for training comparisons."""
    return StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)


def build_pipeline(model, *, include_leakage: bool = False) -> Pipeline:
    """Create fresh preprocessing and a cloned, unfitted classifier.

    include_leakage is for the separate training-only leakage experiment.
    Production features always exclude fee_charged.
    """
    from sklearn.base import clone

    numbers = list(NUMERIC_FEATURES)
    if include_leakage:
        numbers.append("fee_charged")
    numeric = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    preprocess = ColumnTransformer(
        [
            ("numeric", numeric, numbers),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                list(CATEGORICAL_FEATURES),
            ),
        ]
    )
    return Pipeline([("preprocess", preprocess), ("model", clone(model))])
