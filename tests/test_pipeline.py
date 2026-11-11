"""Tests for the completed data and preprocessing foundation."""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from noshows.data import make_appointments
from noshows.pipeline import (
    EXCLUDED_COLUMNS,
    FEATURE_NAMES,
    SEED,
    TARGET,
    build_pipeline,
    make_cv,
    split_appointments,
)


def test_data_is_reproducible():
    pd.testing.assert_frame_equal(
        make_appointments(seed=SEED), make_appointments(seed=SEED)
    )


def test_production_features_exclude_identifiers_target_and_leakage():
    assert not set(FEATURE_NAMES) & {*EXCLUDED_COLUMNS, TARGET}
    assert len(FEATURE_NAMES) == 10


def test_split_has_correct_size_and_disjoint_rows():
    frame = make_appointments(seed=SEED)
    training, test = split_appointments(frame)
    assert (len(training), len(test)) == (3000, 1000)
    assert set(training.appointment_id).isdisjoint(test.appointment_id)
    assert set(training.appointment_id) | set(test.appointment_id) == set(
        frame.appointment_id
    )


def test_split_is_stratified_and_reproducible():
    frame = make_appointments(seed=SEED)
    training, test = split_appointments(frame)
    training2, test2 = split_appointments(frame)
    assert abs(training[TARGET].mean() - frame[TARGET].mean()) < 0.002
    assert abs(test[TARGET].mean() - frame[TARGET].mean()) < 0.002
    pd.testing.assert_frame_equal(training, training2)
    pd.testing.assert_frame_equal(test, test2)


def test_pipeline_handles_missing_distance():
    training, test = split_appointments(make_appointments(n=400, seed=SEED))
    model = build_pipeline(LogisticRegression(max_iter=2000))
    model.fit(training.loc[:, list(FEATURE_NAMES)], training[TARGET])
    example = test.loc[:, list(FEATURE_NAMES)].iloc[:1].copy()
    example["distance_km"] = np.nan
    probabilities = model.predict_proba(example)
    assert probabilities.shape == (1, 2)
    assert np.isfinite(probabilities).all()
    assert np.isclose(probabilities.sum(), 1)


def test_imputer_statistics_come_from_training_only():
    training, _ = split_appointments(make_appointments(n=400, seed=SEED))
    model = build_pipeline(LogisticRegression(max_iter=2000))
    model.fit(training.loc[:, list(FEATURE_NAMES)], training[TARGET])
    imputer = (
        model.named_steps["preprocess"]
        .named_transformers_["numeric"]
        .named_steps["imputer"]
    )
    assert imputer.statistics_[-1] == training.distance_km.median()


def test_pipeline_builds_independent_estimators():
    original = LogisticRegression()
    first, second = build_pipeline(original), build_pipeline(original)
    assert first is not second
    assert first.named_steps["model"] is not original
    assert first.named_steps["model"] is not second.named_steps["model"]


def test_cv_is_stratified_five_fold_and_reproducible():
    frame = make_appointments(n=400, seed=SEED)
    first = list(make_cv().split(frame, frame[TARGET]))
    second = list(make_cv().split(frame, frame[TARGET]))
    assert len(first) == 5
    for (train1, val1), (train2, val2) in zip(first, second, strict=True):
        np.testing.assert_array_equal(train1, train2)
        np.testing.assert_array_equal(val1, val2)
        assert set(train1).isdisjoint(val1)
        assert abs(frame.iloc[val1][TARGET].mean() - frame[TARGET].mean()) < 0.02
