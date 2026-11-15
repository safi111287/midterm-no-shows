"""Recreate the model and reports with python -m noshows.train."""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, make_scorer, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, cross_val_predict, cross_validate

from noshows.data import make_appointments
from noshows.evaluation import (
    CALL_COST,
    MISS_COST,
    choose_threshold,
    savings_intervals,
    summarize_predictions,
    threshold_costs,
)
from noshows.pipeline import (
    CATEGORICAL_FEATURES,
    FEATURE_NAMES,
    NUMERIC_FEATURES,
    SEED,
    TARGET,
    build_pipeline,
    make_cv,
    split_appointments,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCORING = {
    "accuracy": "accuracy",
    "precision": make_scorer(precision_score, zero_division=0),
    "recall": make_scorer(recall_score, zero_division=0),
    "f1": make_scorer(f1_score, zero_division=0),
    "roc_auc": "roc_auc",
}


def _candidates() -> dict:
    return {
        "dummy": DummyClassifier(strategy="most_frequent"),
        "logistic": LogisticRegression(max_iter=2000, random_state=SEED),
        "logistic_balanced": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=SEED,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=120,
            min_samples_leaf=5,
            random_state=SEED,
            n_jobs=1,
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            max_iter=100,
            max_leaf_nodes=15,
            random_state=SEED,
        ),
    }


def _grid(name: str) -> dict:
    if name.startswith("logistic"):
        return {"model__C": [0.1, 1.0, 10.0], "model__fit_intercept": [True, False]}
    if name == "random_forest":
        return {"model__max_depth": [None, 8, 16], "model__min_samples_leaf": [2, 5]}
    return {"model__learning_rate": [0.03, 0.1, 0.2], "model__max_leaf_nodes": [7, 15]}


def _write_json(path: Path, contents: dict) -> None:
    path.write_text(
        json.dumps(contents, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def main() -> None:
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError("Train in the midterm-no-shows Python 3.12 environment.")
    frame = make_appointments(seed=SEED)
    training, test = split_appointments(frame)
    x_train = training.loc[:, list(FEATURE_NAMES)]
    y_train = training[TARGET]
    candidates = _candidates()
    comparison_rows = []
    for name, model in candidates.items():
        scores = cross_validate(
            build_pipeline(model),
            x_train,
            y_train,
            cv=make_cv(),
            scoring=SCORING,
            n_jobs=1,
            error_score="raise",
        )
        row = {"model": name}
        for metric in SCORING:
            row[f"{metric}_mean"] = float(np.mean(scores[f"test_{metric}"]))
            row[f"{metric}_std"] = float(np.std(scores[f"test_{metric}"], ddof=1))
        comparison_rows.append(row)
        print(f"Compared {name}", flush=True)
    comparison = pd.DataFrame(comparison_rows)
    eligible = comparison.loc[comparison["model"] != "dummy"]
    winner = eligible.sort_values(
        ["roc_auc_mean", "model"],
        ascending=[False, True],
    ).iloc[0]
    winner_name = str(winner["model"])
    search = GridSearchCV(
        build_pipeline(candidates[winner_name]),
        _grid(winner_name),
        scoring="roc_auc",
        cv=make_cv(),
        n_jobs=1,
        refit=False,
        error_score="raise",
    )
    search.fit(x_train, y_train)
    tuned_mean = float(search.best_score_)
    untuned_mean = float(winner["roc_auc_mean"])
    use_tuned = tuned_mean > untuned_mean
    chosen = build_pipeline(candidates[winner_name])
    if use_tuned:
        chosen.set_params(**search.best_params_)
    settings = {
        key.removeprefix("model__"): value for key, value in search.best_params_.items()
    }
    tuned_std = float(search.cv_results_["std_test_score"][search.best_index_])
    # These probabilities exclude each row from its own fold's model fitting.
    oof = cross_val_predict(
        chosen,
        x_train,
        y_train,
        cv=make_cv(),
        method="predict_proba",
        n_jobs=1,
    )[:, 1]
    threshold = choose_threshold(y_train, oof)
    # Leakage is measured on training folds only and never enters the saved model.
    selected_model = clone(chosen.named_steps["model"])
    leakage_x = training.loc[:, [*FEATURE_NAMES, "fee_charged"]]
    leakage_scores = cross_validate(
        build_pipeline(selected_model, include_leakage=True),
        leakage_x,
        y_train,
        cv=make_cv(),
        scoring="roc_auc",
        n_jobs=1,
        error_score="raise",
    )["test_score"]
    honest_scores = cross_validate(
        chosen,
        x_train,
        y_train,
        cv=make_cv(),
        scoring="roc_auc",
        n_jobs=1,
        error_score="raise",
    )["test_score"]
    chosen.fit(x_train, y_train)
    # Diagnostic only: compare an accepted category with an unseen spelling.
    spelling_example = pd.DataFrame(
        [
            {
                "lead_days": 7,
                "age": 46,
                "prior_no_shows": 0,
                "hour": 11,
                "first_visit": 0,
                "distance_km": 5.0,
                "weekday": "mon",
                "booking": "phone",
                "insurance": "private",
                "treatment": "physio",
            }
        ],
        columns=list(FEATURE_NAMES),
    )
    valid_probability = float(chosen.predict_proba(spelling_example)[0, 1])
    spelling_example["weekday"] = "Mon"
    unknown_probability = float(chosen.predict_proba(spelling_example)[0, 1])
    spelling_check = {
        "field": "weekday",
        "accepted_spelling": "mon",
        "accepted_probability": valid_probability,
        "unseen_spelling": "Mon",
        "unseen_encoder_probability": unknown_probability,
        "note": "Raw-pipeline diagnostic only; Appointment must reject Mon.",
    }
    # The held-out test features are scored only after all choices are fixed.
    classes = list(chosen.classes_)
    test_probability = chosen.predict_proba(test.loc[:, list(FEATURE_NAMES)])[
        :, classes.index(1)
    ]
    test_result = summarize_predictions(test[TARGET], test_probability, threshold)
    intervals = savings_intervals(test[TARGET], test_probability, threshold, seed=SEED)
    tuning = {
        "scoring": "roc_auc",
        "combinations": len(search.cv_results_["params"]),
        "best_search_params": settings,
        "untuned_mean": untuned_mean,
        "untuned_std": float(winner["roc_auc_std"]),
        "tuned_mean": tuned_mean,
        "tuned_std": tuned_std,
        "improvement": tuned_mean - untuned_mean,
        "improvement_exceeds_untuned_std": bool(
            tuned_mean - untuned_mean > float(winner["roc_auc_std"])
        ),
        "used_tuned": use_tuned,
    }
    leakage = {
        "honest_roc_auc_mean": float(np.mean(honest_scores)),
        "honest_roc_auc_std": float(np.std(honest_scores, ddof=1)),
        "leaked_roc_auc_mean": float(np.mean(leakage_scores)),
        "leaked_roc_auc_std": float(np.std(leakage_scores, ddof=1)),
    }
    report = {
        "seed": SEED,
        "n_training": len(training),
        "n_test": len(test),
        "training_no_show_rate": float(y_train.mean()),
        "selection_metric": (
            "highest training CV mean ROC-AUC among non-dummy candidates"
        ),
        "selected_model": winner_name,
        "threshold": threshold,
        "threshold_source": (
            "minimum training out-of-fold cost on 0.05-0.95 grid, step 0.01"
        ),
        "oof_threshold_cost": float(
            threshold_costs(y_train, oof)
            .query("threshold == @threshold")["cost"]
            .iloc[0]
        ),
        "cost_rule": {"call_cost": CALL_COST, "miss_cost": MISS_COST},
        "category_spelling_check": spelling_check,
        "tuning": tuning,
        "leakage": leakage,
        "test_results": test_result,
        "test_savings_95_percent_bootstrap_intervals": intervals,
        "cv_selection_caveat": (
            "Model family and settings were selected using the training folds. "
            "The same folds were reused for OOF threshold choice, "
            "so training selection "
            "scores are not a nested-CV estimate. "
            "The held-out test was not used to choose."
        ),
    }
    info = {
        "sklearn_version": sklearn.__version__,
        "python_version": "3.12",
        "seed": SEED,
        "threshold": threshold,
        "selected_model": winner_name,
        "numeric_features": list(NUMERIC_FEATURES),
        "categorical_features": list(CATEGORICAL_FEATURES),
        "feature_names": list(FEATURE_NAMES),
        "class_labels": {"attend": 0, "no_show": 1},
        "best_settings": chosen.named_steps["model"].get_params(),
        "test_results": test_result,
        "cost_rule": report["cost_rule"],
    }
    models = PROJECT_ROOT / "models"
    reports = PROJECT_ROOT / "reports"
    models.mkdir(exist_ok=True)
    reports.mkdir(exist_ok=True)
    joblib.dump(chosen, models / "no_show.joblib")
    _write_json(models / "model_info.json", info)
    _write_json(reports / "metrics.json", report)
    comparison.to_csv(reports / "cv_results.csv", index=False)
    pd.DataFrame(search.cv_results_).to_csv(reports / "grid_search.csv", index=False)
    threshold_costs(y_train, oof).to_csv(reports / "threshold_costs.csv", index=False)
    print(comparison.to_string(index=False))
    print(
        json.dumps(
            {
                "selected_model": winner_name,
                "threshold": threshold,
                "test_results": test_result,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
