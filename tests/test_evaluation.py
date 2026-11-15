"""Tests of hand-calculated costs and training-only threshold decisions."""

import numpy as np
import pytest

from noshows.evaluation import (
    choose_threshold,
    summarize_predictions,
    threshold_costs,
    total_cost,
)


def test_cost_matches_hand_worked_example():
    # Two calls cost 16; two uncalled no-shows cost 70; total 86.
    assert total_cost([1, 0, 1, 1, 0], [1, 1, 0, 0, 0]) == 86


@pytest.mark.parametrize(
    ("truth", "flags", "expected"),
    [
        ([0, 1, 1], [0, 0, 0], 70),
        ([0, 1, 1], [1, 1, 1], 24),
        ([0, 0], [0, 0], 0),
        ([1, 1], [1, 1], 16),
    ],
)
def test_call_none_and_call_all_costs(truth, flags, expected):
    assert total_cost(truth, flags) == expected


def test_calling_nobody_is_cheapest_when_everyone_attends():
    truth = [0, 0, 0]
    assert total_cost(truth, [0, 0, 0]) < total_cost(truth, [1, 1, 1])


def test_calling_everybody_is_cheapest_when_everyone_misses():
    truth = [1, 1, 1]
    assert total_cost(truth, [1, 1, 1]) < total_cost(truth, [0, 0, 0])


@pytest.mark.parametrize(
    ("call_cost", "miss_cost", "expected_calls"),
    [(100, 1, 0), (1, 1000, 4)],
)
def test_threshold_extremes(call_cost, miss_cost, expected_calls):
    truth, probabilities = np.array([1, 1, 1, 1]), np.array([0.1, 0.3, 0.6, 0.9])
    threshold = choose_threshold(
        truth, probabilities, call_cost=call_cost, miss_cost=miss_cost
    )
    assert int((probabilities >= threshold).sum()) == expected_calls


def test_threshold_minimizes_grid_cost_and_uses_inclusive_boundary():
    truth, probabilities = [0, 1, 1, 0], np.array([0.1, 0.3, 0.7, 0.2])
    table = threshold_costs(truth, probabilities)
    threshold = choose_threshold(truth, probabilities)
    assert len(table) == 91
    assert (table.threshold.min(), table.threshold.max()) == (0.05, 0.95)
    assert total_cost(truth, probabilities >= threshold) == table.cost.min()


@pytest.mark.parametrize(
    "probabilities", [[-0.1, 0.5], [0.5, 1.1], [np.nan, 0.4], [np.inf, 0.4]]
)
def test_threshold_rejects_invalid_probabilities(probabilities):
    with pytest.raises(ValueError, match="probabilities"):
        choose_threshold([0, 1], probabilities)


def test_cost_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="length"):
        total_cost([0, 1], [1])


def test_summary_reports_all_three_costs():
    result = summarize_predictions([0, 1, 1, 0], [0.1, 0.8, 0.6, 0.2], 0.5)
    assert result["costs"] == {
        "call_nobody": 70.0,
        "call_everybody": 32.0,
        "model": 16.0,
    }
    assert result["confusion_matrix"] == [[2, 0], [0, 2]]
    assert result["beats_both"]
