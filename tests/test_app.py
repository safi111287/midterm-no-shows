"""Tests for booking form behavior and Gradio interface wiring."""

import gradio as gr
import pytest

import app
from noshows.appointment import (
    CATEGORY_RULES,
    FEATURE_NAMES,
    MISSING_DISTANCE_WARNING,
    NUMERIC_RULES,
    Appointment,
)
from noshows.predictor import NoShowPredictor


@pytest.fixture
def form_predictor(monkeypatch):
    predictor = NoShowPredictor()
    monkeypatch.setattr(app, "predictor", predictor)
    return predictor


@pytest.fixture
def demo(form_predictor, monkeypatch):
    calls = []

    def create_predictor():
        calls.append(True)
        return form_predictor

    monkeypatch.setattr(app, "NoShowPredictor", create_predictor)
    interface = app.build_demo()

    try:
        assert len(calls) == 1
        yield interface
    finally:
        interface.close()


def submit(booking):
    return app.assess_booking(*(booking[name] for name in FEATURE_NAMES))


def submission_config(demo):
    config = demo.get_config_file()
    dependencies = [
        dependency
        for dependency in config["dependencies"]
        if dependency.get("backend_fn")
        and len(dependency["inputs"]) == len(FEATURE_NAMES)
        and len(dependency["outputs"]) == 3
    ]
    assert len(dependencies) == 1

    components = {component["id"]: component for component in config["components"]}
    return dependencies[0], components


def test_valid_submission_displays_action_and_probability(
    valid_booking, form_predictor
):
    expected = form_predictor.predict(Appointment(**valid_booking))

    action, probability, warning = submit(valid_booking)

    expected_action = "Call patient" if expected["call_patient"] else "No call needed"
    assert action == expected_action
    assert probability == f"{expected['no_show_probability']:.2%}"
    assert warning == ""
    assert all(isinstance(value, str) for value in (action, probability, warning))


def test_invalid_submission_clears_old_results(valid_booking, form_predictor):
    assert submit(valid_booking)[1] != ""

    action, probability, warning = submit({**valid_booking, "age": 11})

    assert action.startswith("Input error:")
    assert "age" in action
    assert probability == ""
    assert warning == ""


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("age", 40.5),
        ("lead_days", -1),
        ("hour", 18),
        ("first_visit", 2),
        ("weekday", "sun"),
        ("booking", "email"),
        ("insurance", "unknown"),
        ("treatment", "unknown"),
    ],
)
def test_invalid_fields_display_named_errors(
    valid_booking, form_predictor, field, value
):
    action, probability, warning = submit({**valid_booking, field: value})

    assert action.startswith("Input error:")
    assert field in action
    assert probability == ""
    assert warning == ""


@pytest.mark.parametrize("distance", [None, ""])
def test_blank_distance_displays_warning(valid_booking, form_predictor, distance):
    booking = {**valid_booking, "distance_km": distance}

    action, probability, warning = submit(booking)

    assert action in ("Call patient", "No call needed")
    assert probability.endswith("%")
    assert warning == MISSING_DISTANCE_WARNING


def test_valid_distance_clears_previous_warning(valid_booking, form_predictor):
    missing = {**valid_booking, "distance_km": None}
    assert submit(missing)[2] == MISSING_DISTANCE_WARNING

    action, probability, warning = submit(valid_booking)

    assert action in ("Call patient", "No call needed")
    assert probability.endswith("%")
    assert warning == ""


def test_repeated_submissions_reuse_predictor(
    valid_booking, form_predictor, monkeypatch
):
    def unexpected_load(*args, **kwargs):
        pytest.fail("The form must reuse its existing predictor.")

    monkeypatch.setattr(app, "NoShowPredictor", unexpected_load)
    monkeypatch.setattr("joblib.load", unexpected_load)

    first = submit(valid_booking)
    second = submit(valid_booking)

    assert first == second
    assert app.predictor is form_predictor


def test_build_demo_returns_blocks_and_reuses_predictor(
    demo, form_predictor, valid_booking, monkeypatch
):
    assert isinstance(demo, gr.Blocks)
    assert app.predictor is form_predictor

    def unexpected_load(*args, **kwargs):
        pytest.fail("Submitting the form must not reload the model.")

    monkeypatch.setattr(app, "NoShowPredictor", unexpected_load)
    monkeypatch.setattr("joblib.load", unexpected_load)

    assert submit(valid_booking)[1].endswith("%")
    assert submit(valid_booking)[1].endswith("%")


def test_interface_inputs_follow_feature_order(demo):
    dependency, components = submission_config(demo)

    assert len(dependency["inputs"]) == 10

    for name, component_id in zip(FEATURE_NAMES, dependency["inputs"], strict=True):
        component = components[component_id]
        props = component["props"]
        expected_label = name.replace("_", " ").title()

        assert props["label"].startswith(expected_label)

        if name in NUMERIC_RULES:
            assert component["type"] == "number"
        else:
            assert component["type"] == "dropdown"


def test_numeric_fields_show_limits_and_preserve_fractions(demo):
    dependency, components = submission_config(demo)

    for name, component_id in zip(FEATURE_NAMES, dependency["inputs"], strict=True):
        if name not in NUMERIC_RULES:
            continue

        minimum, maximum, whole = NUMERIC_RULES[name]
        props = components[component_id]["props"]

        assert str(minimum) in props["label"]
        assert str(maximum) in props["label"]
        assert props.get("precision") is None

        if whole:
            assert "whole number required" in props["label"]


def test_dropdown_choices_match_shared_rules(demo):
    dependency, components = submission_config(demo)

    for name, component_id in zip(FEATURE_NAMES, dependency["inputs"], strict=True):
        if name not in CATEGORY_RULES:
            continue

        choices = components[component_id]["props"]["choices"]
        values = [
            choice[1] if isinstance(choice, (list, tuple)) else choice
            for choice in choices
        ]

        assert values == list(CATEGORY_RULES[name])


def test_distance_is_optional_and_blank_by_default(demo):
    dependency, components = submission_config(demo)
    index = list(FEATURE_NAMES).index("distance_km")
    props = components[dependency["inputs"][index]]["props"]

    assert props.get("value") is None
    assert "optional" in props["label"].lower()


def test_submit_button_connects_callback_and_three_outputs(demo):
    dependency, components = submission_config(demo)

    click_targets = [target for target in dependency["targets"] if target[1] == "click"]
    assert len(click_targets) == 1

    button = components[click_targets[0][0]]
    assert button["type"] == "button"
    assert button["props"]["value"] == "Assess Booking"

    callback = demo.fns[dependency["id"]]
    assert callback.fn is app.assess_booking

    expected_labels = [
        "Action / Input Error",
        "No-Show Probability",
        "Warning Text",
    ]
    outputs = [components[component_id] for component_id in dependency["outputs"]]

    assert [output["props"]["label"] for output in outputs] == expected_labels
    assert all(output["type"] == "textbox" for output in outputs)
    assert all(output["props"]["interactive"] is False for output in outputs)
