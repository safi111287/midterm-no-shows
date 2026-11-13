"""Deployment checks to be implemented by Rafael."""

import json
import re
import sys
from pathlib import Path

import gradio
import sklearn


def test_saved_model_and_metadata_exist():
    """Check models/no_show.joblib and models/model_info.json from repository root."""

    assert Path("models/no_show.joblib").exists()
    assert Path("models/model_info.json").exists()


def test_sklearn_versions_agree():
    """Installed sklearn == metadata sklearn_version == requirements.txt pin."""

    with open("models/model_info.json") as f:
        metadata = json.load(f)

    metadata_version = metadata["sklearn_version"]

    with open("requirements.txt") as f:
        requirements = f.read()

    requirement_version = None

    for line in requirements.splitlines():
        if line.startswith("scikit-learn=="):
            requirement_version = line.split("==")[1]

    installed_version = sklearn.__version__

    assert requirement_version is not None
    assert metadata_version == requirement_version
    assert installed_version == requirement_version


def test_runtime_dependencies_are_pinned_and_exclude_gradio():
    """Check that runtime dependencies are pinned and Gradio is excluded."""

    requirements = Path("requirements.txt").read_text(encoding="utf-8")

    requirement_lines = [
        line.strip()
        for line in requirements.splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]

    assert requirement_lines, "requirements.txt must not be empty"

    for line in requirement_lines:
        assert re.match(r"^[A-Za-z0-9_.-]+==", line), (
            f"Dependency is not pinned with ==: {line}"
        )

    assert not any(
        re.match(r"(?i)^gradio(?:==|$)", line) for line in requirement_lines
    ), "Gradio must not be included in runtime requirements."


def test_gradio_sdk_matches_installed_version():
    """Parse README front matter and compare sdk_version with installed Gradio."""

    readme = Path("README.md").read_text(encoding="utf-8")

    sdk_version = None

    for line in readme.splitlines():
        if line.startswith("sdk_version:"):
            sdk_version = line.split(":")[1].strip()

    assert sdk_version is not None
    assert sdk_version == gradio.__version__


def test_python_versions_agree():
    """Compare Python major.minor in README, conda, metadata, and runtime."""

    readme = Path("README.md").read_text(encoding="utf-8")

    readme_python = None

    for line in readme.splitlines():
        if line.startswith("python_version:"):
            readme_python = line.split(":", 1)[1].strip().replace('"', "")

    environment = Path("environment.yml").read_text(encoding="utf-8")

    environment_python = None

    for line in environment.splitlines():
        line = line.strip()
        if line.startswith("- python="):
            environment_python = line.split("=", 1)[1].strip()

    with open("models/model_info.json") as f:
        metadata = json.load(f)

    metadata_python = metadata["python_version"]

    runtime_python = f"{sys.version_info.major}.{sys.version_info.minor}"

    assert readme_python == "3.12"
    assert environment_python == "3.12"
    assert metadata_python == "3.12"
    assert runtime_python == "3.12"


def test_space_uses_gradio_and_app_file_exists():
    """Check sdk is gradio and the README app_file path exists."""

    readme = Path("README.md").read_text(encoding="utf-8")

    sdk = None
    app_file = None

    for line in readme.splitlines():
        if line.startswith("sdk:"):
            sdk = line.split(":", 1)[1].strip()

        if line.startswith("app_file:"):
            app_file = line.split(":", 1)[1].strip()

    assert sdk == "gradio"
    assert app_file is not None
    assert Path(app_file).is_file()


def test_dev_requirements_include_runtime_and_pinned_tools():
    """Check -r requirements.txt and exact pins for gradio, pytest, and ruff."""

    requirements = Path("requirements-dev.txt").read_text(encoding="utf-8")

    requirement_lines = [
        line.strip()
        for line in requirements.splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]

    assert "-r requirements.txt" in requirement_lines
    assert "gradio==6.31.0" in requirement_lines
    assert "pytest==8.4.2" in requirement_lines
    assert "ruff==0.14.3" in requirement_lines


def test_generated_reports_exist():
    """Check cv_results.csv and metrics.json, including test costs and leakage."""

    cv_results = Path("reports/cv_results.csv")
    metrics_file = Path("reports/metrics.json")

    assert cv_results.is_file()
    assert metrics_file.is_file()

    metrics = json.loads(metrics_file.read_text(encoding="utf-8"))

    assert "test_results" in metrics
    assert "leakage" in metrics

    test_results = metrics["test_results"]
    costs = test_results["costs"]

    assert "call_nobody" in costs
    assert "call_everybody" in costs
    assert "model" in costs

    assert "savings_vs_nobody" in test_results
    assert "savings_vs_everybody" in test_results
    assert "beats_both" in test_results

    leakage = metrics["leakage"]

    assert "honest_roc_auc_mean" in leakage
    assert "leaked_roc_auc_mean" in leakage

    assert leakage["leaked_roc_auc_mean"] > leakage["honest_roc_auc_mean"]
