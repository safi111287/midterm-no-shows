"""Deployment checks to be implemented by Rafael."""


def test_saved_model_and_metadata_exist():
    """Check models/no_show.joblib and models/model_info.json from repository root."""
    raise NotImplementedError("Implement model-file existence checks.")


def test_sklearn_versions_agree():
    """Installed sklearn == metadata sklearn_version == requirements.txt pin."""
    raise NotImplementedError("Implement sklearn version agreement.")


def test_runtime_dependencies_are_pinned_and_exclude_gradio():
    """Check exact scikit-learn/pandas/joblib pins; Gradio belongs in dev and SDK."""
    raise NotImplementedError("Implement runtime dependency checks.")


def test_gradio_sdk_matches_installed_version():
    """Parse README front matter and compare sdk_version with installed Gradio."""
    raise NotImplementedError("Implement Gradio SDK check.")


def test_python_versions_agree():
    """Compare Python major.minor in README, conda, metadata, and runtime."""
    raise NotImplementedError("Implement Python 3.12 agreement check.")


def test_space_uses_gradio_and_app_file_exists():
    """Check sdk is gradio and the README app_file path exists."""
    raise NotImplementedError("Implement Space entry-point check.")


def test_dev_requirements_include_runtime_and_pinned_tools():
    """Check -r requirements.txt and exact pins for gradio, pytest, and ruff."""
    raise NotImplementedError("Implement development dependency checks.")


def test_generated_reports_exist():
    """Check cv_results.csv and metrics.json, including test costs and leakage."""
    raise NotImplementedError("Implement report existence checks.")
