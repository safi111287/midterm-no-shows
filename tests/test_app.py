"""Interface tests to be implemented by Art."""


def test_valid_submission_displays_action_and_probability(valid_booking):
    """Submit FEATURE_NAMES-ordered values and check the three output strings."""
    raise NotImplementedError("Implement valid form test.")


def test_invalid_submission_clears_old_results(valid_booking):
    """Change age to an invalid value; display Input error and blank other outputs."""
    raise NotImplementedError("Implement invalid form test.")


def test_blank_distance_displays_warning(valid_booking):
    """Submit distance=None; keep prediction and show the median-imputation warning."""
    raise NotImplementedError("Implement missing-distance form test.")
