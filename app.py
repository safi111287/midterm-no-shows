"""Maple Physio Gradio page; implementation owned by Art."""

import gradio as gr

# ZeroGPU compatibility; local installations do not require spaces.
try:
    import spaces
except ImportError:
    spaces = None

if spaces is not None:

    @spaces.GPU
    def gpu_placeholder():
        """Provide the GPU entry point required by ZeroGPU hosting."""
        return None


def assess_booking(*values) -> tuple[str, str, str]:
    """Return action/error, no-show probability, and warning text.

    Map values in Appointment's FEATURE_NAMES order, construct Appointment,
    and call one reusable NoShowPredictor instance. Catch ValueError and
    clear probability/warning outputs on invalid input. Do not read model
    files or implement prediction/validation rules in this module.
    """
    raise NotImplementedError("Implement the class-based form callback.")


def build_demo() -> gr.Blocks:
    """Build inputs from the rules exposed by noshows.appointment.

    Use one predictor instance for all submissions, not one per click.
    Display numeric limits, keep distance blank by default, and preserve
    fractional input so Appointment can reject non-whole numbers.
    Show a plain call/no-call action, a percentage, and missing-distance warning.
    """
    raise NotImplementedError("Implement the Gradio Blocks page.")


if __name__ == "__main__":
    demo = build_demo()
    demo.launch()
