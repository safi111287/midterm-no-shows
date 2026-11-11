"""Maple Physio Gradio page; implementation owned by Art."""

import gradio as gr

from noshows.appointment import (
    Appointment,
    CATEGORY_RULES,
    FEATURE_NAMES,
    NUMERIC_RULES,
)
from noshows.predictor import NoShowPredictor

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
    with gr.Blocks(title="Maple Physio No-Show Check") as demo:
        gr.Markdown("# Maple Physio No-Show Check")
        gr.Markdown(
            "Enter appointment details to estimate no-show risk "
            "and determine whether a confirmation call is recommended."
        )

        # TODO: Add exactly ten input components.
        # Follow FEATURE_NAMES order from noshows.appointment.
        # Read numeric limits and categories from that module.
        # Use precision=None for numeric fields.
        # Leave distance_km blank by default.

        submit_button = gr.Button("Assess Booking")

        action_output = gr.Textbox(
            label="Action / Input Error",
            interactive=False,
        )

        probability_output = gr.Textbox(
            label="No-Show Probability",
            interactive=False,
        )

        warning_output = gr.Textbox(
            label="Warning",
            interactive=False,
        )

        # TODO: Connect submit_button.click(
        #     fn=assess_booking,
        #     inputs=[...],
        #     outputs=[
        #         action_output,
        #         probability_output,
        #         warning_output,
        #     ],
        # )

    return demo

    raise NotImplementedError("Implement the Gradio Blocks page.")


if __name__ == "__main__":
    demo = build_demo()
    demo.launch()
