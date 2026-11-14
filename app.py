"""Gradio interface for appointment no-show predictions."""

import gradio as gr

from noshows.appointment import (
    CATEGORY_RULES,
    FEATURE_NAMES,
    NUMERIC_RULES,
    Appointment,
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
        """Provide the GPU entry point for ZeroGPU hosting."""
        return None


predictor = None


def assess_booking(*values) -> tuple[str, str, str]:
    """Return the call decision, no-show probability, and input warning."""
    try:
        appointment = Appointment(**dict(zip(FEATURE_NAMES, values, strict=True)))
        result = predictor.predict(appointment)

        action = "Call patient" if result["call_patient"] else "No call needed"
        probability = f"{result['no_show_probability']:.2%}"
        warning = "\n".join(result["warnings"])

        return action, probability, warning

    except ValueError as exc:
        return f"Input error: {exc}", "", ""


def build_demo() -> gr.Blocks:
    """Build the booking form and reuse one predictor for submissions."""
    global predictor
    predictor = NoShowPredictor()

    with gr.Blocks(title="Maple Physio No-Show Check") as demo:
        gr.Markdown("# Maple Physio No-Show Check")
        gr.Markdown("Enter appointment details to assess no-show risk.")

        inputs = []

        for name in FEATURE_NAMES:
            if name in NUMERIC_RULES:
                minimum, maximum, whole_number = NUMERIC_RULES[name]
                label = f"{name.replace('_', ' ').title()} ({minimum}–{maximum})"

                if whole_number:
                    label += " — whole number required"

                if name == "distance_km":
                    label += " — optional"

                component = gr.Number(
                    label=label,
                    value=None,
                    precision=None,
                )

            elif name in CATEGORY_RULES:
                component = gr.Dropdown(
                    choices=list(CATEGORY_RULES[name]),
                    label=name.replace("_", " ").title(),
                    value=None,
                )

            else:
                raise ValueError(f"Unknown feature: {name}")

            inputs.append(component)

        submit_button = gr.Button(
            "Assess Booking",
            variant="primary",
        )

        action_output = gr.Textbox(
            label="Action / Input Error",
            interactive=False,
        )
        probability_output = gr.Textbox(
            label="No-Show Probability",
            interactive=False,
        )
        warning_output = gr.Textbox(
            label="Warning Text",
            interactive=False,
        )

        submit_button.click(
            fn=assess_booking,
            inputs=inputs,
            outputs=[
                action_output,
                probability_output,
                warning_output,
            ],
        )

    return demo


if __name__ == "__main__":
    demo = build_demo()
    demo.launch()
