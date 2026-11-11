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



    try:
        appointment = Appointment(
            **dict(zip(FEATURE_NAMES, values, strict=True))
        )
        result = predictor.predict(appointment)

        action = (
            "Call patient"
            if result["call_patient"]
            else "No call needed"
        )
        probability = f'{result["no_show_probability"]:.2%}'
        warning = "\n".join(result["warnings"])

        return action, probability, warning

    except ValueError as exc:
        return f"Input error: {exc}", "", ""

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
        gr.Markdown("Enter appointment details to assess no-show risk.")

        inputs = []

        for name in FEATURE_NAMES:
            if name in NUMERIC_RULES:
                minimum, maximum, whole_number = NUMERIC_RULES[name]

                label = (
                    f"{name.replace('_', ' ').title()} "
                    f"({minimum}–{maximum})"
                )

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


    raise NotImplementedError("Implement the Gradio Blocks page.")


if __name__ == "__main__":
    demo = build_demo()
    demo.launch()
