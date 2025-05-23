import gradio as gr

# Colori del brand DualMe
DUAL_COLORS = {
    "primary": "#051860",  # Blu Dual
    "secondary": "#eccbc8",  # Rosa Me
    "primary_rgba": "rgba(5,24,96,255)",
    "secondary_rgba": "rgba(236,203,200,255)"
}

def get_theme():
    """Restituisce il tema personalizzato per l'interfaccia Gradio"""
    return gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="pink",
        neutral_hue="slate",
        font=["Inter", "sans-serif"],
        font_mono=["JetBrains Mono", "monospace"],
        radius_size="md",
        text_size="md",
        spacing_size="md",
        primary_color=DUAL_COLORS["primary"],
        secondary_color=DUAL_COLORS["secondary"],
        background_fill_primary="white",
        background_fill_secondary="white",
        block_background_fill="white",
        block_title_text_color=DUAL_COLORS["primary"],
        block_label_text_color=DUAL_COLORS["primary"],
        button_primary_background_fill=DUAL_COLORS["primary"],
        button_primary_text_color="white",
        button_secondary_background_fill=DUAL_COLORS["secondary"],
        button_secondary_text_color=DUAL_COLORS["primary"],
        border_color_primary=DUAL_COLORS["primary"],
        border_color_secondary=DUAL_COLORS["secondary"],
        slider_color=DUAL_COLORS["primary"],
        checkbox_background_color=DUAL_COLORS["secondary"],
        checkbox_border_color=DUAL_COLORS["primary"],
        checkbox_check_color=DUAL_COLORS["primary"],
        radio_background_color=DUAL_COLORS["secondary"],
        radio_border_color=DUAL_COLORS["primary"],
        radio_check_color=DUAL_COLORS["primary"]
    )