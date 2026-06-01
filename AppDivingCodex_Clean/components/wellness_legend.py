import streamlit as st

def render_wellness_legend():
    """Renders the global reference legend for Wellness."""
    st.markdown("### Escala de Referencia (1-7)")
    legend_data = {
        "1 (Óptimo)": "😃",
        "2": "🙂",
        "3": "🙂",
        "4 (Neutral)": "😐",
        "5": "😟",
        "6": "😫",
        "7 (Crítico)": "😵"
    }
    cols = st.columns(len(legend_data))
    for i, (label, emoji) in enumerate(legend_data.items()):
        cols[i].metric(label, emoji)
