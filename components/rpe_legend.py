import streamlit as st

def render_rpe_legend():
    """Renders the reference legend for RPE (1-10)."""
    st.markdown("### Escala de Esfuerzo Percibido (RPE 1-10)")
    
    # Legend mapping: Value -> (Emoji, Description)
    legend_data = {
        "1": ("🧘", "Muy liviano"),
        "2": ("🚶", "Muy liviano"),
        "3": ("😊", "Liviano"),
        "4": ("🙂", "Algo liviano"),
        "5": ("😐", "Moderado"),
        "6": ("😬", "Algo duro"),
        "7": ("😰", "Duro"),
        "8": ("😫", "Muy duro"),
        "9": ("🥵", "Extremo"),
        "10": ("💀", "Máximo")
    }
    
    # Use multiple rows if 10 columns are too many, but let's try one row first.
    # Alternatively, 2 rows of 5.
    
    cols = st.columns(10)
    for i, (val, (emoji, desc)) in enumerate(legend_data.items()):
        with cols[i]:
            st.markdown(f"<div style='text-align: center;' title='{desc}'><span style='font-size: 1.5rem;'>{emoji}</span><br><b>{val}</b></div>", unsafe_allow_html=True)
