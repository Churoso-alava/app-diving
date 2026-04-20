import streamlit as st
import pandas as pd
from datetime import datetime

# Assume db.py exists in the root directory and has the necessary functions
# For now, we'll use placeholders for db functions if they are not directly accessible.
# In a real scenario, you'd import them like: from db import cargar_lesiones, insertar_lesion, actualizar_estado_lesion
try:
    from db import cargar_lesiones, insertar_lesion, actualizar_estado_lesion
except ImportError:
    st.warning("Could not import database functions from db.py. Using placeholder functions.")
    # Placeholder functions if db.py is not available or structured differently
    def cargar_lesiones():
        st.info("Placeholder: Fetching active lesions from DB...")
        # Return dummy data for now
        return pd.DataFrame({
            'id': ['uuid-1', 'uuid-2'],
            'atleta': ['Athlete A', 'Athlete B'],
            'fecha_inicio': [datetime(2024, 1, 1), datetime(2024, 1, 5)],
            'fecha_alta': [None, datetime(2024, 1, 15)],
            'zona_cuerpo': ['Rodilla', 'Tobillo'],
            'sistema': ['Musculoesquelético', 'Musculoesquelético'],
            'estado': ['Lesionado', 'Alta'],
            'notas': ['Dolor agudo', 'Esguince leve']
        })

    def insertar_lesion(atleta, fecha_inicio, zona_cuerpo, sistema, estado, notas):
        st.info(f"Placeholder: Inserting new lesion for {atleta}...")
        # In a real app, this would insert into the DB and return success/failure
        return True

    def actualizar_estado_lesion(lesion_id, nuevo_estado, fecha_alta=None):
        st.info(f"Placeholder: Updating lesion {lesion_id} to {nuevo_estado}...")
        # In a real app, this would update the DB
        return True

def render_tab_lesiones():
    st.title("Injury Tracking")

    # --- Form for Registering New Injuries ---
    st.header("Register New Injury")
    with st.form("new_injury_form"):
        atleta = st.text_input("Athlete Name")
        fecha_inicio = st.date_input("Start Date", datetime.now())
        zona_cuerpo = st.text_input("Body Part (e.g., Knee, Ankle)")
        sistema = st.selectbox(
            "Body System",
            ('Musculoesquelético', 'Cardiopulmonar', 'Neuromuscular', 'Tegumentario')
        )
        estado = st.selectbox(
            "Status",
            ('Lesionado', 'Sigue lesionado', 'Alta') # Note: 'Lesionado' and 'Sigue lesionado' might be redundant, but following plan.
        )
        notas = st.text_area("Notes")

        submitted = st.form_submit_button("Register Injury")
        if submitted:
            if atleta and fecha_inicio and zona_cuerpo and sistema and estado:
                try:
                    success = insertar_lesion(atleta, fecha_inicio, zona_cuerpo, sistema, estado, notas)
                    if success:
                        st.success("Injury registered successfully!")
                        # Optionally clear form or refresh list
                    else:
                        st.error("Failed to register injury. Please try again.")
                except Exception as e:
                    st.error(f"An error occurred: {e}")
            else:
                st.warning("Please fill in all required fields.")

    st.markdown("---")

    # --- List/Table for Active Injuries ---
    st.header("Active Injuries")
    try:
        lesiones_df = cargar_lesiones()
        # Filter for active injuries (those not marked as 'Alta')
        lesiones_activas_df = lesiones_df[lesiones_df['estado'] != 'Alta'].copy() # Use .copy() to avoid SettingWithCopyWarning

        if lesiones_activas_df.empty:
            st.info("No active injuries found.")
        else:
            # Add a column for actions, but display only relevant columns
            lesiones_activas_df['Action'] = '...' # Placeholder for buttons

            # Displaying relevant columns and providing update mechanism
            for index, row in lesiones_activas_df.iterrows():
                col1, col2, col3, col4, col5, col6 = st.columns([2, 2, 2, 1, 1, 1])
                with col1:
                    st.write(f"**Athlete:** {row['atleta']}")
                with col2:
                    st.write(f"**Start:** {row['fecha_inicio'].strftime('%Y-%m-%d') if pd.notna(row['fecha_inicio']) else 'N/A'}")
                with col3:
                    st.write(f"**Body Part:** {row['zona_cuerpo']}")
                with col4:
                    st.write(f"**System:** {row['sistema']}")
                with col5:
                    st.write(f"**Status:** {row['estado']}")
                with col6:
                    # Button to update status to 'Alta'
                    if st.button("Mark as Alta", key=f"alta_btn_{row['id']}"):
                        fecha_alta = datetime.now()
                        if actualizar_estado_lesion(row['id'], 'Alta', fecha_alta):
                            st.success(f"Marked '{row['atleta']}' as Alta.")
                            st.rerun() # Refresh to update the list
                        else:
                            st.error(f"Failed to update status for '{row['atleta']}'.")

                if row['notas']:
                    with st.expander("Notes"):
                        st.write(row['notas'])
                st.markdown("---") # Separator for each injury entry

    except Exception as e:
        st.error(f"An error occurred while fetching injuries: {e}")

if __name__ == "__main__":
    render_tab_lesiones()
