import streamlit as st
from datetime import date
from data.db import cargar_atletas, insertar_wellness
from components.wellness_legend import render_wellness_legend
import pandas as pd

def render_wellness_registro():
    st.subheader("Registro Wellness Hooper")
    render_wellness_legend()
    
    mode = st.radio("Modo de Registro", ["Individual", "Grupal"], horizontal=True, key="mode_well")
    
    atletas = cargar_atletas()
    if not atletas:
        st.warning("No hay atletas activos para registrar.")
        return

    if mode == "Individual":
        with st.form("form_wellness"):
            col1, col2 = st.columns(2)
            with col1:
                atleta = st.selectbox("Atleta", atletas)
                fecha = st.date_input("Fecha", date.today())
            
            st.write("---")
            col_s1, col_s2, col_s3 = st.columns(3)
            with col_s1:
                dolor = st.slider("Dolor", 1, 7, 4)
                estres = st.slider("Estrés", 1, 7, 4)
            with col_s2:
                fatiga = st.slider("Fatiga Hooper", 1, 7, 4)
                humor = st.slider("Humor", 1, 7, 4)
            with col_s3:
                sueno = st.slider("Sueño", 1, 7, 4)
                
            notas = st.text_area("Notas")
            
            submitted = st.form_submit_button("Registrar Wellness")
            
            if submitted:
                success, msg = insertar_wellness(
                    nombre=atleta,
                    fecha=fecha,
                    sueno=sueno,
                    fatiga_hooper=fatiga,
                    estres=estres,
                    dolor=dolor,
                    humor=humor,
                    notas=notas
                )
                if success:
                    st.session_state['last_wellness_msg'] = ('success', msg)
                else:
                    st.session_state['last_wellness_msg'] = ('error', msg)
                st.rerun()

    else:
        # Task 2 Implementation
        st.subheader("Registro Grupal")
        fecha_grupal = st.date_input("Fecha Grupal", date.today())
        
        df_grupal = pd.DataFrame({
            "Atleta": atletas,
            "Sueño": [4] * len(atletas),
            "Fatiga": [4] * len(atletas),
            "Estrés": [4] * len(atletas),
            "Dolor": [4] * len(atletas),
            "Humor": [4] * len(atletas)
        })
        
        edited_df = st.data_editor(df_grupal, use_container_width=True)
        # Save logic for Task 3

    # Mostrar mensaje persistente si existe
    if 'last_wellness_msg' in st.session_state:
        tipo, msg = st.session_state.pop('last_wellness_msg')
        if tipo == 'success':
            st.success(msg)
            st.balloons()
        else:
            st.error(msg)
