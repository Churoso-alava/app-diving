"""
app.py — Punto de entrada principal para AppDivingCodex.
Reconstruido de forma modular según el plan de auditoría.
"""
import streamlit as st
import os
import pandas as pd
from datetime import date

from core.fuzzy_engine import construir_motor_fuzzy
from core.services import pipeline_batch, pipeline_diagnostico, calcular_historial_fatiga, get_vmp_history
from data.db import cargar_sesiones_raw, cargar_atletas, insertar_sesion
from ui.charts import fig_vmp_tendencia, fig_semaforo_barras, fig_semaforo_historico
from ui.auth_session import is_authenticated, get_role, get_user_id
from ui.auth_pages import render_login, render_user_info

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN Y CARGA DE RECURSOS
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="AppDivingCodex v4.5",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Integración de secretos con variables de entorno para data/db.py
if "SUPABASE_URL" in st.secrets:
    os.environ["SUPABASE_URL"] = st.secrets["SUPABASE_URL"]
if "SUPABASE_KEY" in st.secrets:
    os.environ["SUPABASE_KEY"] = st.secrets["SUPABASE_KEY"]

# ── AUTENTICACIÓN ─────────────────────────────────────────────────────────────
if not is_authenticated():
    render_login()
    st.stop()

render_user_info()

@st.cache_resource
def load_fuzzy_engine():
    """Cachea el motor fuzzy para evitar reconstrucción constante."""
    _, simulador = construir_motor_fuzzy()
    return simulador

@st.cache_data(ttl=600)
def load_data():
    """Carga datos crudos desde la base de datos."""
    return cargar_sesiones_raw()

# ─────────────────────────────────────────────────────────────────────────────
# MAIN UI
# ─────────────────────────────────────────────────────────────────────────────

simulador = load_fuzzy_engine()
df_raw = load_data()

if df_raw.empty:
    st.warning("⚠️ No se encontraron datos de sesiones VMP en la base de datos.")
    st.stop()

rol = get_role()

if rol == "staff":
    # ── VISTA STAFF: DASHBOARD COMPLETO ───────────────────────────────────────
    tab1, tab2, tab3 = st.tabs(["📊 Dashboard Global", "👤 Análisis Individual", "📝 Registro"])

    # ── TAB 1: DASHBOARD GLOBAL ──────────────────────────────────────────────────
    with tab1:
        st.header("Dashboard de Estado del Equipo")
        
        with st.spinner("Calculando estados..."):
            df_batch = pipeline_batch(df_raw, simulador)
            
        if not df_batch.empty:
            # Métricas resumen
            c1, c2, c3, c4 = st.columns(4)
            total = len(df_batch)
            optimos = len(df_batch[df_batch["estado"].str.contains("ÓPTIMO")])
            criticos = len(df_batch[df_batch["estado"].str.contains("CRÍTICO")])
            
            c1.metric("Total Atletas", total)
            c2.metric("🟢 Óptimos", optimos)
            c3.metric("🔴 Críticos", criticos)
            c4.metric("📅 Última Sesión", df_raw["fecha"].max())
            
            # Gráfico semáforo
            df_semaforo = df_batch.rename(columns={"atleta": "nombre", "indice_fatiga": "score"})
            fig_global = fig_semaforo_barras(df_semaforo)
            st.plotly_chart(fig_global, use_container_width=True)
            
            # Tabla detallada
            st.subheader("Detalle por Atleta")
            st.dataframe(
                df_batch[["atleta", "estado", "indice_fatiga", "acwr", "delta_pct", "calidad_dato", "accion"]],
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info("No hay atletas con datos suficientes (mínimo 4 sesiones).")

    # ── TAB 2: ANÁLISIS INDIVIDUAL ────────────────────────────────────────────────
    with tab2:
        st.header("Seguimiento Individual")
        
        atletas = cargar_atletas()
        atleta_sel = st.selectbox("Seleccione un atleta", [""] + atletas)
        
        if atleta_sel:
            res = pipeline_diagnostico(atleta_sel, df_raw, simulador)
            
            if res:
                # Encabezado de estado
                c1, c2 = st.columns([1, 3])
                with c1:
                    st.markdown(f"### {res['estado']}")
                    st.metric("Índice de Fatiga", f"{res['indice_fatiga']}%")
                with c2:
                    st.success(f"**Acción recomendada:** {res['accion']}")
                    if res['advertencias']:
                        for adv in res['advertencias']:
                            st.warning(adv)
                
                st.markdown("---")
                
                # Gráficos
                col_g1, col_g2 = st.columns(2)
                
                with col_g1:
                    # Encontrar los datos calculados para este atleta en df_batch o llamar directamente
                    # (fig_vmp_tendencia espera df con métricas temporales)
                    df_atleta_history = get_vmp_history(df_raw, atleta_sel)
                    fig_vmp = fig_vmp_tendencia(df_atleta_history, atleta_sel, res["delta_pct"])
                    st.plotly_chart(fig_vmp, use_container_width=True)
                    
                with col_g2:
                    df_hist = calcular_historial_fatiga(df_raw, atleta_sel, simulador)
                    fig_hist = fig_semaforo_historico(df_hist, f"Evolución de Fatiga - {atleta_sel}")
                    st.plotly_chart(fig_hist, use_container_width=True)
                    
                # Métricas biomecánicas
                st.subheader("Variables Biomecánicas (Inputs Fuzzy)")
                m1, m2, m3, m4, m5 = st.columns(5)
                m1.metric("ACWR", f"{res['acwr']:.2f}")
                m2.metric("Delta %", f"{res['delta_pct']:.1f}%")
                m3.metric("Z-Score Meso", f"{res['z_meso']:.2f}")
                m4.metric("Beta Aguda", f"{res['beta_aguda']:.3f}")
                m5.metric("Beta 28d", f"{res['beta_28']:.3f}")
                
                with st.expander("Ver Contexto Científico"):
                    st.write(res["contexto_cientifico"])
                    if res["nota_swc"]:
                        st.info(res["nota_swc"])

    # ── TAB 3: REGISTRO ──────────────────────────────────────────────────────────
    with tab3:
        st.header("Registrar Nueva Sesión")
        
        with st.form("form_registro"):
            col1, col2 = st.columns(2)
            with col1:
                atleta_reg = st.selectbox("Atleta", cargar_atletas())
                fecha_reg = st.date_input("Fecha", date.today())
            with col2:
                vmp_reg = st.number_input("VMP Hoy (m/s)", min_value=0.1, max_value=2.5, value=1.0, step=0.01)
                vmp_ref_reg = st.number_input("VMP Referencia (opcional)", min_value=0.0, max_value=2.5, value=0.0, step=0.01)
            
            notas_reg = st.text_area("Notas")
            
            submitted = st.form_submit_button("Guardar Sesión")
            
            if submitted:
                v_ref = vmp_ref_reg if vmp_ref_reg > 0 else None
                ok, msg = insertar_sesion(atleta_reg, fecha_reg, vmp_reg, v_ref, notas_reg)
                if ok:
                    st.success(msg)
                    st.cache_data.clear()
                else:
                    st.error(msg)

elif rol == "deportista":
    # ── VISTA DEPORTISTA: SOLO SUS DATOS ─────────────────────────────────────
    id_deportivo = get_user_id()
    st.header(f"Hola, {id_deportivo}")
    
    # El id_deportivo debe coincidir con el nombre en la tabla sesiones_vmp
    res = pipeline_diagnostico(id_deportivo, df_raw, simulador)
    
    if res:
        # Reutilizar parte de la UI de análisis individual
        c1, c2 = st.columns([1, 3])
        with c1:
            st.markdown(f"### {res['estado']}")
            st.metric("Índice de Fatiga", f"{res['indice_fatiga']}%")
        with c2:
            st.info(f"**Indicación:** {res['accion']}")
            if res['advertencias']:
                for adv in res['advertencias']:
                    st.warning(adv)
        
        st.markdown("---")
        
        col_g1, col_g2 = st.columns(2)
        with col_g1:
            df_atleta_history = get_vmp_history(df_raw, id_deportivo)
            fig_vmp = fig_vmp_tendencia(df_atleta_history, id_deportivo, res["delta_pct"])
            st.plotly_chart(fig_vmp, use_container_width=True)
        with col_g2:
            df_hist = calcular_historial_fatiga(df_raw, id_deportivo, simulador)
            fig_hist = fig_semaforo_historico(df_hist, "Mi Historial de Fatiga")
            st.plotly_chart(fig_hist, use_container_width=True)
    else:
        st.warning("Todavía no tenemos datos suficientes para generar tu diagnóstico (mínimo 4 sesiones).")
        st.info("Sigue entrenando y registrando tus sesiones con el equipo.")

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR (Común)
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("🌊 AppDivingCodex")
    st.markdown("---")
    
    if st.button("🔄 Refrescar Datos"):
        st.cache_data.clear()
        st.rerun()

    st.markdown("---")
    st.info("Sistema de monitoreo de fatiga para clavados mediante motor fuzzy Mamdani.")
