"""
app.py — Punto de entrada principal para AppDivingCodex.
Reconstruido de forma modular según el plan de auditoría.
"""
import streamlit as st
import os
import pandas as pd
from datetime import date

from core.fuzzy_engine import construir_motor_fuzzy
from core.services import pipeline_batch, pipeline_diagnostico, calcular_historial_fatiga, get_vmp_history, get_wellness_history
from components.tab_wellness_registro import render_wellness_registro
from components.rpe_legend import render_rpe_legend
from data.db import (
    cargar_sesiones_raw, cargar_atletas, insertar_sesion, get_last_db_error,
    insertar_carga_sesion, insertar_carga_sesion_batch, cargar_lesiones_activas
)
from components.tab_lesiones import render_tab_lesiones
from ui.charts import fig_vmp_tendencia, fig_semaforo_barras, fig_semaforo_historico, fig_fatiga_radial
from ui.charts_redesign import fig_vmp_tendencia_redesign, fig_fatiga_tendencia, fig_wellness_tendencia
from ui.auth_session import is_authenticated, get_role, get_user_id, get_atleta_vmp
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
for secret_name in (
    "SUPABASE_URL",
    "SUPABASE_SERVICE_ROLE_KEY",
    "SUPABASE_SECRET_KEY",
    "SUPABASE_KEY",
):
    if secret_name in st.secrets:
        os.environ[secret_name] = st.secrets[secret_name]

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
rol = get_role()

if df_raw.empty and rol != "staff":
    last_error = get_last_db_error()
    if last_error:
        st.error(last_error)
    else:
        st.warning("⚠️ No se encontraron datos de sesiones VMP en la base de datos.")
    st.stop()

if rol == "staff":
    # ── VISTA STAFF: DASHBOARD COMPLETO ───────────────────────────────────────
    tab1, tab2, tab3, tab4, tab_carga, tab_lesiones = st.tabs([
        "📊 Dashboard Global", "👤 Análisis Individual",
        "👟 SALTO CMJ", "📝 Wellness (Hooper)",
        "🏋️ Carga de Entrenamiento", "🩹 Lesiones",
    ])

    # ── TAB 1: DASHBOARD GLOBAL ──────────────────────────────────────────────────
    with tab1:
        st.header("Dashboard de Estado del Equipo")
        
        if df_raw.empty:
            last_error = get_last_db_error()
            if last_error:
                st.error(last_error)
            else:
                st.warning("⚠️ No se encontraron datos de sesiones VMP en la base de datos.")
            st.info("El registro de nuevas sesiones VMP sigue disponible en la pestaña SALTO CMJ.")
            df_batch = pd.DataFrame()
        else:
            with st.spinner("Calculando estados..."):
                df_batch = pipeline_batch(df_raw, simulador)
            
        if not df_batch.empty:
            # Ordenar datos: Primero críticos/fatiga (menor score) hasta óptimos (mayor score)
            df_batch = df_batch.sort_values("indice_fatiga", ascending=True)

            # Gráfico semáforo
            df_semaforo = df_batch.rename(columns={"atleta": "nombre", "indice_fatiga": "score"})
            df_semaforo["fecha"] = df_semaforo["ultima_fecha"]
            fig_global = fig_semaforo_barras(df_semaforo)
            st.plotly_chart(fig_global, use_container_width=True)
            
            # Tabla detallada (estructura de tabla con las columnas solicitadas)
            st.subheader("Detalle por Atleta")
            tabla_data = []
            for _, row in df_batch.iterrows():
                tabla_data.append({
                    "Atleta": row['atleta'],
                    "Estado": row['estado'],
                    "Fecha": row['ultima_fecha'],
                    "SWC": row.get('swc_personal', 'N/A'),
                    "DQI": f"{row.get('dqi', 'N/A')} ({row.get('calidad_dato', 'N/A')})",
                    "Feedback": row.get('accion', 'Sin feedback')
                })
            st.table(pd.DataFrame(tabla_data))
        else:
            st.info("No hay atletas con datos suficientes (mínimo 4 sesiones).")

    # ── TAB 2: ANÁLISIS INDIVIDUAL ────────────────────────────────────────────────
    with tab2:
        st.header("Seguimiento Individual")
        
        atletas = cargar_atletas()
        if not atletas:
            st.warning("No hay atletas activos registrados.")
        else:
            atleta_sel = st.selectbox("Seleccione un atleta", [""] + atletas)
            
            if atleta_sel:
                res = pipeline_diagnostico(atleta_sel, df_raw, simulador)
                
                if res and res["estado"] == "INSUFICIENTE":
                    st.warning("Datos insuficientes.")
                elif res:
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
                    
                    # Gráficos (Rediseñados - Apilados)
                    df_atleta_history = get_vmp_history(df_raw, atleta_sel)
                    df_hist = calcular_historial_fatiga(df_raw, atleta_sel, simulador)
                    df_wellness = get_wellness_history(atleta_sel)
                    
                    # Pasar los umbrales de seguridad desde 'res'
                    st.plotly_chart(fig_vmp_tendencia_redesign(df_atleta_history, atleta_sel, res["mmc28"]), use_container_width=True)
                    st.plotly_chart(fig_fatiga_tendencia(df_hist, "Evolución Fatiga"), use_container_width=True)
                    if not df_wellness.empty:
                        st.plotly_chart(fig_wellness_tendencia(df_wellness, atleta_sel), use_container_width=True)
                    else:
                        st.info("No hay datos de Wellness para graficar.")

                    # Gráfico de Carga de Entrenamiento
                    df_atleta_raw = df_raw[df_raw["nombre"] == atleta_sel].copy()
                    df_atleta_raw["carga_interna"] = df_atleta_raw["carga_subjetiva"] * df_atleta_raw["duracion_min"]
                    st.plotly_chart(fig_carga_entrenamiento(df_atleta_raw, atleta_sel), use_container_width=True)
                    
                    # Tabla fija de variables
                    st.subheader("Variables Biomecánicas")
                    tabla_bio = pd.DataFrame({
                        "Variable": ["VMP Hoy", "ACWR", "Delta %", "Z-Meso", "Beta Aguda", "Beta 28d"],
                        "Valor": [
                            f"{res.get('vmp_hoy') or 0.0:.2f}", 
                            f"{res.get('acwr') or 0.0:.2f}", 
                            f"{res.get('delta_pct') or 0.0:.1f}%", 
                            f"{res.get('z_meso') or 0.0:.2f}", 
                            f"{res.get('beta_aguda') or 0.0:.3f}", 
                            f"{res.get('beta_28') or 0.0:.3f}"
                        ],
                        "Fórmula": ["Medida directa", "Media 7d/28d", "(Hoy-MM28)/MM28", "(Hoy-MM28)/SD28", "Regresión 7d", "Regresión 28d"],
                        "Explicación": ["Velocidad actual", "Carga aguda/crónica", "Cambio relativo", "Posición mesociclo", "Tendencia breve", "Tendencia prolongada"]
                    })
                    st.table(tabla_bio)
                    
                    # Contexto científico fijo
                    st.markdown("### Contexto Científico")
                    st.write(res["contexto_cientifico"])

    # ── TAB 3: SALTO CMJ ──────────────────────────────────────────────────────────
    with tab3:
        st.header("Registrar Salto CMJ")
        mode = st.radio("Modo de Registro de Sesiones", ["Individual", "Grupal"], horizontal=True, key="mode_vmp")
        
        if mode == "Individual":
            with st.form("form_registro_indiv"):
                col1, col2 = st.columns(2)
                with col1:
                    atletas_registro = cargar_atletas()
                    atleta_reg = st.selectbox("Atleta", atletas_registro) if atletas_registro else st.text_input("Atleta")
                    fecha_reg = st.date_input("Fecha", date.today(), key="fecha_vmp_indiv")
                with col2:
                    vmp_reg = st.number_input("VMP Hoy (m/s)", min_value=0.1, max_value=2.5, value=1.0, step=0.01)
                
                if st.form_submit_button("Guardar Sesión"):
                    if not atleta_reg or not str(atleta_reg).strip():
                        st.error("Ingresa el nombre del atleta.")
                    else:
                        ok, msg = insertar_sesion(str(atleta_reg).strip(), fecha_reg, vmp_reg, None, "")
                        if ok:
                            st.success(msg)
                            st.cache_data.clear()
                        else:
                            st.error(msg)
        else:
            # Grupal
            st.subheader("Registro Grupal")
            fecha_grupal = st.date_input("Fecha Grupal", date.today(), key="fecha_vmp_grupal")
            atletas_list = cargar_atletas()
            
            # Crear editor de datos para carga grupal
            df_grupal = pd.DataFrame({
                "Atleta": atletas_list,
                "VMP Hoy": [1.0] * len(atletas_list)
            })
            
            edited_df = st.data_editor(df_grupal, use_container_width=True)
            
            if st.button("Guardar Sesiones Grupales"):
                sesiones = []
                for _, row in edited_df.iterrows():
                    if row["VMP Hoy"] > 0:
                        sesiones.append({
                            "nombre": row["Atleta"],
                            "fecha": str(fecha_grupal),
                            "vmp_hoy": float(row["VMP Hoy"]),
                            "vmp_ref": None,
                            "notas": ""
                        })
                
                if sesiones:
                    from data.db import insertar_sesiones_batch
                    ok, msg = insertar_sesiones_batch(sesiones)
                    if ok:
                        st.success(msg)
                        st.cache_data.clear()
                    else:
                        st.error(msg)
                else:
                    st.warning("No hay sesiones con VMP válido para guardar.")
    
    # ── TAB 4: WELLNESS ──────────────────────────────────────────────────────────
    with tab4:
        render_wellness_registro()

    # ── TAB 5: CARGA DE ENTRENAMIENTO ─────────────────────────────────────────
    with tab_carga:
        st.header("🏋️ Carga de Entrenamiento Subjetiva")
        render_rpe_legend()
        st.markdown("---")

        mode_carga = st.radio("Modo de Registro", ["Individual", "Grupal"], horizontal=True, key="mode_carga")
        atletas_carga = cargar_atletas()

        if not atletas_carga:
            st.warning("No hay atletas activos registrados.")
        else:
            if mode_carga == "Individual":
                with st.form("form_carga_indiv"):
                    col1, col2 = st.columns(2)
                    with col1:
                        atleta_c = st.selectbox("Atleta", atletas_carga)
                        fecha_c = st.date_input("Fecha", date.today(), key="fecha_carga_indiv")
                    with col2:
                        rpe_c = st.slider("RPE (1-10)", 1, 10, 5)
                        duracion_c = st.number_input("Duración (min)", min_value=1, value=60, step=5)
                    
                    st.caption("1-2: Muy liviano | 3-4: Liviano | 5-6: Moderado | 7-8: Duro | 9-10: Máximo")
                    
                    if st.form_submit_button("💾 Guardar Carga"):
                        ok, msg = insertar_carga_sesion(atleta_c, fecha_c, int(rpe_c), int(duracion_c))
                        if ok:
                            st.success(msg)
                            st.cache_data.clear()
                        else:
                            st.error(msg)
            else:
                st.subheader("Registro Grupal")
                fecha_g = st.date_input("Fecha Grupal", date.today(), key="fecha_carga_grupal")
                
                # Cargar datos existentes para esa fecha desde df_raw
                # Nota: df_raw ya tiene la columna 'fecha' convertida a date
                df_fecha = df_raw[df_raw["fecha"] == fecha_g]
                
                # Crear DataFrame base con todos los atletas
                df_carga_grupal = pd.DataFrame({
                    "Atleta": atletas_carga,
                    "RPE (1-10)": [5] * len(atletas_carga),
                    "Duración (min)": [60] * len(atletas_carga)
                })
                
                # Mezclar con datos existentes si los hay
                if not df_fecha.empty:
                    # Renombrar columnas para que coincidan con el editor
                    df_existente = df_fecha[["nombre", "carga_subjetiva", "duracion_min"]].copy()
                    df_existente.columns = ["Atleta", "RPE (1-10)", "Duración (min)"]
                    
                    # Usar merge para actualizar los valores por defecto
                    df_carga_grupal = pd.merge(
                        df_carga_grupal[["Atleta"]], 
                        df_existente, 
                        on="Atleta", 
                        how="left"
                    )
                    # Llenar nulos con valores por defecto
                    df_carga_grupal["RPE (1-10)"] = df_carga_grupal["RPE (1-10)"].fillna(5).astype(int)
                    df_carga_grupal["Duración (min)"] = df_carga_grupal["Duración (min)"].fillna(60).astype(int)
                
                edited_carga_df = st.data_editor(df_carga_grupal, use_container_width=True, key="editor_carga_grupal")
                
                if st.button("💾 Guardar Cargas Grupales"):
                    cargas = []
                    for _, row in edited_carga_df.iterrows():
                        if row["RPE (1-10)"] > 0:
                            cargas.append({
                                "nombre": row["Atleta"],
                                "fecha": str(fecha_g),
                                "carga_subjetiva": int(row["RPE (1-10)"]),
                                "duracion_min": int(row["Duración (min)"])
                            })
                    
                    if cargas:
                        ok, msg = insertar_carga_sesion_batch(cargas)
                        if ok:
                            st.success(msg)
                            st.cache_data.clear()
                        else:
                            st.error(msg)
                    else:
                        st.warning("No hay datos de carga válidos para guardar.")

    # ── TAB 6: LESIONES ───────────────────────────────────────────────────────
    with tab_lesiones:
        df_lesiones = cargar_lesiones_activas()
        atletas = cargar_atletas()
        render_tab_lesiones(atletas, df_lesiones)

elif rol == "deportista":
    # ── VISTA DEPORTISTA: SOLO SUS DATOS ─────────────────────────────────────
    atleta_nombre = get_atleta_vmp()
    
    # FAIL-SAFE (P0): Si el perfil no tiene mapeo o el atleta no existe en los datos
    if not atleta_nombre:
        st.error("⚠️ Error de configuración: tu perfil no está vinculado a un atleta. Contacta al staff.")
        st.stop()
    
    # Validar existencia en el set de datos crudos
    if atleta_nombre not in df_raw["nombre"].unique():
        st.error(f"⚠️ Error de datos: No se encuentran registros para '{atleta_nombre}'. Contacta al staff.")
        st.stop()

    st.header(f"Hola, {atleta_nombre}")
    
    # El filtrado se hace estrictamente por nombre_atleta_vmp mapeado
    res = pipeline_diagnostico(atleta_nombre, df_raw, simulador)
    
    if res["estado"] != "INSUFICIENTE":
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
            df_atleta_history = get_vmp_history(df_raw, atleta_nombre)
            fig_vmp = fig_vmp_tendencia(df_atleta_history, atleta_nombre, res["delta_pct"])
            st.plotly_chart(fig_vmp, use_container_width=True)
        with col_g2:
            df_hist = calcular_historial_fatiga(df_raw, atleta_nombre, simulador)
            fig_hist = fig_semaforo_historico(df_hist, "Mi Historial de Fatiga")
            st.plotly_chart(fig_hist, use_container_width=True)
    else:
        st.warning(f"Todavía no tenemos datos suficientes para generar tu diagnóstico (mínimo 4 sesiones). Sesiones actuales: {res['n_sesiones']}/4")
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
