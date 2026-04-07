"""
app.py — Interfaz Streamlit (solo UI)
Lógica de negocio → services.py | Motor difuso → fuzzy.py | Base de datos → db.py
Sistema de Diseño: Obsidian — High-Contrast Dark
"""
import logging
import warnings
from datetime import date

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import skfuzzy as fuzz
import streamlit as st

import db
import fuzzy as fz
from services import SessionInput, calcular_metricas

warnings.filterwarnings("ignore")

# =============================================================================
#  LOGGING
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# =============================================================================
#  CONFIGURACIÓN DE PÁGINA
# =============================================================================

st.set_page_config(
    page_title="Elite Performance · Obsidian Core",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# =============================================================================
#  SISTEMA DE DISEÑO: OBSIDIAN — HIGH-CONTRAST DARK
# =============================================================================

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Geist:wght@100..900&display=swap');

    :root {
        --bg-main: #09090b;
        --bg-card: #0c0c0f;
        --border-zinc: #27272a;
        --text-primary: #fafafa;
        --text-secondary: #a1a1aa;
        --accent-violet: #a78bfa;
        --success-emerald: #34d399;
        --error-red: #ef4444;
        --warning-amber: #f59e0b;
        --info-blue: #3b82f6;
    }

    /* Reset global */
    .main {
        background-color: var(--bg-main);
        color: var(--text-primary);
        font-family: 'Geist', sans-serif;
    }

    /* Encabezados */
    h1, h2, h3 {
        color: var(--text-primary) !important;
        font-family: 'Geist', sans-serif !important;
        letter-spacing: -0.02em !important;
        font-weight: 900 !important;
    }

    /* Contenedores y Cards */
    div[data-testid="stVerticalBlock"] > div[style*="border"] {
        background-color: var(--bg-card) !important;
        border: 1px solid var(--border-zinc) !important;
        border-radius: 8px !important;
        padding: 1.5rem !important;
    }

    /* Tabs Personalizadas */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
        background-color: transparent;
        border-bottom: 1px solid var(--border-zinc);
    }

    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background-color: transparent !important;
        border: none !important;
        color: var(--text-secondary) !important;
        font-weight: 500;
    }

    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        color: var(--accent-violet) !important;
        border-bottom: 2px solid var(--accent-violet) !important;
    }

    /* Botones */
    .stButton > button {
        border-radius: 6px !important;
        font-weight: 600 !important;
        transition: all 0.2s ease;
    }

    /* Botón Primario */
    .stButton > button[kind="primary"] {
        background-color: var(--accent-violet) !important;
        color: #000000 !important;
        border: none !important;
    }

    /* Botón Secundario/Normal */
    .stButton > button[kind="secondary"] {
        background-color: transparent !important;
        color: var(--text-primary) !important;
        border: 1px solid var(--border-zinc) !important;
    }

    .stButton > button:hover {
        opacity: 0.9;
        transform: translateY(-1px);
    }

    /* Inputs */
    div[data-baseweb="input"], div[data-baseweb="select"], div[data-baseweb="number-input"] {
        background-color: var(--bg-card) !important;
        border: 1px solid var(--border-zinc) !important;
        border-radius: 6px !important;
    }

    /* Metric UI */
    [data-testid="stMetricValue"] {
        color: var(--text-primary) !important;
        font-weight: 800 !important;
    }

    [data-testid="stMetricLabel"] {
        color: var(--text-secondary) !important;
    }

    /* Hide Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# =============================================================================
#  UTILS: GRÁFICOS (MATPLOTLIB)
# =============================================================================

def apply_obsidian_style(fig, ax):
    """Aplica el estilo Obsidian a figuras de Matplotlib."""
    fig.patch.set_facecolor('#09090b')
    ax.set_facecolor('#09090b')
    for spine in ['bottom', 'top', 'right', 'left']:
        ax.spines[spine].set_color('#27272a')
    ax.tick_params(axis='x', colors='#fafafa')
    ax.tick_params(axis='y', colors='#fafafa')
    ax.yaxis.label.set_color('#fafafa')
    ax.xaxis.label.set_color('#fafafa')
    ax.title.set_color('#fafafa')
    if ax.get_legend():
        for text in ax.get_legend().get_texts():
            text.set_color('#fafafa')

def fig_semaforo(valor_fatiga: float):
    """Gauge horizontal Obsidian para el índice de fatiga."""
    fig, ax = plt.subplots(figsize=(6, 1.2))
    colors = ['#34d399', '#3b82f6', '#f59e0b', '#ef4444']
    boundaries = [0, 25, 50, 75, 100]

    for i in range(len(colors)):
        ax.barh(0, boundaries[i+1]-boundaries[i], left=boundaries[i], 
                color=colors[i], alpha=0.3, height=0.4)

    ax.scatter(valor_fatiga, 0, color='#a78bfa', s=150, zorder=5, edgecolors='#fafafa')
    ax.axvline(valor_fatiga, color='#a78bfa', linestyle='--', linewidth=1.5)

    ax.set_xlim(0, 100)
    ax.set_yticks([])
    # Centramos las 4 marcas en medio de cada zona para que coincidan con las 4 etiquetas
    ax.set_xticks([12.5, 37.5, 62.5, 87.5])
    ax.set_xticklabels(['Óptimo', 'Estable', 'Alerta', 'Crítico'], fontsize=8)
    
    apply_obsidian_style(fig, ax)
    plt.tight_layout()
    return fig

def fig_tendencia(fechas, historial):
    """Gráfico de línea con estilo Obsidian mostrando evolución de VMP."""
    fig, ax = plt.subplots(figsize=(7, 3))
    
    # Manejar listas vacías para prevenir errores en matplotlib
    if not historial or not fechas:
        ax.text(0.5, 0.5, "Sin datos de tendencia suficientes", color='#a1a1aa', 
                ha='center', va='center', transform=ax.transAxes)
    else:
        ax.plot(fechas, historial, marker='o', color='#a78bfa', linewidth=2, markersize=5, label='VMP')
        ax.fill_between(fechas, historial, color='#a78bfa', alpha=0.1)
        ax.grid(True, axis='y', linestyle=':', alpha=0.2, color='#fafafa')
        plt.xticks(rotation=45)
    
    ax.set_ylabel("VMP (m/s)")
    apply_obsidian_style(fig, ax)
    
    return fig

# =============================================================================
#  VISTAS (TABS) ARTICULADAS CON DB Y SERVICES
# =============================================================================

def tab_dashboard(df, simulador, vars_tuple, cfg):
    """Dashboard de monitoreo vinculado a services.py."""
    st.markdown("### Rendimiento en Tiempo Real")
    
    atleta = st.selectbox("Seleccionar Atleta", sorted(df["Nombre"].unique()))
    df_atleta = df[df["Nombre"] == atleta].sort_values("Fecha")
    
    if df_atleta.empty:
        st.warning("No hay datos suficientes para este atleta.")
        return

    # 1. Llamada robusta a calcular_metricas (soluciona el NotImplementedError)
    try:
        # services.py espera: df_completo, nombre_atleta
        metrics = calcular_metricas(df, atleta)
    except TypeError:
        try:
            # Fallback en caso de que espere la config
            metrics = calcular_metricas(df, atleta, cfg)
        except Exception as e:
            st.error(f"Error procesando métricas del atleta: {e}")
            return
            
    # 2. Garantizar la evaluación Fuzzy (si no viene ya incrustada en services)
    if "indice_fatiga" not in metrics and hasattr(fz, 'evaluar_fatiga'):
        try:
            # Algunos backends pasan el simulador
            metrics = fz.evaluar_fatiga(metrics, simulador)
        except TypeError:
            try:
                metrics = fz.evaluar_fatiga(metrics)
            except Exception as e:
                log.warning(f"No se pudo evaluar la fatiga difusa: {e}")
            
    # Extracción segura de variables
    fatiga_val = metrics.get('indice_fatiga', 0.0)
    estado_lbl = metrics.get('estado', 'Sin Evaluación')
    acwr_val = metrics.get('acwr', 0.0)
    delta_val = metrics.get('delta_pct', 0.0)
    
    # UI: Obsidian Metrics
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Índice Fatiga", f"{fatiga_val:.1f}%")
    with c2:
        st.metric("Estado", estado_lbl)
    with c3:
        st.metric("ACWR", f"{acwr_val:.2f}")
    with c4:
        st.metric("Variación VMP", f"{delta_val:.1f}%")

    st.markdown("---")
    
    col_left, col_right = st.columns([1, 1], gap="large")
    
    with col_left:
        st.markdown(f"#### Estado de {atleta}")
        st.pyplot(fig_semaforo(fatiga_val))
        
        with st.expander("Ver Recomendación Técnica", expanded=True):
            accion = metrics.get('accion', 'Mantener rutina actual.')
            st.info(accion)
            
            # Mostrar advertencias si existen
            advertencias = metrics.get('advertencias', [])
            if advertencias:
                for adv in advertencias:
                    st.warning(adv)
            
    with col_right:
        st.markdown("#### Evolución de VMP")
        # Trata de extraer fechas e historial del motor de metricas, sino de la base filtrada
        fechas = metrics.get("fechas", df_atleta["Fecha"].tolist())
        historial = metrics.get("historial", df_atleta["VMP_Hoy"].tolist())
        st.pyplot(fig_tendencia(fechas, historial))

def tab_ingreso(atletas_lista):
    """Formulario de entrada de datos validado por SessionInput y DB."""
    st.markdown("### Registro de Nueva Sesión")
    
    with st.form("form_sesion", clear_on_submit=True):
        c1, c2 = st.columns(2)
        with c1:
            nombre = st.selectbox("Atleta", ["-- Seleccionar --"] + atletas_lista)
            fecha = st.date_input("Fecha de Sesión", value=date.today())
        with c2:
            vmp = st.number_input("Velocidad Media Propulsiva (m/s)", min_value=0.1, max_value=2.5, value=1.0, step=0.01)
            notas = st.text_input("Notas u observaciones")
        
        submitted = st.form_submit_button("Guardar Registro", type="primary")
        
        if submitted:
            if nombre == "-- Seleccionar --":
                st.error("Por favor selecciona un atleta.")
            else:
                try:
                    # Validar con el DataClass de services.py
                    sesion = SessionInput(nombre=nombre, fecha=str(fecha), vmp=float(vmp), notas=notas)
                    
                    # Insertar usando db.py
                    ok, msg = db.insertar_sesion(sesion.nombre, sesion.fecha, sesion.vmp, sesion.notas)
                    if ok:
                        st.success(f"Sesión registrada correctamente para {nombre}.")
                        st.cache_data.clear() # Limpiar caché para actualizar vistas
                    else:
                        st.error(msg)
                except Exception as e:
                    st.error(f"Error de validación: {e}")

def tab_historial(df):
    """Gestión de registros vinculado a Supabase mediante db.py."""
    st.markdown("### Historial de Sesiones")
    
    # Mostrar tabla
    st.dataframe(
        df.sort_values("Fecha", ascending=False),
        use_container_width=True,
        hide_index=True
    )
    
    # Interfaz de eliminación real
    st.markdown("#### Eliminar Registro")
    c1, c2 = st.columns([3, 1])
    with c1:
        # Usamos la columna 'id' que viene de Supabase
        id_borrar = st.selectbox("Seleccione el ID de la sesión a eliminar", df['id'].tolist() if 'id' in df else [])
    with c2:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Eliminar", kind="secondary", use_container_width=True):
            if id_borrar:
                ok, msg = db.eliminar_sesion(id_borrar)
                if ok:
                    st.success(msg)
                    st.cache_data.clear()
                    st.rerun()
                else:
                    st.error(msg)

def tab_importacion():
    """Importar datos externos vía db.importar_dataframe."""
    st.markdown("### Importar Datos (CSV)")
    archivo = st.file_uploader("Subir archivo de entrenamiento", type=["csv"])
    
    if archivo:
        try:
            df_import = pd.read_csv(archivo)
            st.dataframe(df_import.head(3), use_container_width=True)
            
            if st.button("Procesar Importación", type="primary"):
                with st.spinner("Sincronizando con Supabase..."):
                    insertados, omitidos, errores = db.importar_dataframe(df_import)
                    st.success(f"✅ Completado: {insertados} registros insertados | {omitidos} omitidos.")
                    if errores:
                        with st.expander("Ver detalles de registros omitidos"):
                            for err in errores:
                                st.write(err)
                    st.cache_data.clear()
        except Exception as e:
            st.error(f"Error al leer el archivo: {e}")

# =============================================================================
#  APP PRINCIPAL
# =============================================================================

@st.cache_data
def obtener_datos_cached():
    """Conecta directamente con la función real de db.py"""
    return db.cargar_sesiones()

@st.cache_resource
def construir_motor_fuzzy_cached():
    """Conecta con fuzzy.py para inicializar las variables difusas"""
    try:
        # Devuelve las variables según fuzzy.py
        return fz.construir_sistema_fuzzy()
    except Exception as e:
        log.error(f"Error cargando motor fuzzy: {e}")
        return None, None

def main():
    # Header Obsidian
    st.markdown("""
        <div style="text-align: center; padding: 1rem 0 2rem 0;">
            <h1 style="margin-bottom: 0;">ELITE PERFORMANCE</h1>
            <p style="color: #a1a1aa; font-family: 'Geist'; letter-spacing: 0.1em;">
                OBSIDIAN CORE · ENGINE v4.1
            </p>
        </div>
    """, unsafe_allow_html=True)

    # 1. Cargar Datos
    df_raw = obtener_datos_cached()
    atletas_lista = sorted(df_raw["Nombre"].unique().tolist()) if not df_raw.empty else []
    
    # 2. Resumen Superior
    if not df_raw.empty:
        n_at = len(atletas_lista)
        ultima = pd.to_datetime(df_raw["Fecha"]).max().strftime("%d/%m/%Y")
        st.markdown(f"""
            <div style="background: #0c0c0f; border: 1px solid #27272a; border-radius: 8px; padding: 10px; text-align: center; margin-bottom: 20px;">
                <span style="color: #34d399;">●</span> 
                <span style="color: #fafafa; font-weight: 500;">{len(df_raw)} Registros</span> 
                <span style="color: #27272a; margin: 0 10px;">|</span>
                <span style="color: #fafafa; font-weight: 500;">{n_at} Atletas activos</span>
                <span style="color: #27272a; margin: 0 10px;">|</span>
                <span style="color: #a1a1aa;">Sincronizado: {ultima}</span>
            </div>
        """, unsafe_allow_html=True)

    # 3. Inicializar Motor Fuzzy
    motor_fuzzy = construir_motor_fuzzy_cached()
    
    # Manejo seguro si fuzzy.py retorna una tupla (vars, simulador) o solo las variables
    if isinstance(motor_fuzzy, tuple) and len(motor_fuzzy) == 2:
        vars_tuple, simulador = motor_fuzzy
    else:
        vars_tuple = motor_fuzzy
        simulador = None

    # Navegación Obsidian Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Dashboard",
        "➕ Registro",
        "✏️ Historial",
        "📤 Importar",
    ])

    cfg = {} # Fallback config object

    with tab1:
        if not df_raw.empty:
            tab_dashboard(df_raw, simulador, vars_tuple, cfg)
        else:
            st.info("Sin datos. Registra sesiones para comenzar.")

    with tab2:
        tab_ingreso(atletas_lista)

    with tab3:
        if not df_raw.empty:
            tab_historial(df_raw)
        else:
            st.info("No hay sesiones registradas.")

    with tab4:
        tab_importacion()

    # Footer
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("---")
    c_f1, c_f2 = st.columns([3, 1])
    with c_f1:
        st.caption("Modelo Fuzzy Mamdani v4.1 · Obsidian High-Contrast UI")
    with c_f2:
        if st.button("Limpiar Caché / Forzar Refresco", kind="secondary", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

if __name__ == "__main__":
    main()
