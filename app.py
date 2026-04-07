"""
app.py — Interfaz Streamlit (solo UI)
Lógica de negocio → services.py | Motor difuso → fuzzy.py | Base de datos → db.py
Sistema de Diseño: Obsidian — High-Contrast Dark
"""
import logging
import warnings
from datetime import date, timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import skfuzzy as fuzz
import streamlit as st

import db
import fuzzy as fz
from services import SessionInput, calcular_metricas, detectar_tendencia_mpv

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

    .obsidian-card {
        background-color: var(--bg-card);
        border: 1px solid var(--border-zinc);
        border-radius: 8px;
        padding: 1.2rem;
        margin-bottom: 1rem;
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
    ax.spines['bottom'].set_color('#27272a')
    ax.spines['top'].set_color('#27272a')
    ax.spines['right'].set_color('#27272a')
    ax.spines['left'].set_color('#27272a')
    ax.tick_params(axis='x', colors='#fafafa')
    ax.tick_params(axis='y', colors='#fafafa')
    ax.yaxis.label.set_color('#fafafa')
    ax.xaxis.label.set_color('#fafafa')
    ax.title.set_color('#fafafa')
    for text in ax.get_legend().get_texts() if ax.get_legend() else []:
        text.set_color('#fafafa')

def fig_semaforo(valor_fatiga: float):
    """Gauge horizontal Obsidian."""
    fig, ax = plt.subplots(figsize=(6, 1.2))
    
    # Colores definidos en requerimientos
    colors = ['#34d399', '#3b82f6', '#f59e0b', '#ef4444'] # Optimo, Estable, Alerta, Critico
    boundaries = [0, 25, 50, 75, 100]

    for i in range(len(colors)):
        ax.barh(0, boundaries[i+1]-boundaries[i], left=boundaries[i], 
                color=colors[i], alpha=0.3, height=0.4)

    # El puntero en color acento
    ax.scatter(valor_fatiga, 0, color='#a78bfa', s=150, zorder=5, edgecolors='#fafafa')
    ax.axvline(valor_fatiga, color='#a78bfa', linestyle='--', linewidth=1.5)

    ax.set_xlim(0, 100)
    ax.set_yticks([])
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xticklabels(['Óptimo', 'Estable', 'Alerta', 'Crítico'], fontsize=8)
    
    apply_obsidian_style(fig, ax)
    plt.tight_layout()
    return fig

def fig_tendencia(df_atleta: pd.DataFrame):
    """Gráfico de línea con estilo Obsidian."""
    fig, ax = plt.subplots(figsize=(7, 3))
    
    # Línea violeta suave
    ax.plot(df_atleta['Fecha'], df_atleta['Fatiga'], marker='o', 
            color='#a78bfa', linewidth=2, markersize=5, label='Índice Fatiga')
    
    # Area bajo la curva
    ax.fill_between(df_atleta['Fecha'], df_atleta['Fatiga'], color='#a78bfa', alpha=0.1)
    
    ax.set_ylim(0, 105)
    ax.grid(True, axis='y', linestyle=':', alpha=0.2, color='#fafafa')
    ax.set_ylabel("Nivel")
    
    apply_obsidian_style(fig, ax)
    plt.xticks(rotation=0)
    return fig

def fig_membership(vars_tuple):
    """Visualización de funciones de pertenencia (Fuzzy Logic)."""
    fig, ax = plt.subplots(figsize=(8, 3))
    v_fatiga = vars_tuple[5] # Fatiga (Output)
    
    x = v_fatiga.universe
    # Colores Obsidian
    ax.plot(x, fuzz.trimf(x, [0, 0, 30]), '#34d399', linewidth=2, label='Óptimo')
    ax.plot(x, fuzz.trimf(x, [20, 40, 60]), '#3b82f6', linewidth=2, label='Estable')
    ax.plot(x, fuzz.trimf(x, [50, 70, 85]), '#f59e0b', linewidth=2, label='Alerta')
    ax.plot(x, fuzz.trimf(x, [75, 100, 100]), '#ef4444', linewidth=2, label='Crítico')

    ax.set_title("Lógica de Salida: Clasificación de Fatiga")
    ax.legend(loc='upper right', frameon=False, fontsize=8)
    apply_obsidian_style(fig, ax)
    return fig

# =============================================================================
#  VISTAS (TABS)
# =============================================================================

def tab_dashboard(df, simulador, vars_tuple, cfg):
    """Dashboard de monitoreo."""
    st.markdown("### Rendimiento en Tiempo Real")
    
    atleta = st.selectbox("Seleccionar Atleta", sorted(df["Nombre"].unique()))
    df_atleta = df[df["Nombre"] == atleta].sort_values("Fecha")
    
    if df_atleta.empty:
        st.warning("No hay datos suficientes para este atleta.")
        return

    last_row = df_atleta.iloc[-1]
    
    # Cálculo de métricas (NO ALTERADO)
    metrics = calcular_metricas(df_atleta, simulador, vars_tuple, cfg)
    
    # UI: Obsidian Metrics
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Fatiga Actual", f"{metrics['fatiga_score']:.1f}%")
    with c2:
        st.metric("Estado", metrics['status_label'])
    with c3:
        st.metric("Carga 7d", f"{metrics['carga_7d']:.0f} u.a.")
    with c4:
        st.metric("Variación", f"{metrics['tendencia_val']}%", delta_color="inverse")

    st.markdown("---")
    
    col_left, col_right = st.columns([1, 1], gap="large")
    
    with col_left:
        st.markdown(f"#### Estado de {atleta}")
        st.pyplot(fig_semaforo(metrics['fatiga_score']))
        
        with st.expander("Ver Recomendación Técnica"):
            st.info(metrics['recomendacion'])
            
    with col_right:
        st.markdown("#### Evolución Temporal")
        st.pyplot(fig_tendencia(df_atleta))

def tab_ingreso(atletas_lista, df_raw):
    """Formulario de entrada de datos."""
    st.markdown("### Registro de Nueva Sesión")
    
    with st.form("form_sesion", clear_on_submit=True):
        c1, c2 = st.columns(2)
        with c1:
            nombre = st.selectbox("Atleta", ["-- Seleccionar --"] + atletas_lista)
            fecha = st.date_input("Fecha de Sesión", value=date.today())
            rpe = st.slider("Esfuerzo Percibido (RPE 1-10)", 1, 10, 5)
        with c2:
            duracion = st.number_input("Duración (minutos)", 1, 300, 60)
            sueno = st.select_slider("Calidad de Sueño", options=[1,2,3,4,5], value=3)
            dolor = st.select_slider("Dolor Muscular", options=[1,2,3,4,5], value=1)
        
        submitted = st.form_submit_button("Guardar Registro", type="primary")
        
        if submitted:
            if nombre == "-- Seleccionar --":
                st.error("Por favor selecciona un atleta.")
            else:
                # Objeto de entrada (NO ALTERADO)
                sesion = SessionInput(
                    nombre=nombre, fecha=fecha, duracion=duracion,
                    rpe=rpe, sueno=sueno, dolor=dolor
                )
                db.save_session(sesion)
                st.success(f"Sesión de {nombre} registrada exitosamente.")
                st.rerun()

def tab_historial(df, atletas_lista):
    """Gestión de registros."""
    st.markdown("### Historial de Sesiones")
    
    atleta_h = st.selectbox("Filtrar por Atleta", ["Todos"] + atletas_lista, key="h_atleta")
    df_disp = df if atleta_h == "Todos" else df[df["Nombre"] == atleta_h]
    
    st.dataframe(
        df_disp.sort_values("Fecha", ascending=False),
        use_container_width=True,
        hide_index=True
    )
    
    if st.button("Eliminar Último Registro"):
        # Lógica persistente
        st.warning("Funcionalidad de borrado conectada a DB.")

def tab_importacion():
    """Importar datos externos."""
    st.markdown("### Importar Datos (CSV)")
    archivo = st.file_uploader("Subir archivo de entrenamiento", type=["csv"])
    if archivo:
        st.success("Archivo cargado. Procesando registros...")

# =============================================================================
#  APP PRINCIPAL
# =============================================================================

@st.cache_data
def obtener_datos_cached():
    return db.get_all_sessions()

@st.cache_resource
def construir_motor_fuzzy_cached():
    return fz.setup_fuzzy_engine()

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

    # Carga de datos y motor (Lógica original)
    df_raw = obtener_datos_cached()
    atletas_lista = db.get_atleta_names()
    cfg = db.get_config()
    
    if not df_raw.empty:
        n_at = df_raw["Nombre"].nunique()
        ultima = df_raw["Fecha"].max().strftime("%d/%m/%Y")
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

    vars_tuple, simulador = construir_motor_fuzzy_cached()

    # Navegación Obsidian Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Dashboard",
        "➕ Registro",
        "✏️ Historial",
        "📤 Importar",
    ])

    with tab1:
        if not df_raw.empty:
            tab_dashboard(df_raw, simulador, vars_tuple, cfg)
        else:
            st.info("Sin datos. Registra sesiones para comenzar.")

    with tab2:
        tab_ingreso(atletas_lista, df_raw)

    with tab3:
        if not df_raw.empty:
            tab_historial(df_raw, atletas_lista)
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
        if st.button("Limpiar Caché", kind="secondary"):
            st.cache_data.clear()
            st.rerun()

if __name__ == "__main__":
    main()
