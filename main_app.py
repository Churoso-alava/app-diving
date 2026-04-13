"""
app.py — NMF-Optimizer v4.4
Entry point Streamlit. Orquesta capas desacopladas:
  data/db.py            → CRUD puro
  logic/services.py     → métricas + pipeline Mamdani
  fuzzy/fuzzy_engine.py → Motor Mamdani v4.1 (28 reglas)
  components/           → pestañas UI
  visualization/        → gráficos Plotly

RBAC: panel de membresía solo para rol_usuario == 'analitico'
Security: MAX_IMPORT_ROWS desde data.db (única fuente de verdad)
"""
from __future__ import annotations

import logging

import pandas as pd
import streamlit as st

import data.db as db
from components.tab_ingreso   import render_tab_ingreso
from components.tab_dashboard import render_tab_dashboard

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# MOTOR FUZZY — construido una vez por sesión
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource
def _construir_motor():
    """Construye el motor Mamdani v4.1 y lo cachea para toda la sesión."""
    try:
        from fuzzy.fuzzy_engine import construir_motor_fuzzy
        _vars, simulador = construir_motor_fuzzy()
        log.info("Motor fuzzy Mamdani v4.1 construido correctamente.")
        return simulador
    except Exception as exc:
        log.error("Error construyendo motor fuzzy: %s", exc)
        return None


# ─────────────────────────────────────────────────────────────────────────────
# DATOS — cargados y cacheados con TTL
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=30)
def _cargar_datos() -> tuple[list[str], pd.DataFrame]:
    """Carga atletas y sesiones desde Supabase (TTL 30 s)."""
    atletas = db.cargar_atletas() or ["Atleta Demo"]
    df_raw  = db.cargar_sesiones_raw()
    return atletas, df_raw


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    st.set_page_config(
        page_title="NMF-Optimizer v4.4",
        page_icon="⚡",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # ── CSS global ─────────────────────────────────────────────────────────────
    try:
        from visualization.themes import get_global_css
        st.markdown(get_global_css(), unsafe_allow_html=True)
    except ImportError:
        pass

    st.title("⚡ NMF-Optimizer v4.4 — Monitoreo de Fatiga Neuromuscular")

    # ── Sidebar ────────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### ⚙️ Configuración")
        ventana_meso = st.slider("Ventana mesociclo (días)", 21, 42, 28, key="ventana_meso")
        st.markdown("---")
        st.markdown("### 👤 Rol de usuario")
        rol = st.radio(
            "Acceso", ["entrenador", "analitico"], key="rol_selector",
            help="'analítico' habilita el panel de funciones de membresía fuzzy."
        )
        st.session_state["rol_usuario"] = rol

    cfg = {"ventana_meso": ventana_meso}

    # ── Motor fuzzy ────────────────────────────────────────────────────────────
    simulador = _construir_motor()
    if simulador is None:
        st.error("❌ Motor fuzzy no disponible. Verifica la instalación de scikit-fuzzy.")

    # ── Datos ──────────────────────────────────────────────────────────────────
    atletas, df_raw = _cargar_datos()

    # ── Pestañas ───────────────────────────────────────────────────────────────
    tab_ing, tab_dash, tab_hist = st.tabs([
        "➕ Ingreso",
        "📊 Dashboard",
        "✏️ Historial / Edición",
    ])

    with tab_ing:
        render_tab_ingreso(atletas)

    with tab_dash:
        if simulador is not None:
            render_tab_dashboard(atletas, df_raw, simulador, cfg)
        else:
            st.info("Dashboard no disponible sin el motor fuzzy.")

    with tab_hist:
        st.info("Historial y edición de sesiones — próxima versión.")


if __name__ == "__main__":
    main()
