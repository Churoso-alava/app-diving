"""
app.py — Interfaz Streamlit (solo UI)
Lógica de negocio → services.py | Motor difuso → fuzzy.py | Base de datos → db.py
"""
import logging
import warnings
from datetime import date

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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
    page_title="Dashboard Fatiga · Club Tornados",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
  .block-container { padding-top: 1.2rem; }
  .stTabs [data-baseweb="tab-list"] { gap: 8px; }
  .stTabs [data-baseweb="tab"] {
    background: #1e293b; border-radius: 6px 6px 0 0; 
    padding: 10px 20px; color: #cbd5e1; font-weight: 500;
  }
  .stTabs [data-baseweb="tab"][aria-selected="true"] {
    background: #3b82f6; color: white;
  }
</style>
""", unsafe_allow_html=True)

# =============================================================================
#  UTILS: GRÁFICOS (MATPLOTLIB)
# =============================================================================

def fig_semaforo(valor_fatiga: float):
    """Gauge horizontal para el índice de fatiga."""
    fig, ax = plt.subplots(figsize=(6, 1.2))
    fig.patch.set_facecolor('#0e1117')
    ax.set_facecolor('#0e1117')
    
    colors = ['#10b981', '#3b82f6', '#f59e0b', '#ef4444']
    boundaries = [0, 25, 50, 75, 100]

    for i in range(len(colors)):
        ax.barh(0, boundaries[i+1]-boundaries[i], left=boundaries[i], 
                color=colors[i], alpha=0.3, height=0.4)

    ax.scatter(valor_fatiga, 0, color='white', s=150, zorder=5, edgecolors='black')
    ax.axvline(valor_fatiga, color='white', linestyle='--', linewidth=1)

    ax.set_xlim(0, 100)
    ax.set_yticks([])
    ax.set_xticks([12.5, 37.5, 62.5, 87.5])
    ax.set_xticklabels(['Óptimo', 'Estable', 'Alerta', 'Crítico'], color='white', fontsize=8)
    
    for spine in ['top', 'right', 'left', 'bottom']:
        ax.spines[spine].set_color('#333333')
        
    ax.tick_params(axis='x', colors='white')
    plt.tight_layout()
    return fig

def fig_tendencia(fechas, historial):
    """Gráfico de línea mostrando evolución de VMP."""
    fig, ax = plt.subplots(figsize=(7, 3))
    fig.patch.set_facecolor('#0e1117')
    ax.set_facecolor('#0e1117')
    
    if not historial or not fechas:
        ax.text(0.5, 0.5, "Sin datos de tendencia suficientes", color='white', 
                ha='center', va='center', transform=ax.transAxes)
    else:
        ax.plot(fechas, historial, marker='o', color='#3b82f6', linewidth=2, markersize=5)
        ax.fill_between(fechas, historial, color='#3b82f6', alpha=0.1)
        ax.grid(True, axis='y', linestyle=':', alpha=0.2, color='white')
        plt.xticks(rotation=45, color='white')
        plt.yticks(color='white')
        ax.set_ylabel("VMP (m/s)", color='white')
        
    for spine in ['top', 'right', 'left', 'bottom']:
        ax.spines[spine].set_color('#333333')
        
    plt.tight_layout()
    return fig

# =============================================================================
#  VISTAS (TABS)
# =============================================================================

def tab_dashboard(df, simulador):
    st.markdown("### Rendimiento en Tiempo Real")
    
    c_atleta, c_vacio = st.columns([1, 2])
    with c_atleta:
        atletas_lista = sorted(df["Nombre"].unique().tolist())
        atleta = st.selectbox("Seleccionar Atleta", atletas_lista)
        
    df_atleta = df[df["Nombre"] == atleta].sort_values("Fecha")
    if df_atleta.empty:
        st.warning("No hay datos suficientes para este atleta.")
        return

    # Cálculos Matemáticos
    metrics = calcular_metricas(df, atleta)
    
    if "indice_fatiga" not in metrics and hasattr(fz, 'evaluar_fatiga'):
        metrics = fz.evaluar_fatiga(metrics, simulador)

    fatiga_val = metrics.get('indice_fatiga', 0.0)
    estado_lbl = metrics.get('estado', 'Sin Evaluación')
    acwr_val = metrics.get('acwr', 0.0)
    delta_val = metrics.get('delta_pct', 0.0)
    
    # Fila de métricas
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Índice Fatiga (0-100)", f"{fatiga_val}%")
    c2.metric("Estado Actual", estado_lbl)
    c3.metric("ACWR (Carga)", f"{acwr_val:.2f}")
    c4.metric("Variación VMP", f"{delta_val}%")

    st.markdown("---")
    
    # Recomendaciones Técnicas
    st.markdown("#### Recomendación Técnica")
    st.info(f"**Acción Sugerida:** {metrics.get('accion', 'Mantener rutina actual.')}")
    
    for adv in metrics.get('advertencias', []):
        st.warning(adv)

    # Gráficos
    st.markdown("<br>", unsafe_allow_html=True)
    c_graf1, c_graf2 = st.columns(2)
    with c_graf1:
        st.markdown("**Estado del Atleta (Semáforo)**")
        st.pyplot(fig_semaforo(fatiga_val))
    with c_graf2:
        st.markdown("**Evolución VMP (m/s)**")
        fechas = metrics.get("fechas", df_atleta["Fecha"].tolist())
        historial = metrics.get("historial", df_atleta["VMP_Hoy"].tolist())
        st.pyplot(fig_tendencia(fechas, historial))

def tab_ingreso(atletas_lista):
    st.markdown("### ➕ Ingreso de Datos (Sesión)")
    
    with st.form("form_sesion", clear_on_submit=True):
        c1, c2 = st.columns(2)
        with c1:
            nombre = st.selectbox("Atleta", ["-- Seleccionar --"] + atletas_lista)
            fecha = st.date_input("Fecha", value=date.today())
        with c2:
            vmp = st.number_input("VMP_Hoy (m/s)", min_value=0.1, max_value=2.5, value=1.0, step=0.01)
            notas = st.text_input("Notas (Opcional)")
        
        if st.form_submit_button("Guardar Registro", type="primary"):
            if nombre == "-- Seleccionar --":
                st.error("Por favor selecciona un atleta.")
            else:
                try:
                    sesion = SessionInput(nombre=nombre, fecha=str(fecha), vmp=float(vmp), notas=notas)
                    ok, msg = db.insertar_sesion(sesion.nombre, sesion.fecha, sesion.vmp, sesion.notas)
                    if ok:
                        st.success(f"✅ {msg}")
                        st.cache_data.clear()
                    else:
                        st.error(msg)
                except Exception as e:
                    st.error(f"Error de validación: {e}")

def tab_historial(df, atletas_lista):
    st.markdown("### ✏️ Historial y Edición")
    
    c1, c2 = st.columns(2)
    with c1:
        f_atleta = st.selectbox("Filtrar por Atleta", ["Todos"] + atletas_lista)
        
    df_show = df if f_atleta == "Todos" else df[df["Nombre"] == f_atleta]
    st.dataframe(df_show.sort_values("Fecha", ascending=False), use_container_width=True, hide_index=True)
    
    st.markdown("---")
    st.markdown("#### 🗑️ Eliminar Registro")
    
    df_ids = df_show.dropna(subset=["id"]) if "id" in df_show.columns else df_show
    if not df_ids.empty:
        c_del1, c_del2 = st.columns([3, 1])
        with c_del1:
            id_borrar = st.selectbox("Seleccione el ID de la sesión a eliminar", df_ids["id"].astype(str).tolist())
        with c_del2:
            st.markdown("<br>", unsafe_allow_html=True)
            # ERROR CORREGIDO AQUI: Se remueve kind="secondary" y se usa type="secondary"
            if st.button("Eliminar Sesión", type="secondary", use_container_width=True):
                ok, msg = db.eliminar_sesion(id_borrar)
                if ok:
                    st.success(msg)
                    st.cache_data.clear()
                    st.rerun()
                else:
                    st.error(msg)
    else:
        st.info("No hay registros con ID válidos para eliminar.")

def tab_importacion():
    st.markdown("### 📤 Importar Datos Masivos (CSV)")
    archivo = st.file_uploader("Sube un archivo CSV con columnas: Nombre, Fecha, VMP_Hoy, notas", type=["csv"])
    
    if archivo:
        try:
            df_import = pd.read_csv(archivo)
            st.dataframe(df_import.head(3), use_container_width=True)
            
            if st.button("Procesar Importación", type="primary"):
                with st.spinner("Sincronizando con Supabase..."):
                    insertados, omitidos, errores = db.importar_dataframe(df_import)
                    st.success(f"✅ Completado: **{insertados}** registros insertados | **{omitidos}** omitidos.")
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
    return db.cargar_sesiones()

@st.cache_resource
def construir_motor_fuzzy_cached():
    return fz.construir_sistema_fuzzy()

def main():
    # 1. Cargar Datos
    df_raw = obtener_datos_cached()
    atletas_lista = sorted(df_raw["Nombre"].unique().tolist()) if not df_raw.empty else []
    
    st.title("Dashboard Fatiga · Club Tornados")
    
    # 2. Resumen Superior estilo Original
    if not df_raw.empty:
        n_atletas = len(atletas_lista)
        ultima = pd.to_datetime(df_raw["Fecha"]).max().strftime("%d/%m/%Y")
        st.success(
            f"✅ **{len(df_raw)} registros** · **{n_atletas} atletas** · "
            f"Última sesión: **{ultima}**"
        )

    # 3. Inicializar Motor Fuzzy
    motor_fuzzy = construir_motor_fuzzy_cached()
    simulador = motor_fuzzy[1] if isinstance(motor_fuzzy, tuple) and len(motor_fuzzy) == 2 else None

    # Navegación Tabs Original
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Dashboard",
        "➕ Ingreso de Datos",
        "✏️ Historial / Edición",
        "📤 Importar CSV",
    ])

    with tab1:
        if not df_raw.empty:
            tab_dashboard(df_raw, simulador)
        else:
            st.info("Sin datos. Registra sesiones en **➕ Ingreso de Datos**.")

    with tab2:
        tab_ingreso(atletas_lista)

    with tab3:
        if not df_raw.empty:
            tab_historial(df_raw, atletas_lista)
        else:
            st.info("No hay sesiones registradas aún.")

    with tab4:
        tab_importacion()

    st.markdown("---")
    st.caption("Modelo Fuzzy Mamdani v4.1 · 5 variables · 23 reglas · Defuzzificación COA")

if __name__ == "__main__":
    main()
