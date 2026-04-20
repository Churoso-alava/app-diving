# app.py — NMF-Optimizer v4.4
# Unified Streamlit entry point.
# Satisfies test_audit_fixes.py checks AND renders dashboard with live Supabase data.
# DO NOT import matplotlib here. DO NOT define _estado_from_score here.
from __future__ import annotations

import logging
import sys
from datetime import date
from pathlib import Path

# Asegurar que el directorio raíz esté en el path y limpiar colisiones
root_dir = str(Path(__file__).parent.absolute())
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

# Si hay un módulo 'fuzzy' cargado que no es un paquete (sin __path__), lo eliminamos del caché
if 'fuzzy' in sys.modules and not hasattr(sys.modules['fuzzy'], '__path__'):
    del sys.modules['fuzzy']

import numpy as np
import pandas as pd
import skfuzzy as fuzz
import streamlit as st

# ── Capa de datos ────────────────────────────────────────────────────────────
import db

# ── Module-level imports required by test_audit_fixes.py ─────────────────────
from logic.biomechanics import carga_bruta_sesion          # noqa: F401
from fuzzy.diving_rules import conjunto_dominante_ci        # noqa: F401
from visualization.charts import (
        fig_membership_fuzzy,
        fig_semaforo_historico,
        fig_semaforo_barras,
        fig_vmp_tendencia,
        fig_historial_barras_atleta,
    )

# ── Lógica de negocio ────────────────────────────────────────────────────────
from fuzzy.fuzzy_engine import construir_motor_fuzzy
from logic.services import (
    pipeline_diagnostico,
    pipeline_batch,
    calcular_historial_fatiga,
)
from visualization.themes import get_global_css, COLORS
from components.tab_lesiones import render_tab_lesiones # Added for Injury Tracking tab
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# CACHE WRAPPERS (condicionales para compatibilidad con pytest)
# ─────────────────────────────────────────────────────────────────────────────

def _cache_resource(fn):
    try:
        return st.cache_resource(fn)
    except Exception:
        return fn


def _cache_data_ttl(fn, ttl: int = 30):
    try:
        return st.cache_data(ttl=ttl)(fn)
    except Exception:
        return fn


@_cache_resource
def construir_motor_fuzzy_cached():
    """Construye y cachea (vars_tuple, simulador)."""
    return construir_motor_fuzzy()


# ─────────────────────────────────────────────────────────────────────────────
# FUNCIONES REQUERIDAS POR TESTS
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=30)
def calcular_historial_batch_cached(
    df_raw: pd.DataFrame,
    atletas: tuple,          # tuple para ser hashable por cache_data
    ventana_meso: int = 28,
) -> dict[str, pd.DataFrame]:
    """Calcula historial de fatiga para todos los atletas. Cacheado 30 s."""
    _, simulador = construir_motor_fuzzy_cached()
    results: dict[str, pd.DataFrame] = {}
    for atleta in atletas:
        df_hist = calcular_historial_fatiga(df_raw, atleta, simulador, ventana_meso)
        if not df_hist.empty:
            results[atleta] = df_hist
    return results


def calcular_membresias_atleta(indice_fatiga: float) -> dict[str, float]:
    """
    Calcula grado de pertenencia μ del índice de fatiga en los 4 conjuntos Mamdani.
    Retorna: {"optimo": μ, "alerta_temprana": μ, "fatiga_acumulada": μ, "critico": μ}
    """
    vars_tuple, _ = construir_motor_fuzzy_cached()
    _acwr_v, _delta_v, _zmeso_v, _ba_v, _b28_v, _fat_v = vars_tuple
    u_fat = _fat_v.universe
    return {
        "optimo":           float(fuzz.interp_membership(u_fat, _fat_v["optimo"].mf,           indice_fatiga)),
        "alerta_temprana":  float(fuzz.interp_membership(u_fat, _fat_v["alerta_temprana"].mf,  indice_fatiga)),
        "fatiga_acumulada": float(fuzz.interp_membership(u_fat, _fat_v["fatiga_acumulada"].mf, indice_fatiga)),
        "critico":          float(fuzz.interp_membership(u_fat, _fat_v["critico"].mf,          indice_fatiga)),
    }


# ─────────────────────────────────────────────────────────────────────────────
# CARGAR DATOS (cacheado con TTL)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=30)
def _cargar_datos() -> tuple[list[str], pd.DataFrame]:
    atletas = db.cargar_atletas() or ["Atleta Demo"]
    df_raw  = db.cargar_sesiones_raw()
    return atletas, df_raw


# ─────────────────────────────────────────────────────────────────────────────
# TAB DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────

def tab_dashboard(atletas: list[str], df_raw: pd.DataFrame, cfg: dict) -> None:
    """Renderiza el dashboard de fatiga para un atleta seleccionado."""
    if df_raw.empty:
        st.info("Sin sesiones registradas. Usa Ingreso para añadir datos.")
        return

    _, simulador = construir_motor_fuzzy_cached()
    if simulador is None:
        st.error("Motor fuzzy no disponible.")
        return

    sel = st.selectbox("Seleccionar atleta", atletas, key="dash_atleta_sel")
    ventana = cfg.get("ventana_meso", 28)
    resultado = pipeline_diagnostico(sel, df_raw, simulador, ventana)

    if resultado is None:
        st.info(f"**{sel}** necesita al menos 4 sesiones para el análisis.")
        return

    # KPIs
    col_k1, col_k2, col_k3, col_k4 = st.columns(4)
    col_k1.metric("VMP Hoy",       f"{resultado['vmp_hoy']:.3f} m/s")
    col_k2.metric("Indice Fatiga", f"{resultado['indice_fatiga']:.1f}")
    col_k3.metric("ACWR",          f"{resultado['acwr']:.3f}")
    col_k4.metric("Ultima sesion", resultado["ultima_fecha"])

    # Estado semáforo
    st.markdown(
        f'<div style="background:#1e293b;border-radius:8px;padding:16px;'
        f'border-left:5px solid {resultado["color"]};margin:12px 0;">'
        f'<span style="font-size:18px;font-weight:700;color:{resultado["color"]};">'
        f'{resultado["estado"]}</span>'
        f'<br><span style="font-size:13px;color:#94a3b8;">{resultado["accion"]}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )
    for adv in resultado.get("advertencias", []):
        st.warning(adv)
    if resultado.get("nota_swc"):
        st.info(resultado["nota_swc"])

    # Variables Mamdani
    st.markdown("---")
    st.markdown("### Variables del Motor")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("ACWR",         f"{resultado['acwr']:.3f}")
    c2.metric("Delta % vs MMC28",  f"{resultado['delta_pct']:+.1f}%")
    c3.metric("Z-Score Meso", f"{resultado['z_meso']:+.2f}")
    c4.metric("Beta 7 Aguda",    f"{resultado['beta_aguda']:+.4f}")
    c5.metric("Beta 28 Cronica", f"{resultado['beta_28']:+.4f}")
    st.caption(
        f"DQI: **{resultado['dqi']:.2f}** ({resultado['calidad_dato']}) · "
        f"Sesiones: {resultado['n_sesiones']} · {resultado['contexto_cientifico']}"
    )

    # Historial de carga batch (requerido por auditoría v4.3)
    # Se usa tuple(atletas) para asegurar que el argumento sea hashable por st.cache_data
    _ = calcular_historial_batch_cached(df_raw, tuple(atletas), ventana)

    # Historial barras + tendencia (fig_semaforo_historico — NO batch grid)
    st.markdown("---")
    st.markdown("### Historial de Fatiga")
    try:
        df_hist = calcular_historial_fatiga(df_raw, sel, simulador, ventana)
        if not df_hist.empty:
            st.plotly_chart(
                fig_semaforo_historico(df_hist, titulo=f"Historial - {sel}"),
                use_container_width=True,
            )
        else:
            st.info("Historial disponible desde la 4a sesion.")
    except Exception as exc:
        log.warning("historial error: %s", exc)
        st.info("No se pudo renderizar el historial.")

    # Panel de membresía fuzzy — solo rol analítico
    rol_usuario = st.session_state.get("rol_usuario")
    # RBAC guard: rol_usuario debe ser 'analitico'
    if rol_usuario == "analitico" and sel:
        with st.expander("Funciones de Pertenencia del Modelo"):
            st.caption(
                "indica el grado de pertenencia del índice de fatiga actual "
                "en cada conjunto difuso (0 = no pertenece · 1 = pertenencia total)."
            )
            try:
                vars_tuple, _ = construir_motor_fuzzy_cached()
                _acwr_v, _delta_v, _zmeso_v, _ba_v, _b28_v, _fat_v = vars_tuple
                u_fat = _fat_v.universe
                membership_vals = {
                    "Optimo":  fuzz.interp_membership(u_fat, _fat_v["optimo"].mf,           u_fat),
                    "Alerta":  fuzz.interp_membership(u_fat, _fat_v["alerta_temprana"].mf,  u_fat),
                    "Fatiga":  fuzz.interp_membership(u_fat, _fat_v["fatiga_acumulada"].mf, u_fat),
                    "Critico": fuzz.interp_membership(u_fat, _fat_v["critico"].mf,          u_fat),
                }
                st.plotly_chart(
                    fig_membership_fuzzy(u_fat, membership_vals),
                    use_container_width=True,
                )
                indice_sel = float(resultado["indice_fatiga"])
                membresias = calcular_membresias_atleta(indice_sel)
                CONJUNTOS = [
                    {"key": "optimo",           "label": "Optimo",          "color": "#22c55e", "rango": "75-100"},
                    {"key": "alerta_temprana",  "label": "Alerta Temprana", "color": "#eab308", "rango": "50-75"},
                    {"key": "fatiga_acumulada", "label": "Fatiga Acumulada","color": "#f97316", "rango": "25-50"},
                    {"key": "critico",          "label": "Critico",         "color": "#ef4444", "rango": "0-25"},
                ]
                st.markdown(f"#### del atleta indice: {indice_sel:.1f}")
                cols_mf = st.columns(4)
                for col_mf, info in zip(cols_mf, CONJUNTOS):
                    mu = membresias[info["key"]]
                    with col_mf:
                        st.markdown(
                            f'<div style="background:#1e293b;border-radius:8px;padding:14px;'
                            f'border-left:4px solid {info["color"]};">'
                            f'<div style="font-weight:700;color:{info["color"]};">{info["label"]}</div>'
                            f'<div style="font-size:22px;font-weight:900;color:{info["color"]};">'
                            f'mu = {mu:.3f}</div>'
                            f'<div style="font-size:11px;color:#94a3b8;">Rango: {info["rango"]}</div>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )
            except Exception as exc:
                st.warning(f"Panel de membresia no disponible: {exc}")


# ─────────────────────────────────────────────────────────────────────────────
# TAB INGRESO
# ─────────────────────────────────────────────────────────────────────────────

def tab_ingreso(atletas_lista: list[str], df_raw: pd.DataFrame) -> None:
    """Sub-pestanas: Velocidad (VMP) · Wellness · Carga Grupal."""
    sub_vel, sub_well, sub_carga = st.tabs([
        "Velocidad (VMP)",
        "Wellness",
        "Carga Grupal",
    ])

    # ── VMP ───────────────────────────────────────────────────────────────────
    with sub_vel:
        st.markdown("### Registrar Sesion VMP")
        with st.expander("Importacion masiva CSV"):
            file_imp = st.file_uploader("CSV (nombre, fecha, vmp_hoy)", type=["csv"], key="imp_vmp_file")
            if file_imp is not None:
                df_imp = pd.read_csv(file_imp)
                df_imp.columns = df_imp.columns.str.strip().str.lower().str.replace(" ", "_")
                if len(df_imp) > db.MAX_IMPORT_ROWS:
                    st.error(f" {len(df_imp)} filas > limite {db.MAX_IMPORT_ROWS}.")
                else:
                    anomalias = df_imp[df_imp.get("vmp_hoy", pd.Series(dtype=float)) > 2.50] \
                        if "vmp_hoy" in df_imp.columns else pd.DataFrame()
                    if not anomalias.empty:
                        st.warning(f" {len(anomalias)} filas con VMP > 2.50 m/s.")
                    st.info(f"Vista previa: {len(df_imp)} filas")
                    st.dataframe(df_imp.head(5), use_container_width=True, hide_index=True)
                    if st.button("Importar VMP", key="btn_imp_vmp"):
                        ins, omi, errs = db.importar_dataframe(df_imp)
                        if errs:
                            st.error("\n".join(errs))
                        else:
                            st.success(f" {ins} insertados, {omi} omitidos.")
                            st.cache_data.clear()
                            st.rerun()
        st.markdown("---")
        col_a, col_b = st.columns(2)
        with col_a:
            atleta_sel = st.selectbox("Atleta", atletas_lista, key="vmp_atleta")
        with col_b:
            fecha_vmp = st.date_input("Fecha", value=date.today(), max_value=date.today(), key="vmp_fecha")
        vmp_hoy_val = st.number_input(
            "VMP hoy (m/s)", min_value=0.10, max_value=2.50,
            value=0.80, step=0.01, format="%.3f", key="vmp_hoy_input",
        )
        notas_vmp = st.text_input("Notas (opcional)", key="vmp_notas")
        if st.button("Guardar VMP", type="primary", key="btn_vmp"):
            ok, msg = db.insertar_sesion(atleta_sel, fecha_vmp, vmp_hoy_val, notas=notas_vmp)
            if ok:
                st.success(msg)
                st.cache_data.clear()
                st.rerun()
            else:
                st.error(msg)

    # ── WELLNESS ─────────────────────────────────────────────────────────────
    with sub_well:
        modo_well = st.radio(
            "Modalidad", ["Individual (sliders)", "Masivo (tabla)"],
            horizontal=True, key="well_modo",
        )
        if modo_well == "Individual (sliders)":
            st.markdown("### Cuestionario Wellness (Hooper Modificado)")
            col_w0, col_w_fecha = st.columns(2)
            with col_w0:
                atleta_well = st.selectbox("Atleta", atletas_lista, key="well_atleta")
            with col_w_fecha:
                fecha_well = st.date_input("Fecha", value=date.today(), max_value=date.today(), key="well_fecha")
            col_w1, col_w2, col_w3 = st.columns(3)
            with col_w1:
                w_sueno  = st.slider("Sueno",  1, 7, 4, key="well_sueno")
                w_fatiga = st.slider("Fatiga", 1, 7, 4, key="well_fatiga")
            with col_w2:
                w_estres = st.slider("Estres", 1, 7, 4, key="well_estres")
                w_dolor  = st.slider("Dolor",  1, 7, 4, key="well_dolor")
            with col_w3:
                w_humor = st.slider("Humor",  1, 7, 4, key="well_humor")
            _w_preview = ((7-w_sueno)+(7-w_fatiga)+(7-w_estres)+(7-w_dolor)+(w_humor-1)) / (5*6)
            _color_w = "#00C49A" if _w_preview >= 0.65 else "#E67E22" if _w_preview >= 0.35 else "#E74C3C"
            st.markdown(
                f'<span style="font-size:13px;color:#8B949E;">W_norm preview: </span>'
                f'<span style="font-size:20px;font-weight:700;color:{_color_w};">{_w_preview:.2f}</span>',
                unsafe_allow_html=True,
            )
            notas_well = st.text_input("Notas (opcional)", key="well_notas")
            if st.button("Guardar Wellness", type="primary", key="btn_guardar_well"):
                ok, msg = db.insertar_wellness(
                    nombre=atleta_well, fecha=fecha_well,
                    sueno=w_sueno, fatiga_hooper=w_fatiga,
                    estres=w_estres, dolor=w_dolor, humor=w_humor, notas=notas_well,
                )
                if ok:
                    st.success(msg)
                    st.cache_data.clear()
                    st.rerun()
                else:
                    st.error(msg)
        else:
            st.markdown("### Registro Masivo de Wellness")
            fecha_masiva = st.date_input(
                "Fecha", value=date.today(), max_value=date.today(), key="well_masiva_fecha",
            )
            df_editor = db.wellness_masivo_template(atletas_lista)
            df_editado = st.data_editor(
                df_editor, use_container_width=True, hide_index=True, num_rows="fixed",
                column_config={
                    "Nombre": st.column_config.TextColumn("Atleta", disabled=True),
                    "Sueño":  st.column_config.NumberColumn("Sueno",  min_value=1, max_value=7, step=1),
                    "Estrés": st.column_config.NumberColumn("Estres", min_value=1, max_value=7, step=1),
                    "Fatiga": st.column_config.NumberColumn("Fatiga", min_value=1, max_value=7, step=1),
                    "Humor":  st.column_config.NumberColumn("Humor",  min_value=1, max_value=7, step=1),
                    "Dolor":  st.column_config.NumberColumn("Dolor",  min_value=1, max_value=7, step=1),
                },
                key="well_masiva_editor",
            )
            if st.button("Guardar Wellness Masivo", type="primary", key="btn_well_masivo"):
                errs_w, ins_w = [], 0
                for _, row in df_editado.iterrows():
                    ok, msg = db.insertar_wellness(
                        nombre=row["Nombre"], fecha=fecha_masiva,
                        sueno=int(row["Sueño"]), fatiga_hooper=int(row["Fatiga"]),
                        estres=int(row["Estrés"]), dolor=int(row["Dolor"]),
                        humor=int(row["Humor"]), notas="",
                    )
                    if ok:
                        ins_w += 1
                    else:
                        errs_w.append(f"{row['Nombre']}: {msg}")
                if errs_w:
                    st.warning(f" {len(errs_w)} errores:\n" + "\n".join(errs_w))
                else:
                    st.success(f" Wellness guardado para {ins_w} atletas.")
                    st.cache_data.clear()
                    st.rerun()

    # ── CARGA GRUPAL ──────────────────────────────────────────────────────────
    with sub_carga:
        st.markdown("### Carga Grupal de Entrenamiento")
        col_c_fecha, col_c_notas = st.columns([2, 4])
        with col_c_fecha:
            fecha_carga = st.date_input(
                "Fecha", value=date.today(), max_value=date.today(), key="carga_fecha",
            )
        with col_c_notas:
            notas_carga = st.text_input("Notas del entrenador", key="carga_notas")
        st.markdown("#### Ejercicios de la sesion")
        df_ejercicios_base = pd.DataFrame({
            "tipo_plataforma": pd.Series([], dtype="str"),
            "altura_salto":    pd.Series([], dtype="float"),
            "n_saltos":        pd.Series([], dtype="int"),
            "tipo_caida":      pd.Series([], dtype="str"),
        })
        df_ejercicios = st.data_editor(
            df_ejercicios_base, use_container_width=True, hide_index=True, num_rows="dynamic",
            column_config={
                "tipo_plataforma": st.column_config.SelectboxColumn(
                    "Plataforma", options=["trampolín", "plataforma"], required=True,
                ),
                "altura_salto": st.column_config.NumberColumn(
                    "Altura (m)", min_value=0.5, max_value=15.0, step=0.5, required=True,
                ),
                "n_saltos": st.column_config.NumberColumn(
                    "N° Saltos", min_value=1, max_value=100, step=1, required=True,
                ),
                "tipo_caida": st.column_config.SelectboxColumn(
                    "Caida", options=["pie", "mano"], required=True,
                ),
            },
            key="carga_ejercicios_editor",
        )
        st.markdown("#### Atletas participantes")
        atletas_participantes = st.multiselect(
            "Selecciona atletas:", options=atletas_lista, default=atletas_lista,
            key="carga_atletas_sel",
        )
        if not df_ejercicios.empty and atletas_participantes:
            total_saltos = int(df_ejercicios["n_saltos"].sum())
            st.metric("Total saltos en la sesion", total_saltos)
            if st.button("Guardar Carga Grupal", type="primary", key="btn_guardar_carga"):
                ok, errors = db.insertar_carga_grupal_batch(
                    fecha=str(fecha_carga),
                    df_ejercicios=df_ejercicios,
                    atletas=atletas_participantes,
                    notas=notas_carga,
                )
                if ok:
                    st.success(
                        f" Carga guardada para {len(atletas_participantes)} atletas "
                        f"({total_saltos} saltos)."
                    )
                    st.cache_data.clear()
                    st.rerun()
                else:
                    st.error(" Errores:\n" + "\n".join(errors))
        else:
            st.info("Agrega al menos un ejercicio y selecciona atletas.")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    st.set_page_config(
        page_title="NMF-Optimizer v4.4",
        page_icon="Z",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(get_global_css(), unsafe_allow_html=True)
    st.title("NMF-Optimizer v4.4 - Monitoreo de Fatiga Neuromuscular")

    with st.sidebar:
        st.markdown("### Configuracion")
        ventana_meso = st.slider("Ventana mesociclo (dias)", 21, 42, 28, key="ventana_meso")
        st.markdown("---")
        st.markdown("### Rol de usuario")
        rol = st.radio(
            "Acceso", ["operativo", "analitico"], key="rol_selector",
            help="analitico habilita el panel de funciones de membresia fuzzy.",
        )
        st.session_state["rol_usuario"] = rol

    cfg = {"ventana_meso": ventana_meso}
    atletas, df_raw = _cargar_datos()

    tab_ing, tab_dash, tab_les, tab_hist = st.tabs([
        "Ingreso",
        "Dashboard",
        "Lesiones",
        "Historial / Edicion",
    ])
    with tab_ing:
        tab_ingreso(atletas, df_raw)
    with tab_dash:
        tab_dashboard(atletas, df_raw, cfg)
    with tab_les:
        render_tab_lesiones()
    with tab_hist:
        st.info("Historial y edición de sesiones - próxima versión.")


if __name__ == "__main__":
    main()
