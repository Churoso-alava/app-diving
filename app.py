"""
app.py — NMF-Optimizer v4.3
Sistema de monitoreo de fatiga neuromuscular.
SQL-First · Clark-Wilson · Glassmorphism UI

RBAC: membresía fuzzy solo if rol_usuario == 'analitico'
Security: MAX_IMPORT_ROWS referenciado desde db (única fuente de verdad)
"""
from __future__ import annotations
import pandas as pd

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
    )
    return df


import logging
from datetime import date

import pandas as pd
import plotly.graph_objects as go

# Importar db antes que streamlit para que los tests puedan importar sin UI
import db

log = logging.getLogger(__name__)

# ── Imports condicionales (Streamlit + Fuzzy) ─────────────────────────────
try:
    import streamlit as st
    import skfuzzy as fuzz
    import skfuzzy.control as ctrl
    import numpy as np
    _STREAMLIT_AVAILABLE = True
except ImportError:
    _STREAMLIT_AVAILABLE = False

try:
    from visualization.charts import (
        fig_semaforo_historico,
        fig_membership_fuzzy,
        fig_historial_barras_atleta,
        _DARK_LAYOUT,
    )
    _CHARTS_AVAILABLE = True
except ImportError:
    _CHARTS_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────────────
# MOTOR FUZZY MAMDANI v4.1
# ─────────────────────────────────────────────────────────────────────────────

def _cache_resource(fn):
    """Wrapper: usa st.cache_resource si Streamlit está disponible."""
    if _STREAMLIT_AVAILABLE:
        return st.cache_resource(fn)  # type: ignore[misc]
    return fn


@_cache_resource
def construir_motor_fuzzy_cached():
    """
    Construye el motor Mamdani de 5 entradas y 23 reglas.
    Cacheado como recurso para toda la sesión.
    Retorna (vars_tuple, simulacion).
    """
    import numpy as np
    import skfuzzy as fuzz
    import skfuzzy.control as ctrl

    u_acwr  = np.linspace(0, 2, 100)
    u_delta = np.linspace(-30, 30, 100)
    u_zmeso = np.linspace(0, 100, 100)
    u_ba    = np.linspace(0, 100, 100)
    u_b28   = np.linspace(0, 100, 100)
    u_fat   = np.linspace(0, 100, 100)

    acwr_v  = ctrl.Antecedent(u_acwr,  "acwr")
    delta_v = ctrl.Antecedent(u_delta, "delta_pct")
    zmeso_v = ctrl.Antecedent(u_zmeso, "z_meso")
    ba_v    = ctrl.Antecedent(u_ba,    "b_aguda")
    b28_v   = ctrl.Antecedent(u_b28,   "b_cronica")
    fat_v   = ctrl.Consequent(u_fat,   "fatiga", defuzzify_method="centroid")

    # Funciones de pertenencia — output
    fat_v["optimo"]           = fuzz.trapmf(u_fat, [60, 75, 100, 100])
    fat_v["alerta_temprana"]  = fuzz.trimf(u_fat,  [40, 57, 75])
    fat_v["fatiga_acumulada"] = fuzz.trimf(u_fat,  [15, 35, 55])
    fat_v["critico"]          = fuzz.trapmf(u_fat, [0,  0, 20, 35])

    # Funciones de pertenencia — entradas (simplificadas para el sistema)
    acwr_v["bajo"]    = fuzz.trapmf(u_acwr, [0, 0, 0.8, 1.0])
    acwr_v["optimo"]  = fuzz.trimf(u_acwr,  [0.8, 1.05, 1.3])
    acwr_v["alto"]    = fuzz.trapmf(u_acwr, [1.1, 1.3, 2, 2])

    delta_v["mejora"]   = fuzz.trapmf(u_delta, [-30, -30, -5, 0])
    delta_v["estable"]  = fuzz.trimf(u_delta,  [-8, 0, 8])
    delta_v["fatiga"]   = fuzz.trapmf(u_delta, [5, 10, 30, 30])

    zmeso_v["bajo"]    = fuzz.trapmf(u_zmeso, [0, 0, 30, 50])
    zmeso_v["medio"]   = fuzz.trimf(u_zmeso,  [30, 55, 80])
    zmeso_v["alto"]    = fuzz.trapmf(u_zmeso, [65, 80, 100, 100])

    ba_v["bajo"]  = fuzz.trapmf(u_ba, [0, 0, 30, 50])
    ba_v["medio"] = fuzz.trimf(u_ba,  [30, 55, 80])
    ba_v["alto"]  = fuzz.trapmf(u_ba, [65, 80, 100, 100])

    b28_v["bajo"]  = fuzz.trapmf(u_b28, [0, 0, 30, 50])
    b28_v["medio"] = fuzz.trimf(u_b28,  [30, 55, 80])
    b28_v["alto"]  = fuzz.trapmf(u_b28, [65, 80, 100, 100])

    # 23 reglas Mamdani
    reglas = [
        ctrl.Rule(acwr_v["optimo"] & delta_v["estable"],                    fat_v["optimo"]),
        ctrl.Rule(acwr_v["optimo"] & delta_v["mejora"],                     fat_v["optimo"]),
        ctrl.Rule(acwr_v["bajo"]   & delta_v["mejora"],                     fat_v["optimo"]),
        ctrl.Rule(acwr_v["bajo"]   & delta_v["estable"],                    fat_v["alerta_temprana"]),
        ctrl.Rule(acwr_v["alto"]   & delta_v["estable"] & b28_v["alto"],    fat_v["alerta_temprana"]),
        ctrl.Rule(acwr_v["alto"]   & delta_v["estable"] & b28_v["medio"],   fat_v["alerta_temprana"]),
        ctrl.Rule(acwr_v["alto"]   & delta_v["fatiga"],                     fat_v["fatiga_acumulada"]),
        ctrl.Rule(acwr_v["optimo"] & delta_v["fatiga"],                     fat_v["alerta_temprana"]),
        ctrl.Rule(acwr_v["alto"]   & ba_v["alto"] & b28_v["bajo"],          fat_v["fatiga_acumulada"]),
        ctrl.Rule(acwr_v["alto"]   & ba_v["alto"] & b28_v["alto"],          fat_v["critico"]),
        ctrl.Rule(delta_v["fatiga"] & ba_v["alto"],                         fat_v["fatiga_acumulada"]),
        ctrl.Rule(delta_v["fatiga"] & ba_v["alto"] & b28_v["bajo"],         fat_v["critico"]),
        ctrl.Rule(zmeso_v["alto"]  & acwr_v["alto"],                        fat_v["fatiga_acumulada"]),
        ctrl.Rule(zmeso_v["alto"]  & acwr_v["alto"] & delta_v["fatiga"],    fat_v["critico"]),
        ctrl.Rule(zmeso_v["bajo"]  & acwr_v["optimo"],                      fat_v["optimo"]),
        ctrl.Rule(zmeso_v["medio"] & acwr_v["optimo"],                      fat_v["alerta_temprana"]),
        ctrl.Rule(b28_v["alto"]    & ba_v["bajo"],                          fat_v["alerta_temprana"]),
        ctrl.Rule(b28_v["bajo"]    & ba_v["alto"],                          fat_v["fatiga_acumulada"]),
        ctrl.Rule(b28_v["bajo"]    & ba_v["alto"] & delta_v["fatiga"],      fat_v["critico"]),
        ctrl.Rule(acwr_v["bajo"]   & b28_v["bajo"] & ba_v["bajo"],          fat_v["optimo"]),
        ctrl.Rule(acwr_v["bajo"]   & b28_v["alto"] & delta_v["estable"],    fat_v["alerta_temprana"]),
        ctrl.Rule(acwr_v["alto"]   & b28_v["alto"] & delta_v["fatiga"],     fat_v["critico"]),
        ctrl.Rule(acwr_v["optimo"] & b28_v["medio"] & zmeso_v["medio"],     fat_v["alerta_temprana"]),
    ]

    sistema    = ctrl.ControlSystem(reglas)
    simulacion = ctrl.ControlSystemSimulation(sistema)
    vars_tuple = (acwr_v, delta_v, zmeso_v, ba_v, b28_v, fat_v)
    return vars_tuple, simulacion


def calcular_membresias_atleta(indice_fatiga: float) -> dict[str, float]:
    """
    Calcula el grado de pertenencia μ del índice de fatiga de un atleta
    en cada conjunto del output del motor Mamdani.

    Retorna: {"optimo": μ, "alerta_temprana": μ, "fatiga_acumulada": μ, "critico": μ}
    """
    import skfuzzy as fuzz

    vars_tuple, _ = construir_motor_fuzzy_cached()
    _acwr_v, _delta_v, _zmeso_v, _ba_v, _b28_v, _fat_v = vars_tuple
    u_fat = _fat_v.universe
    return {
        "optimo":           float(fuzz.interp_membership(u_fat, _fat_v["optimo"].mf,           indice_fatiga)),
        "alerta_temprana":  float(fuzz.interp_membership(u_fat, _fat_v["alerta_temprana"].mf,  indice_fatiga)),
        "fatiga_acumulada": float(fuzz.interp_membership(u_fat, _fat_v["fatiga_acumulada"].mf, indice_fatiga)),
        "critico":          float(fuzz.interp_membership(u_fat, _fat_v["critico"].mf,          indice_fatiga)),
    }


def _cache_data_ttl(fn, ttl=30):
    if _STREAMLIT_AVAILABLE:
        return st.cache_data(ttl=ttl)(fn)  # type: ignore[misc]
    return fn


@_cache_data_ttl
def calcular_historial_batch_cached(
    df_raw: pd.DataFrame,
    atletas: tuple[str, ...],
    ventana_meso: int,
) -> dict[str, pd.DataFrame]:
    """
    Calcula historial de fatiga por atleta (últimas 12 sesiones).
    O(N) cacheado — no usar pipeline_historial directo.
    """
    result: dict[str, pd.DataFrame] = {}
    for atleta in atletas:
        df_ath = df_raw[df_raw["Nombre"] == atleta].copy()
        if df_ath.empty:
            continue
        df_ath = df_ath.sort_values("Fecha").tail(12).reset_index(drop=True)
        # Calcular índice de fatiga simplificado (en prod: via función PG)
        if "VMP_Hoy" in df_ath.columns and "VMP_Ref" in df_ath.columns:
            df_ath["delta_pct"] = ((df_ath["VMP_Hoy"] - df_ath["VMP_Ref"])
                                   / df_ath["VMP_Ref"].replace(0, float("nan")) * 100)
            df_ath["fatiga"] = (100 - df_ath["delta_pct"].clip(-30, 30)
                                .apply(lambda x: (x + 30) / 60 * 100)).clip(0, 100)
        else:
            df_ath["fatiga"] = 50.0
        df_ath["fecha"] = df_ath["Fecha"].astype(str)
        result[atleta] = df_ath[["fecha", "fatiga"]]
    return result


# ─────────────────────────────────────────────────────────────────────────────
# UI — PESTAÑAS PRINCIPALES
# ─────────────────────────────────────────────────────────────────────────────

def tab_ingreso(atletas_lista: list[str], df_raw: pd.DataFrame):
    """
    3 sub-pestañas:
      1. 🏃 Velocidad (VMP)      — sin cambio
      2. 💤 Wellness             — Individual (sliders) + Masivo (data_editor)
      3. 🏋️ Carga Grupal        — tabla de ejercicios → aplica a todos los atletas
    """
    sub_vel, sub_well, sub_carga = st.tabs([
        "🏃 Velocidad (VMP)",
        "💤 Wellness",
        "🏋️ Carga Grupal",
    ])

    # =========================================================================
    #  SUB-TAB 1 — VELOCIDAD (VMP)
    # =========================================================================
    with sub_vel:
        st.markdown("### ➕ Registrar Sesión VMP")
        st.caption("Velocidad media propulsiva ante carga submaximal fija (40-60% 1RM).")

        # Guard V-DOS en importación masiva CSV
        with st.expander("📂 Importación masiva CSV"):
            file_imp = st.file_uploader(
                "Subir CSV (Nombre, Fecha, VMP_Hoy)",
                type=["csv"],
                key="imp_vmp_file",
            )
            if file_imp is not None:
                df_imp = pd.read_csv(file_imp)
                df_imp =normalize_columns(df_imp)

                # ── Guard V-DOS UI ────────────────────────────────────────────
                if len(df_imp) > db.MAX_IMPORT_ROWS:
                    st.error(
                        f"🚫 El archivo contiene **{len(df_imp)} filas**, "
                        f"superando el límite operativo de **{db.MAX_IMPORT_ROWS}**. "
                        "Divide el CSV en lotes y vuelve a importar."
                    )
                    return
                # ─────────────────────────────────────────────────────────────

                anomalias = df_imp[df_imp["VMP_Hoy"] > 2.50]
                if not anomalias.empty:
                    st.warning(f"⚠️ {len(anomalias)} filas con VMP > 2.50 m/s (fuera de rango).")

                st.info(f"Vista previa: {len(df_imp)} filas")
                st.dataframe(df_imp.head(5), use_container_width=True, hide_index=True)

                if st.button("📥 Importar VMP", key="btn_imp_vmp"):
                    ins, omi, errs = db.importar_dataframe(df_imp)
                    if errs:
                        st.error("\n".join(errs))
                    else:
                        st.success(f"✅ {ins} insertados, {omi} omitidos.")
                        st.cache_data.clear()
                        st.rerun()

        st.markdown("---")
        col_a, col_b = st.columns(2)
        with col_a:
            atleta_sel = st.selectbox("Atleta", atletas_lista, key="vmp_atleta")
        with col_b:
            fecha_vmp = st.date_input(
                "Fecha", value=date.today(), max_value=date.today(), key="vmp_fecha"
            )
        vmp_hoy = st.number_input(
            "VMP hoy (m/s)", min_value=0.10, max_value=2.50,
            value=0.80, step=0.01, format="%.3f", key="vmp_hoy"
        )
        notas_vmp = st.text_input("Notas", key="vmp_notas")

        if st.button("💾 Guardar VMP", type="primary", key="btn_vmp"):
            ok, msg = db.insertar_sesion(atleta_sel, fecha_vmp, vmp_hoy, notas=notas_vmp)
            if ok:
                st.success(msg)
                st.cache_data.clear()
                st.rerun()
            else:
                st.error(msg)

    # =========================================================================
    #  SUB-TAB 2 — WELLNESS
    # =========================================================================
    with sub_well:
        modo_well = st.radio(
            "Modalidad de registro",
            ["👤 Individual (sliders)", "👥 Masivo (tabla)"],
            horizontal=True,
            key="well_modo",
        )

        if modo_well == "👤 Individual (sliders)":
            st.markdown("### 💤 Cuestionario de Wellness (Hooper Modificado)")
            st.caption(
                "5 ítems en escala Likert 1–7. "
                "Escala inversa: **Sueño/Fatiga/Estrés/Dolor** → 1 = óptimo. "
                "Escala directa: **Humor** → 7 = óptimo."
            )
            col_w0, col_w_fecha = st.columns([2, 2])
            with col_w0:
                atleta_well = st.selectbox("Atleta", atletas_lista, key="well_atleta")
            with col_w_fecha:
                fecha_well = st.date_input(
                    "Fecha", value=date.today(), max_value=date.today(), key="well_fecha"
                )
            col_w1, col_w2, col_w3 = st.columns(3)
            with col_w1:
                w_sueno  = st.slider("😴 Sueño (1=óptimo)",  1, 7, 4, key="well_sueno")
                w_fatiga = st.slider("😓 Fatiga (1=óptimo)", 1, 7, 4, key="well_fatiga")
            with col_w2:
                w_estres = st.slider("😰 Estrés (1=óptimo)", 1, 7, 4, key="well_estres")
                w_dolor  = st.slider("🦵 Dolor (1=sin dolor)", 1, 7, 4, key="well_dolor")
            with col_w3:
                w_humor  = st.slider("😊 Humor (7=óptimo)", 1, 7, 4, key="well_humor")

            notas_well = st.text_input("Notas (opcional)", key="well_notas")

            if st.button("💾 Guardar Wellness", type="primary", key="btn_guardar_well"):
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
            # ── Wellness Masivo (data_editor) ────────────────────────────────
            st.markdown("### 👥 Registro Masivo de Wellness")

            # Guard V-DOS en importación masiva CSV wellness
            with st.expander("📂 Importación masiva CSV Wellness"):
                file_well_imp = st.file_uploader(
                    "Subir CSV Wellness",
                    type=["csv"],
                    key="imp_well_file",
                )
                if file_well_imp is not None:
                    df_well_imp = pd.read_csv(file_well_imp)

                    # ── Guard V-DOS UI ────────────────────────────────────────
                    if len(df_well_imp) > db.MAX_IMPORT_ROWS:
                        st.error(
                            f"🚫 El archivo contiene **{len(df_well_imp)} filas**, "
                            f"superando el límite de **{db.MAX_IMPORT_ROWS}**. "
                            "Divide el CSV en lotes más pequeños."
                        )
                        return
                    # ─────────────────────────────────────────────────────────

                    st.info(f"Vista previa: {len(df_well_imp)} filas")
                    st.dataframe(df_well_imp.head(5), use_container_width=True)

                    if st.button("📥 Importar Wellness CSV", key="btn_imp_well"):
                        ins, omi, errs = db.importar_wellness_dataframe(df_well_imp)
                        if errs:
                            st.error("\n".join(errs))
                        else:
                            st.success(f"✅ {ins} insertados, {omi} omitidos.")
                            st.cache_data.clear()
                            st.rerun()

            st.caption(
                "Edita los valores directamente en la tabla. "
                "Escala 1–7 · Sueño/Estrés/Fatiga/Dolor: 1 = óptimo · Humor: 7 = óptimo."
            )
            fecha_masiva = st.date_input(
                "Fecha de la sesión", value=date.today(),
                max_value=date.today(), key="well_masiva_fecha",
            )
            df_editor = db.wellness_masivo_template(atletas_lista)
            df_editado = st.data_editor(
                df_editor,
                use_container_width=True,
                hide_index=True,
                num_rows="fixed",
                column_config={
                    "Nombre":  st.column_config.TextColumn("Atleta", disabled=True),
                    "Sueño":   st.column_config.NumberColumn("😴 Sueño",  min_value=1, max_value=7, step=1),
                    "Estrés":  st.column_config.NumberColumn("😰 Estrés", min_value=1, max_value=7, step=1),
                    "Fatiga":  st.column_config.NumberColumn("😓 Fatiga", min_value=1, max_value=7, step=1),
                    "Humor":   st.column_config.NumberColumn("😊 Humor",  min_value=1, max_value=7, step=1),
                    "Dolor":   st.column_config.NumberColumn("🦵 Dolor",  min_value=1, max_value=7, step=1),
                },
                key="well_masiva_editor",
            )

            if st.button("💾 Guardar Wellness Masivo", type="primary", key="btn_well_masivo"):
                errores_w = []
                insertados_w = 0
                for _, row in df_editado.iterrows():
                    ok, msg = db.insertar_wellness(
                        nombre=row["Nombre"], fecha=fecha_masiva,
                        sueno=int(row["Sueño"]), fatiga_hooper=int(row["Fatiga"]),
                        estres=int(row["Estrés"]), dolor=int(row["Dolor"]),
                        humor=int(row["Humor"]), notas="",
                    )
                    if ok:
                        insertados_w += 1
                    else:
                        errores_w.append(f"{row['Nombre']}: {msg}")
                if errores_w:
                    st.warning(f"⚠️ {len(errores_w)} errores:\n" + "\n".join(errores_w))
                else:
                    st.success(f"✅ Wellness guardado para {insertados_w} atletas.")
                    st.cache_data.clear()
                    st.rerun()

    # =========================================================================
    #  SUB-TAB 3 — CARGA GRUPAL
    # =========================================================================
    with sub_carga:
        st.markdown("### 🏋️ Carga Grupal de Entrenamiento")
        st.caption(
            "Define los ejercicios de la sesión. La carga se asocia a "
            "**todos los atletas seleccionados** para esa fecha."
        )
        col_c_fecha, col_c_notas = st.columns([2, 4])
        with col_c_fecha:
            fecha_carga = st.date_input(
                "Fecha de la sesión", value=date.today(),
                max_value=date.today(), key="carga_fecha",
            )
        with col_c_notas:
            notas_carga = st.text_input(
                "Notas de la sesión", key="carga_notas",
                placeholder="Contexto del entrenador...",
            )

        st.markdown("#### Ejercicios de la sesión")
        df_ejercicios_base = pd.DataFrame({
            "tipo_plataforma": pd.Series([], dtype="str"),
            "altura_salto":    pd.Series([], dtype="float"),
            "n_saltos":        pd.Series([], dtype="int"),
            "tipo_caida":      pd.Series([], dtype="str"),
        })
        df_ejercicios_editado = st.data_editor(
            df_ejercicios_base,
            use_container_width=True,
            hide_index=True,
            num_rows="dynamic",
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
                    "Caída", options=["pie", "mano"], required=True,
                ),
            },
            key="carga_ejercicios_editor",
        )

        st.markdown("#### Atletas participantes")
        atletas_participantes = st.multiselect(
            "Selecciona atletas que realizaron esta sesión:",
            options=atletas_lista,
            default=atletas_lista,
            key="carga_atletas_sel",
        )

        if not df_ejercicios_editado.empty and atletas_participantes:
            total_saltos = int(df_ejercicios_editado["n_saltos"].sum())
            st.metric("Total de saltos en la sesión", total_saltos)

            if st.button("💾 Guardar Carga Grupal", type="primary", key="btn_carga_grupal"):
                ok, errors = db.insertar_carga_grupal_batch(
                    fecha=str(fecha_carga),
                    df_ejercicios=df_ejercicios_editado,
                    atletas=atletas_participantes,
                    notas=notas_carga,
                )
                if ok:
                    st.success(
                        f"✅ Carga grupal guardada para {len(atletas_participantes)} atletas "
                        f"({total_saltos} saltos totales)."
                    )
                    st.cache_data.clear()
                else:
                    st.error("❌ Errores al guardar:\n" + "\n".join(errors))
        else:
            st.info("Agrega al menos un ejercicio y selecciona atletas para guardar la carga.")


def tab_dashboard(
    atletas: list[str],
    df_raw: pd.DataFrame,
    cfg: dict,
    sel: str | None = None,
):
    """
    Dashboard principal de fatiga neuromuscular.
    Limpiado en v4.3: eliminados bloques redundantes.
    """
    if df_raw.empty:
        st.info("Sin datos de sesiones. Registra sesiones en la pestaña Ingreso.")
        return

    sel = sel or (atletas[0] if atletas else None)
    if not sel:
        return

    # ── [A1] Selector de atleta ───────────────────────────────────────────────
    sel = st.selectbox("Atleta", atletas, key="dash_atleta_sel")
    df_sel = df_raw[df_raw["Nombre"] == sel].copy()

    if df_sel.empty:
        st.warning(f"Sin sesiones registradas para {sel}.")
        return

    # ── [A2] KPIs principales ─────────────────────────────────────────────────
    row = df_sel.sort_values("Fecha").iloc[-1]
    indice_fatiga = float(row.get("indice_fatiga", 50.0))

    col_k1, col_k2, col_k3, col_k4 = st.columns(4)
    col_k1.metric("VMP Hoy",         f"{row.get('VMP_Hoy', 0):.3f} m/s")
    col_k2.metric("Índice Fatiga",   f"{indice_fatiga:.1f}")
    col_k3.metric("Sesiones (30d)",  len(df_sel.tail(30)))
    col_k4.metric("Última sesión",   str(row.get("Fecha", "—")))

    # ── [A3] Historial individual — barras + tendencia ────────────────────────
    st.markdown("---")
    st.markdown("### 📈 Historial de Fatiga")
    frames_dict = calcular_historial_batch_cached(
        df_raw,
        (sel,),
        cfg.get("ventana_meso", 28),
    )
    if sel in frames_dict:
        df_hist = frames_dict[sel].copy()
        df_hist["fecha"] = df_hist["fecha"].astype(str)
        st.plotly_chart(
            fig_semaforo_historico(df_hist, titulo=f"Historial — {sel}"),
            use_container_width=True,
        )

    # ── [A4] Panel de membresía fuzzy (solo rol analítico) ────────────────────
    if st.session_state.get("rol_usuario") == "analitico" and sel:
        with st.expander("📐 Ver Funciones de Pertenencia del Modelo"):
            st.caption(
                "Las funciones de pertenencia definen cómo el motor Mamdani interpreta "
                "el índice de fatiga. **μ** (mu) indica en qué medida el valor actual "
                "del atleta pertenece a cada conjunto difuso (0 = no pertenece · 1 = total)."
            )

            membresias = calcular_membresias_atleta(indice_fatiga)
            vars_tuple, _ = construir_motor_fuzzy_cached()
            _acwr_v, _delta_v, _zmeso_v, _ba_v, _b28_v, _fat_v = vars_tuple
            u_fat = _fat_v.universe

            import skfuzzy as fuzz  # noqa: F401
            membership_vals = {
                "Óptimo":  fuzz.interp_membership(u_fat, _fat_v["optimo"].mf,           u_fat),
                "Alerta":  fuzz.interp_membership(u_fat, _fat_v["alerta_temprana"].mf,  u_fat),
                "Fatiga":  fuzz.interp_membership(u_fat, _fat_v["fatiga_acumulada"].mf, u_fat),
                "Crítico": fuzz.interp_membership(u_fat, _fat_v["critico"].mf,          u_fat),
            }
            if _CHARTS_AVAILABLE:
                st.plotly_chart(
                    fig_membership_fuzzy(u_fat, membership_vals),
                    use_container_width=True,
                )

            _CONJUNTOS_INFO = [
                {
                    "key": "optimo", "label": "🟢 Óptimo", "color": "#22c55e",
                    "desc": (
                        "El atleta muestra adaptación positiva. La carga de entrenamiento "
                        "está bien tolerada y el SNC opera con reserva funcional."
                    ),
                    "rango": "75 – 125",
                },
                {
                    "key": "alerta_temprana", "label": "🟡 Alerta Temprana", "color": "#eab308",
                    "desc": (
                        "Señales incipientes de fatiga acumulada (Weakley 2019: ACWR 1.1–1.3). "
                        "Monitorear tendencia en las próximas 48–72 h."
                    ),
                    "rango": "50 – 75",
                },
                {
                    "key": "fatiga_acumulada", "label": "🟠 Fatiga Acumulada", "color": "#f97316",
                    "desc": (
                        "Fatiga neuromuscular significativa. Reducir volumen e intensidad. "
                        "El sistema recomienda sesión regenerativa o descanso activo."
                    ),
                    "rango": "25 – 50",
                },
                {
                    "key": "critico", "label": "🔴 Crítico", "color": "#ef4444",
                    "desc": (
                        "Sobreentrenamiento o fatiga severa del SNC. Descanso obligatorio. "
                        "Evaluar causas sistémicas (sueño, nutrición, estrés extradeportivo)."
                    ),
                    "rango": "0 – 25",
                },
            ]

            st.markdown(f"#### Pertenencia del atleta **{sel}** · Índice: `{indice_fatiga:.1f}`")
            cols_mf = st.columns(4)
            for col_mf, info in zip(cols_mf, _CONJUNTOS_INFO):
                mu = membresias[info["key"]]
                with col_mf:
                    st.markdown(
                        f'<div style="background:#1e293b;border-radius:8px;padding:14px;'
                        f'border-left:4px solid {info["color"]};margin-bottom:8px;">'
                        f'<div style="font-weight:700;color:{info["color"]};font-size:14px;">'
                        f'{info["label"]}</div>'
                        f'<div style="font-size:22px;font-weight:900;color:{info["color"]};'
                        f'margin:6px 0;">μ = {mu:.3f}</div>'
                        f'<div style="font-size:11px;color:#94a3b8;">Rango: {info["rango"]}</div>'
                        f'<hr style="border-color:#334155;margin:8px 0;"/>'
                        f'<div style="font-size:11px;color:#cbd5e1;">{info["desc"]}</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )


def main():
    """Entry point Streamlit."""
    st.set_page_config(
        page_title="NMF-Optimizer v4.3",
        page_icon="⚡",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.title("⚡ NMF-Optimizer v4.3 — Monitoreo de Fatiga Neuromuscular")

    cfg = {"ventana_meso": 28}
    atletas = db.cargar_atletas() or ["Atleta Demo"]
    df_raw  = db.cargar_sesiones_raw()

    tab_ing, tab_dash, tab_hist = st.tabs([
        "➕ Ingreso",
        "📊 Dashboard",
        "✏️ Historial / Edición",
    ])
    with tab_ing:
        tab_ingreso(atletas, df_raw)
    with tab_dash:
        tab_dashboard(atletas, df_raw, cfg)
    with tab_hist:
        st.info("Historial y edición de sesiones.")


if __name__ == "__main__":
    main()
