"""
components/tab_dashboard.py — NMF-Optimizer v4.4
Dashboard de fatiga neuromuscular para un atleta seleccionado.
Usa logic.services.pipeline_diagnostico + visualization.charts.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from logic.services import pipeline_diagnostico, calcular_historial_fatiga


def _cache_data_ttl(fn, ttl: int = 30):
    """Wrapper condicional para st.cache_data."""
    try:
        return st.cache_data(ttl=ttl)(fn)
    except Exception:
        return fn


@_cache_data_ttl
def _historial_cached(
    df_raw: pd.DataFrame,
    atleta: str,
    _simulador,  # <--- ¡Agrega el guion bajo aquí!
    ventana_meso: int,
) -> pd.DataFrame:
    return calcular_historial_fatiga(df_raw, atleta, _simulador, ventana_meso)

def render_tab_dashboard(
    atletas: list[str],
    df_raw: pd.DataFrame,
    simulador,
    cfg: dict,
) -> None:
    """
    Renderiza el dashboard principal.

    Parámetros
    ----------
    atletas   : Lista de nombres de atletas activos.
    df_raw    : DataFrame de sesiones_vmp (snake_case, desde db.cargar_sesiones_raw).
    simulador : ControlSystemSimulation del motor Mamdani (cacheado en app.py).
    cfg       : Configuración de la sesión {"ventana_meso": int}.
    """
    if df_raw.empty:
        st.info("Sin sesiones registradas. Usa la pestaña ➕ Ingreso para añadir datos.")
        return

    # ── Selector de atleta ────────────────────────────────────────────────────
    sel = st.selectbox("🏊 Seleccionar atleta", atletas, key="dash_atleta_sel")
    df_sel = df_raw[df_raw["nombre"] == sel].copy()

    if df_sel.empty:
        st.warning(f"Sin sesiones para **{sel}**.")
        return

    ventana = cfg.get("ventana_meso", 28)

    # ── Pipeline diagnóstico ──────────────────────────────────────────────────
    resultado = pipeline_diagnostico(sel, df_raw, simulador, ventana)

    if resultado is None:
        st.info(f"**{sel}** necesita al menos 4 sesiones para el análisis.")
        return

    # ── KPIs principales ──────────────────────────────────────────────────────
    col_k1, col_k2, col_k3, col_k4 = st.columns(4)
    col_k1.metric("VMP Hoy",        f"{resultado['vmp_hoy']:.3f} m/s")
    col_k2.metric("Índice Fatiga",  f"{resultado['indice_fatiga']:.1f}")
    col_k3.metric("ACWR",           f"{resultado['acwr']:.3f}")
    col_k4.metric("Última sesión",  resultado["ultima_fecha"])

    # Estado semáforo
    st.markdown(
        f'<div style="background:#1e293b;border-radius:8px;padding:16px;'
        f'border-left:5px solid {resultado["color"]};margin:12px 0;">'
        f'<span style="font-size:18px;font-weight:700;color:{resultado["color"]};">'
        f'{resultado["estado"]}</span>'
        f'<br><span style="font-size:13px;color:#94a3b8;">'
        f'🎯 {resultado["accion"]}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    if resultado["advertencias"]:
        for adv in resultado["advertencias"]:
            st.warning(adv)

    # ── Métricas Mamdani ──────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 📊 Variables del Motor")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("ACWR",        f"{resultado['acwr']:.3f}")
    c2.metric("Δ% vs MMC28", f"{resultado['delta_pct']:+.1f}%")
    c3.metric("Z-Score Meso",f"{resultado['z_meso']:+.2f}")
    c4.metric("β₇ Aguda",   f"{resultado['beta_aguda']:+.4f}")
    c5.metric("β₂₈ Crónica",f"{resultado['beta_28']:+.4f}")

    st.caption(
        f"DQI: **{resultado['dqi']:.2f}** ({resultado['calidad_dato']}) · "
        f"Sesiones: {resultado['n_sesiones']} · "
        f"{resultado['contexto_cientifico']}"
    )

    if resultado["nota_swc"]:
        st.info(resultado["nota_swc"])

    # ── Historial de fatiga — barras + tendencia ──────────────────────────────
    st.markdown("---")
    st.markdown("### 📈 Historial de Fatiga")
    try:
        from visualization.charts import fig_semaforo_historico
        df_hist = _historial_cached(df_raw, sel, simulador, ventana)
        if not df_hist.empty:
            st.plotly_chart(
                fig_semaforo_historico(df_hist, titulo=f"Historial — {sel}"),
                use_container_width=True,
            )
        else:
            st.info("Historial disponible desde la 4ª sesión registrada.")
    except Exception as exc:
        st.warning(f"No se pudo renderizar el historial: {exc}")

    # ── Panel de membresía fuzzy (solo rol analítico) ─────────────────────────
    if st.session_state.get("rol_usuario") == "analitico":
        _render_membership_panel(resultado, simulador)


def _render_membership_panel(resultado: dict, simulador) -> None:
    """Panel de funciones de pertenencia fuzzy (solo rol analítico)."""
    with st.expander("📐 Funciones de Pertenencia del Modelo"):
        st.caption(
            "**μ** indica el grado de pertenencia del índice de fatiga actual "
            "en cada conjunto difuso (0 = no pertenece · 1 = pertenencia total)."
        )
        try:
            import skfuzzy as fuzz
            from visualization.charts import fig_membership_fuzzy

            # Obtener variables del simulador
            fat_v = simulador.ctrl.consequents[0]
            u_fat = fat_v.universe
            membership_vals = {
                "Óptimo":  fuzz.interp_membership(u_fat, fat_v["optimo"].mf,            u_fat),
                "Alerta":  fuzz.interp_membership(u_fat, fat_v["alerta_temprana"].mf,   u_fat),
                "Fatiga":  fuzz.interp_membership(u_fat, fat_v["fatiga_acumulada"].mf,  u_fat),
                "Crítico": fuzz.interp_membership(u_fat, fat_v["critico"].mf,           u_fat),
            }
            st.plotly_chart(fig_membership_fuzzy(u_fat, membership_vals), use_container_width=True)

            # Tarjetas μ por conjunto
            indice = resultado["indice_fatiga"]
            CONJUNTOS = [
                {"key": "optimo",           "label": "🟢 Óptimo",          "color": "#22c55e", "rango": "75–100"},
                {"key": "alerta_temprana",  "label": "🟡 Alerta Temprana", "color": "#eab308", "rango": "50–75"},
                {"key": "fatiga_acumulada", "label": "🟠 Fatiga Acumulada","color": "#f97316", "rango": "25–50"},
                {"key": "critico",          "label": "🔴 Crítico",         "color": "#ef4444", "rango": "0–25"},
            ]
            st.markdown(f"#### μ del atleta · Índice: `{indice:.1f}`")
            cols_mf = st.columns(4)
            for col_mf, info in zip(cols_mf, CONJUNTOS):
                mu = float(fuzz.interp_membership(u_fat, fat_v[info["key"]].mf, indice))
                with col_mf:
                    st.markdown(
                        f'<div style="background:#1e293b;border-radius:8px;padding:14px;'
                        f'border-left:4px solid {info["color"]};">'
                        f'<div style="font-weight:700;color:{info["color"]};">{info["label"]}</div>'
                        f'<div style="font-size:22px;font-weight:900;color:{info["color"]};">'
                        f'μ = {mu:.3f}</div>'
                        f'<div style="font-size:11px;color:#94a3b8;">Rango: {info["rango"]}</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
        except Exception as exc:
            st.warning(f"Panel de membresía no disponible: {exc}")
