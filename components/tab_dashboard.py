"""
components/tab_dashboard.py — NMF-Optimizer v4.4
Dashboard de fatiga neuromuscular para un atleta seleccionado.
Usa logic.services.pipeline_diagnostico + visualization.charts.
"""
from __future__ import annotations
import pandas as pd
import streamlit as st
from logic.services import pipeline_diagnostico, calcular_historial_fatiga, calcular_metricas 

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

    # --- Consolidated Metrics and Status ---
    st.markdown("---") # Separator
    st.markdown("### 📊 Indicadores Clave y Estado")

    # Display Ratio (Δ% vs MMC28), DQI, and other metrics
    # Use columns to arrange these metrics.
    c_meta1, c_meta2, c_meta3, c_meta4, c_meta5 = st.columns(5)

    c_meta1.metric(
        label="Ratio (Δ% vs MMC28)",
        value=f"{resultado['delta_pct']:+.1f}%",
    )
    c_meta2.metric(
        label=f"DQI ({resultado['calidad_dato']})",
        value=f"{resultado['dqi']:.2f}",
    )
    c_meta3.metric("Z-Score Meso", f"{resultado['z_meso']:+.2f}")
    c_meta4.metric("β₇ Aguda",   f"{resultado['beta_aguda']:+.4f}")
    c_meta5.metric("β₂₈ Crónica",f"{resultado['beta_28']:+.4f}")

    # Status semáforo - Moved before warnings/notes
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

    # Warnings and notes remain after the status message.
    if resultado["advertencias"]:
        for adv in resultado["advertencias"]:
            st.warning(adv)

    if resultado["nota_swc"]:
        st.info(resultado["nota_swc"])

    # --- End of consolidated block ---

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

    # --- NEW SECTION FOR VMP/RATIO CHART ---
    st.markdown("---")
    st.markdown("### 📊 VMP y Ratio con Umbrales")
    try:
        from visualization.charts import fig_vmp_ratio_thresholds

        # Prepare historical data for VMP, Ratio, and thresholds.
        # This assumes df_sel contains historical session data for the selected athlete.
        # We need 'fecha', 'vmp_hoy' (for vmp), and ideally 'mmc28' to calculate 'ratio'.
        # Thresholds are assumed constants for now.

        df_hist_vmp_ratio = df_sel[['fecha', 'vmp_hoy']].copy()
        df_hist_vmp_ratio.rename(columns={'vmp_hoy': 'vmp'}, inplace=True)

        # Calculate Ratio and Thresholds - This part might require access to 'mmc28'
        # or a dedicated function from logic.services to calculate historical ratios.
        # For demonstration, we will mock these values if 'mmc28' is not present.
        # In a real implementation, you'd want to calculate these accurately.

        if 'mmc28' in df_sel.columns:
            df_hist_vmp_ratio['ratio'] = df_sel['vmp_hoy'] / df_sel['mmc28']
            # Define thresholds, e.g., 15% below and 15% above a baseline ratio (which might be 1.0 or avg ratio)
            # For simplicity, let's use fixed thresholds for now.
            df_hist_vmp_ratio['threshold_low'] = 0.85
            df_hist_vmp_ratio['threshold_high'] = 1.15
        else:
            # Mocking ratio and thresholds if 'mmc28' is not available.
            # This is a placeholder and should be replaced with actual calculation.
            st.warning("`mmc28` column not found in historical data. Mocking Ratio and Thresholds.")
            df_hist_vmp_ratio['ratio'] = df_hist_vmp_ratio['vmp'] / 10 # Dummy ratio calculation
            df_hist_vmp_ratio['threshold_low'] = 0.8
            df_hist_vmp_ratio['threshold_high'] = 1.2

        # Ensure date format is correct for plotting
        df_hist_vmp_ratio['fecha'] = pd.to_datetime(df_hist_vmp_ratio['fecha']).dt.strftime('%Y-%m-%d')
        # Sort by date and take the last N entries if needed
        df_hist_vmp_ratio = df_hist_vmp_ratio.sort_values('fecha').tail(30) # Last 30 days

        if not df_hist_vmp_ratio.empty:
            st.plotly_chart(
                fig_vmp_ratio_thresholds(df_hist_vmp_ratio, sel),
                use_container_width=True,
            )
        else:
            st.info("Datos históricos para VMP y Ratio no disponibles.")

    except ImportError:
        st.warning("Could not import `fig_vmp_ratio_thresholds`. Ensure `visualization.charts` is correctly set up.")
    except Exception as exc:
        st.warning(f"No se pudo renderizar el gráfico VMP/Ratio: {exc}")
    # --- END NEW SECTION ---

    # --- NEW SECTION: Vista Grupal ---
    st.markdown("---")
    st.markdown("### 👥 Vista Grupal")

    # Get data for all athletes
    all_athlete_metrics = []
    ventana = cfg.get("ventana_meso", 28)

    # Filter out athletes with no data to avoid errors in pipeline_diagnostico
    athletes_with_data = df_raw['nombre'].unique()

    # Use a try-except block for robustness in data fetching
    try:
        for athlete_name in atletas:
            if athlete_name in athletes_with_data:
                result = pipeline_diagnostico(athlete_name, df_raw, simulador, ventana)
                if result: # Ensure result is not None (e.g., if not enough data)
                    all_athlete_metrics.append({
                        "Atleta": athlete_name,
                        "Índice de Fatiga": result.get("indice_fatiga", "N/A"),
                        "Estado": result.get("estado", "N/A"),
                        "DQI": result.get("dqi", "N/A"),
                        "Δ% vs MMC28": f"{result.get('delta_pct', 0):.1f}%" if isinstance(result.get('delta_pct'), (int, float)) else "N/A",
                        "Recomendación": result.get("accion", "N/A"),
                    })

        if all_athlete_metrics:
            df_group = pd.DataFrame(all_athlete_metrics)
            # Define the desired column order
            column_order = ['Atleta', 'Índice de Fatiga', 'Estado', 'DQI', 'Δ% vs MMC28', 'Recomendación']

            # Reorder DataFrame columns to match the desired order
            # Ensure all columns in column_order exist in df_group before reordering
            existing_columns = [col for col in column_order if col in df_group.columns]
            df_group = df_group[existing_columns]

            st.dataframe(
                df_group,
                use_container_width=True,
                hide_index=True # Hide the default DataFrame index
            )
        else:
            st.info("No hay datos suficientes para mostrar la vista grupal.")

    except Exception as e:
        st.error(f"Error al cargar la vista grupal: {e}")
    # --- END NEW SECTION ---

    # --- Panel de membresía fuzzy (solo rol analítico) ─────────────────────────
    if st.session_state.get("rol_usuario") == "analitico":
        _render_membership_panel(resultado, simulador)
