"""
components/tab_lesiones.py — Pestaña de Gestión de Lesiones v1.0
Punto de entrada: render_tab_lesiones().

Estructura interna:
  _render_kpi_banner()       — métricas rápidas del equipo
  _render_form_registro()    — formulario para registrar nueva lesión
  _render_seguimiento_activo() — lista expandible de lesiones activas con edición de estado
  _render_historial()        — historial completo por atleta + timeline Plotly
"""
from __future__ import annotations

from datetime import date
from typing import Optional

import pandas as pd
import plotly.express as px
import streamlit as st

from core.schemas import (
    ZONA_CORPORAL_OPTIONS,
    TIPO_LESION_OPTIONS,
    GRAVEDAD_OPTIONS,
    ESTADO_LESION_OPTIONS,
    TipoTejido,
    MecanismoInicio,
    HistorialRecurrencia,
)
from core.services import resumen_lesiones_equipo
from data.db import (
    insertar_lesion,
    cargar_historial_lesiones,
    actualizar_estado_lesion,
)

_GRAVEDAD_COLOR: dict[str, str] = {
    "Leve":     "#16a34a",
    "Moderada": "#ca8a04",
    "Grave":    "#dc2626",
}


def render_tab_lesiones(atletas: list[str], df_lesiones: pd.DataFrame) -> None:
    """
    Punto de entrada principal para Tab 6.

    Parámetros
    ----------
    atletas     : Lista de nombres de atletas activos (de db.cargar_atletas()).
    df_lesiones : DataFrame de lesiones activas del equipo
                  (de db.cargar_lesiones_activas(), ya filtrado a estados Activa/Recuperación).
    """
    st.header("🩹 Gestión de Lesiones")

    resumen = resumen_lesiones_equipo(df_lesiones)
    _render_kpi_banner(resumen)
    st.divider()

    tab_reg, tab_seg, tab_hist = st.tabs(
        ["➕ Registrar", "📋 Seguimiento Activo", "📈 Historial"]
    )
    with tab_reg:
        _render_form_registro(atletas)
    with tab_seg:
        _render_seguimiento_activo(df_lesiones)
    with tab_hist:
        _render_historial(atletas)


def _render_kpi_banner(resumen: dict) -> None:
    col1, col2, col3 = st.columns(3)
    col1.metric("⚡ Lesiones Activas", resumen["total_activas"])
    col2.metric(
        "📍 Zona más afectada",
        max(resumen["por_zona"], key=resumen["por_zona"].get)
        if resumen["por_zona"] else "—",
    )
    col3.metric("👤 Atletas afectados", resumen["total_atletas_afectados"])


def _render_form_registro(atletas: list[str]) -> None:
    st.subheader("Registrar Nuevo Evento Médico")
    
    # Manejo de estado para la reactividad del formulario
    if "tipo_evento" not in st.session_state:
        st.session_state.tipo_evento = "Lesión"

    def update_tipo():
        st.session_state.tipo_evento = st.session_state.reg_tipo_evento

    tipo_evento = st.radio(
        "Tipo de Evento", ["Lesión", "Enfermedad"], 
        key="reg_tipo_evento", 
        on_change=update_tipo
    )

    with st.form("form_evento_nuevo", clear_on_submit=True):
        col1, col2 = st.columns(2)
        with col1:
            atleta_sel = st.selectbox("Atleta", atletas, key="reg_atleta")
            fecha_inicio = st.date_input("Fecha de Inicio", value=date.today(), key="reg_fecha")
            fecha_alta   = st.date_input("Fecha de Alta", value=None, key="reg_fecha_alta")
            estado_ini = st.selectbox("Estado", ESTADO_LESION_OPTIONS, index=0, key="reg_estado")
        
        with col2:
            if st.session_state.tipo_evento == "Lesión":
                zona       = st.selectbox("Zona Corporal", ZONA_CORPORAL_OPTIONS, key="reg_zona")
                sistema    = st.selectbox("Sistema/Tipo", TIPO_LESION_OPTIONS, key="reg_sistema")
                tipo_enfermedad = None
                es_contagiosa = False
            else: # Enfermedad
                zona = None
                sistema = None
                tipo_enfermedad = st.selectbox("Tipo de Enfermedad", ["Respiratoria", "Digestiva", "Infecciosa", "Otra"], key="reg_tipo_enf")
                es_contagiosa = st.checkbox("¿Es contagiosa?", key="reg_contagiosa")
        
        notas = st.text_area("Notas", key="reg_notas")
        submitted = st.form_submit_button("💾 Registrar Evento", type="primary")

    if submitted:
        # Preparar argumentos dinámicos
        args = {
            "atleta": atleta_sel,
            "fecha_inicio": fecha_inicio,
            "tipo_evento": st.session_state.tipo_evento,
            "estado": estado_ini,
            "notas": notas,
            "fecha_alta_medica": fecha_alta
        }
        if st.session_state.tipo_evento == "Lesión":
            args["zona_cuerpo"] = zona
            args["sistema"] = sistema
        else:
            args["tipo_enfermedad"] = tipo_enfermedad
            args["es_contagiosa"] = es_contagiosa

        ok, msg = insertar_lesion(**args)
        if ok:
            st.success(msg)
            # Limpiar estado si es necesario
            st.rerun()
        else:
            st.error(msg)


def _render_seguimiento_activo(df_lesiones: pd.DataFrame) -> None:
    st.subheader("Lesiones Activas y en Recuperación")
    if df_lesiones.empty:
        st.info("✅ No hay lesiones activas en el equipo.")
        return

    for idx, row in df_lesiones.iterrows():
        fecha_str = str(row["fecha_inicio"])[:10]

        with st.expander(
            f"**{row['atleta']}** — {row['tipo_evento']} ({row['estado']}) | Desde: {fecha_str}",
        ):
            col_info, col_accion = st.columns([2, 1])
            with col_info:
                st.markdown(f"**Zona/Tipo:** {row.get('zona_cuerpo') or row.get('tipo_enfermedad', 'N/A')}")
                if row.get("notas"):
                    st.markdown(f"**Notas:** {row['notas']}")
            with col_accion:
                nuevo_estado = st.selectbox(
                    "Actualizar estado",
                    ESTADO_LESION_OPTIONS,
                    index=ESTADO_LESION_OPTIONS.index(row["estado"]),
                    key=f"sel_estado_{row['id']}",
                )
                
                # Campo para notas del cambio (historial)
                notas_cambio = st.text_input("Notas del cambio", key=f"notas_cambio_{row['id']}")
                
                fecha_alta_input: Optional[date] = None
                if nuevo_estado == "Alta":
                    fecha_alta_input = st.date_input(
                        "Fecha de Alta",
                        value=date.today(),
                        key=f"fecha_alta_{row['id']}",
                    )
                
                if st.button("Guardar", key=f"btn_guardar_{row['id']}"):
                    ok, msg = actualizar_estado_lesion(
                        lesion_id=str(row["id"]),
                        nuevo_estado=nuevo_estado,
                        notas=notas_cambio,
                        fecha_alta=fecha_alta_input,
                    )
                    if ok:
                        st.success(msg)
                        st.rerun()
                    else:
                        st.error(msg)


def _render_historial(atletas: list[str]) -> None:
    st.subheader("Historial por Atleta")
    atleta_sel = st.selectbox("Seleccionar atleta", atletas, key="hist_atleta_sel")
    df_hist = cargar_historial_lesiones(atleta_sel)

    if df_hist.empty:
        st.info(f"Sin lesiones registradas para **{atleta_sel}**.")
        return

    # ── Tabla resumen ──────────────────────────────────────────────────────────
    cols_display = [c for c in
                    ["fecha_inicio", "zona_cuerpo", "sistema", "estado", "fecha_alta", "notas"]
                    if c in df_hist.columns]
    st.dataframe(
        df_hist[cols_display].rename(columns={
            "fecha_inicio":  "Fecha Inicio",
            "zona_cuerpo":   "Zona",
            "sistema":       "Sistema",
            "estado":        "Estado",
            "fecha_alta":    "Fecha Alta",
            "notas":         "Notas",
        }),
        use_container_width=True,
        hide_index=True,
    )

    # ── Línea de tiempo Plotly (sólo si hay ≥ 1 lesión con fecha_alta) ──────
    if "fecha_alta" not in df_hist.columns:
        return

    df_timeline = df_hist.copy()
    df_timeline["Inicio"] = pd.to_datetime(df_timeline["fecha_inicio"], errors="coerce")
    df_timeline["Fin"]    = pd.to_datetime(
        df_timeline["fecha_alta"].fillna(str(date.today())), errors="coerce"
    )
    df_timeline = df_timeline.dropna(subset=["Inicio"])

    if df_timeline.empty:
        return

    fig = px.timeline(
        df_timeline,
        x_start="Inicio",
        x_end="Fin",
        y="zona_cuerpo",
        color="sistema",
        hover_data=["sistema", "estado", "notas"],
        title=f"Línea de Tiempo de Lesiones — {atleta_sel}",
        labels={"zona_cuerpo": "Zona Corporal"},
    )
    fig.update_layout(showlegend=True, height=300)
    st.plotly_chart(fig, use_container_width=True)
