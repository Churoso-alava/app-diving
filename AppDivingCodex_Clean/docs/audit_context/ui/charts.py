"""
visualization/charts.py — NMF-Optimizer v4.4
Funciones Plotly (tema oscuro). Sin dependencias de Streamlit.

FASE 5 — Auditoría de esquema:
  sesiones_vmp devuelve columnas: nombre, fecha, vmp_hoy, vmp_ref, notas, created_at
  Los DataFrames de historial tienen columnas: fecha (str), fatiga (float)
  fig_semaforo_historico: barras por zona + curva de tendencia (tests Task 5 v4.3)
"""
from __future__ import annotations

import logging
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from ui.themes import COLORS, STATUS_COLOR

log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Layout oscuro compartido
# ─────────────────────────────────────────────────────────────────────────────

_DARK_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor=COLORS["card_bg"],
    font=dict(family="Inter, sans-serif", color=COLORS["text_primary"]),
    margin=dict(l=12, r=12, t=40, b=12),
    xaxis=dict(
        gridcolor=COLORS["card_border"], zeroline=False, showline=False,
        tickfont=dict(size=10, color=COLORS["text_muted"]),
    ),
    yaxis=dict(
        gridcolor=COLORS["card_border"], zeroline=False, showline=False,
        tickfont=dict(size=10, color=COLORS["text_muted"]),
    ),
    legend=dict(
        bgcolor="rgba(0,0,0,0)", bordercolor="rgba(0,0,0,0)",
        font=dict(size=10, color=COLORS["text_muted"]),
        orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
    ),
    hoverlabel=dict(
        bgcolor=COLORS["card_bg"], bordercolor=COLORS["card_border"],
        font=dict(size=12, color=COLORS["text_primary"]),
    ),
)


# ─────────────────────────────────────────────────────────────────────────────
# VMP TENDENCIA — líneas multi-serie por atleta
# ─────────────────────────────────────────────────────────────────────────────

def fig_vmp_tendencia(df: pd.DataFrame, nombre_atleta: str, delta_pct: float) -> go.Figure:
    """
    Evolución VMP (vmp_hoy), MMA7 y MMC28 para un atleta.
    df columnas: fecha (str), vmp_hoy, mma7, mmc28.
    """
    if df.empty:
        fig = go.Figure()
        fig.update_layout(**_DARK_LAYOUT)
        fig.update_layout(title=f"Sin datos para {nombre_atleta}")
        return fig

    # Validar presencia de columnas requeridas
    required = ["fecha", "vmp_hoy", "mma7", "mmc28"]
    if not all(c in df.columns for c in required):
        log.warning(f"fig_vmp_tendencia: Faltan columnas requeridas {required}")
        fig = go.Figure()
        fig.update_layout(**_DARK_LAYOUT)
        fig.update_layout(title=f"Datos incompletos para {nombre_atleta}")
        return fig

    delta_str = f"+{delta_pct:.1f}%" if delta_pct >= 0 else f"{delta_pct:.1f}%"
    
    # Data Validation and Cleaning
    initial_rows = len(df)
    
    # Filter out rows with NaN in critical numeric columns
    numeric_cols = ['vmp_hoy', 'mma7', 'mmc28']
    df_filtered_numeric = df.dropna(subset=numeric_cols)
    rows_discarded_numeric = initial_rows - len(df_filtered_numeric)
    if rows_discarded_numeric > 0:
        log.warning(f"fig_vmp_tendencia: Discarded {rows_discarded_numeric} rows due to NaN in numeric columns: {numeric_cols}.")

    # Ensure 'fecha' is in a valid format and filter out NaT
    df_filtered_fecha = df_filtered_numeric.copy() # Operate on a copy to avoid SettingWithCopyWarning
    df_filtered_fecha['fecha'] = pd.to_datetime(df_filtered_fecha['fecha'], errors='coerce')
    
    original_rows_after_numeric_filter = len(df_filtered_fecha)
    df_filtered_fecha = df_filtered_fecha.dropna(subset=['fecha'])
    rows_discarded_fecha = original_rows_after_numeric_filter - len(df_filtered_fecha)
    if rows_discarded_fecha > 0:
        log.warning(f"fig_vmp_tendencia: Discarded {rows_discarded_fecha} rows due to NaT in 'fecha' column.")
        
    df_final = df_filtered_fecha
    
    # Log total discarded rows across all filters if any rows were discarded
    total_rows_discarded = initial_rows - len(df_final)
    if total_rows_discarded > 0:
        log.info(f"fig_vmp_tendencia: Total rows discarded due to NaN/NaT: {total_rows_discarded}.")

    # Proceed with plotting if data is not empty after filtering
    if df_final.empty:
        fig = go.Figure()
        fig.update_layout(**_DARK_LAYOUT)
        fig.update_layout(title=f"Sin datos válidos para {nombre_atleta}")
        return fig

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_final["fecha"], y=df_final["vmp_hoy"], mode="lines+markers",
        name="VMP CMJ (m/s)",
        line=dict(color=COLORS["vmp_line"], width=2),
        marker=dict(color=COLORS["vmp_line"], size=5),
    ))
    fig.add_trace(go.Scatter(
        x=df_final["fecha"], y=df_final["mma7"], mode="lines",
        name="MMA7 (Fatiga Aguda)",
        line=dict(color=COLORS["acute_line"], width=2),
    ))
    fig.add_trace(go.Scatter(
        x=df_final["fecha"], y=df_final["mmc28"], mode="lines",
        name="MMC28 (Baseline Crónico)",
        line=dict(color=COLORS["chronic_line"], width=1.5, dash="dot"),
    ))
    fig.add_trace(go.Scatter(
        x=df_final["fecha"], y=df_final["mmc28"] * 0.85, mode="lines",
        name="Umbral Alerta (~15%)",
        line=dict(color=COLORS["alert_dash"], width=1, dash="dash"),
    ))
    fig.add_trace(go.Scatter(
        x=df_final["fecha"], y=df_final["mmc28"] * 0.75, mode="lines",
        name="Umbral Crítico (~25%)",
        line=dict(color=COLORS["critical_dash"], width=1, dash="dash"),
    ))
    fig.update_layout(**_DARK_LAYOUT)
    fig.update_layout(
        title=dict(
            text=f"VMP CMJ — {nombre_atleta}  [Δ vs MMC28: {delta_str}]",
            font=dict(size=13, color=COLORS["text_primary"]), x=0.01,
        ),
        margin=dict(l=40, r=20, t=52, b=32),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# SEMÁFORO BARRAS — todos los atletas en el momento actual
# ─────────────────────────────────────────────────────────────────────────────

def fig_semaforo_barras(df_estado: pd.DataFrame) -> go.Figure:
    """
    Barras horizontales con estado de cada atleta (peores primero).
    df_estado columnas: nombre, score (0-100), estado, fecha.
    """
    if df_estado.empty:
        fig = go.Figure()
        fig.update_layout(**_DARK_LAYOUT)
        fig.update_layout(title="Sin datos para semáforo de barras")
        return fig

    # Validar y limpiar NaN antes de procesar
    df = df_estado.dropna(subset=['score']).sort_values("score", ascending=True).copy()
    if df.empty:
        fig = go.Figure()
        fig.update_layout(**_DARK_LAYOUT)
        fig.update_layout(title="Sin datos válidos para semáforo de barras")
        return fig

    status_clean = {
        "🔴 CRÍTICO": "CRÍTICO",
        "🟠 FATIGA ACUMULADA": "FATIGA ACUMULADA",
        "🟡 ALERTA TEMPRANA": "ALERTA TEMPRANA",
        "🟢 ÓPTIMO": "ÓPTIMO",
        "INSUFICIENTE": "INSUFICIENTE",
    }
    colores = [STATUS_COLOR.get(status_clean.get(str(e), str(e)), COLORS["text_muted"]) for e in df["estado"]]
    labels  = [f"{s:.0f}% — {e}" for s, e in zip(df["score"], df["estado"])]

    fig = go.Figure(go.Bar(
        x=df["score"], y=df["nombre"], orientation="h",
        marker_color=colores, text=labels, textposition="outside",
        textfont=dict(size=11, color=COLORS["text_primary"]),
        customdata=df[["estado", "fecha"]].values,
        hovertemplate=(
            "<b>%{y}</b><br>Score: %{x:.0f}%<br>"
            "Estado: %{customdata[0]}<br>Sesión: %{customdata[1]}<extra></extra>"
        ),
    ))
    fig.add_vline(x=50, line_dash="dot", line_color=COLORS["accum_fatigue"],
                  line_width=1, opacity=0.5)
    fig.add_vline(x=75, line_dash="dot", line_color=COLORS["optimal"],
                  line_width=1, opacity=0.5)
    fig.update_layout(**_DARK_LAYOUT)
    fig.update_layout(
        title=dict(text="Estado Actual — Atletas (ordenado por severidad)",
                   font=dict(size=13), x=0.01),
        margin=dict(l=110, r=60, t=48, b=12),
    )
    fig.update_xaxes(range=[0, 125], showgrid=False, showticklabels=False)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# SEMÁFORO HISTÓRICO — barras + tendencia (FASE 5 — corregido)
# ─────────────────────────────────────────────────────────────────────────────

def fig_semaforo_historico(
    df: pd.DataFrame,
    titulo: str | None = None,
) -> go.Figure:
    """
    Historial de índice de fatiga como barras coloreadas por zona
    con curva de tendencia lineal superpuesta.

    Parámetros
    ----------
    df     : DataFrame con columnas 'fecha' (str) y 'fatiga' (float 0-100).
             Origen: logic.services.calcular_historial_fatiga().
    titulo : Título del gráfico (opcional).

    Umbrales de color:
      crítico  < 25  → #ef4444
      alerta  25–50  → #f97316
      temprana 50–75 → #eab308
      óptimo   ≥ 75  → #22c55e

    Cumple tests TestSemaforoHistoricoChart (v4.3):
      - ≥ 1 go.Bar trace
      - ≥ 1 go.Scatter trace (tendencia)
      - add_shape en y=25, y=50, y=75
    """
    if df.empty:
        fig = go.Figure()
        fig.update_layout(**_DARK_LAYOUT)
        fig.update_layout(title=titulo or "Sin datos")
        return fig

    # Validar y limpiar NaN/NaT antes de procesar
    df = df.dropna(subset=['fecha', 'fatiga'])
    if df.empty:
        fig = go.Figure()
        fig.update_layout(**_DARK_LAYOUT)
        fig.update_layout(title=titulo or "Sin datos válidos")
        return fig

    fechas = df["fecha"].astype(str).tolist()
    fatiga = df["fatiga"].tolist()
    n      = len(fatiga)

    # Color por zona (Fase 5: umbrales correctos)
    zone_color = []
    for v in fatiga:
        if v >= 75:
            zone_color.append("#22c55e")   # óptimo
        elif v >= 50:
            zone_color.append("#eab308")   # alerta temprana
        elif v >= 25:
            zone_color.append("#f97316")   # fatiga acumulada
        else:
            zone_color.append("#ef4444")   # crítico

    # Tendencia lineal (polyfit) si hay suficientes puntos
    if n > 1:
        x_num = np.arange(n, dtype=float)
        slope, intercept = np.polyfit(x_num, fatiga, 1)
        trend_y = (slope * x_num + intercept).tolist()
    else:
        trend_y = fatiga # Solo un punto, la tendencia es el mismo punto

    fig = go.Figure()

    # Barras de fatiga
    fig.add_trace(go.Bar(
        x=fechas, y=fatiga,
        marker_color=zone_color,
        name="Índice Fatiga",
        hovertemplate="<b>%{x}</b><br>Índice: %{y:.1f}<extra></extra>",
    ))

    # Curva de tendencia
    fig.add_trace(go.Scatter(
        x=fechas, y=trend_y,
        mode="lines",
        line=dict(color="#38bdf8", width=2, dash="dot"),
        name="Tendencia",
        hovertemplate="Tendencia: %{y:.1f}<extra></extra>",
    ))

    # Líneas umbral como shapes (y0=y1 → requerido por test)
    for y_val, color, label in [
        (75, "#22c55e", "Óptimo"),
        (50, "#eab308", "Alerta"),
        (25, "#ef4444", "Crítico"),
    ]:
        fig.add_shape(
            type="line",
            x0=0, x1=1, xref="paper",
            y0=y_val, y1=y_val, yref="y",
            line=dict(color=color, width=1, dash="dash"),
        )
        fig.add_annotation(
            x=1.01, xref="paper",
            y=y_val, yref="y",
            text=label, showarrow=False,
            font=dict(size=10, color=color),
            xanchor="left",
        )

    fig.update_layout(**_DARK_LAYOUT)
    fig.update_layout(
        title=titulo or "Historial de Fatiga",
        xaxis_title="Fecha",
        yaxis_title="Índice de Fatiga",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        bargap=0.15,
    )
    fig.update_xaxes(tickangle=-45)
    fig.update_yaxes(range=[0, 125])
    return fig


def fig_fatiga_radial(score: float, label: str) -> go.Figure:
    """
    Crea un indicador radial (gauge) para un atleta.
    """
    # Determinar color basado en umbrales predefinidos
    if score >= 75: color = "#22c55e" # Óptimo
    elif score >= 50: color = "#eab308" # Alerta
    elif score >= 25: color = "#f97316" # Fatiga
    else: color = "#ef4444" # Crítico

    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = score,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': label, 'font': {'size': 12}},
        gauge = {
            'axis': {'range': [0, 100]},
            'bar': {'color': color},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
        }
    ))
    fig.update_layout(height=150, margin={'t':0,'b':0,'l':0,'r':0})
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# HISTORIAL INDIVIDUAL — barras por atleta (mini)
# ─────────────────────────────────────────────────────────────────────────────

def fig_historial_barras_atleta(df_hist: pd.DataFrame, nombre: str) -> go.Figure:
    """
    Barras de historial de fatiga para UN atleta (últimas 12 sesiones).
    df_hist columnas: fecha, fatiga (0-100), estado.
    """
    if df_hist.empty:
        fig = go.Figure()
        fig.update_layout(**_DARK_LAYOUT)
        fig.update_layout(title=f"Sin historial para {nombre}")
        return fig

    # Validar y limpiar NaN antes de procesar
    df = df_hist.dropna(subset=['fatiga']).copy()
    if df.empty:
        fig = go.Figure()
        fig.update_layout(**_DARK_LAYOUT)
        fig.update_layout(title=f"Sin datos válidos para {nombre}")
        return fig

    _STATUS_CLEAN = {
        "🔴 CRÍTICO": "CRÍTICO",
        "🟠 FATIGA ACUMULADA": "FATIGA ACUMULADA",
        "🟡 ALERTA TEMPRANA": "ALERTA TEMPRANA",
        "🟢 ÓPTIMO": "ÓPTIMO",
    }
    df["fecha"] = df["fecha"].astype(str)
    df = df.sort_values("fecha").tail(12)

    def _clean(e: str) -> str:
        return _STATUS_CLEAN.get(str(e), str(e))

    colores = [STATUS_COLOR.get(_clean(e), COLORS["text_muted"]) for e in df["estado"]]

    fig = go.Figure(go.Bar(
        x=df["fecha"], y=df["fatiga"],
        marker_color=colores, marker_line_width=0,
        text=[f"{v:.0f}" for v in df["fatiga"]],
        textposition="outside",
        textfont=dict(size=8, color=COLORS["text_primary"]),
        hovertemplate=(
            f"<b>{nombre}</b><br>Fecha: %{{x}}<br>Score: %{{y:.0f}}<extra></extra>"
        ),
    ))
    for y_val, color in [(25, COLORS["critical"]), (50, COLORS["accum_fatigue"]),
                         (75, COLORS["optimal"])]:
        fig.add_hline(y=y_val, line_dash="dot", line_color=color, opacity=0.35, line_width=1)

    fig.update_layout(**_DARK_LAYOUT)
    fig.update_layout(
        title=dict(text=nombre, font=dict(size=11), x=0.02),
        margin=dict(l=8, r=8, t=28, b=8), height=200, showlegend=False,
    )
    fig.update_yaxes(range=[0, 115], showgrid=False, showticklabels=False, zeroline=False)
    fig.update_xaxes(tickangle=-45, tickfont=dict(size=7, color=COLORS["text_muted"]),
                     showgrid=False)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# FUNCIONES DE MEMBRESÍA FUZZY
# ─────────────────────────────────────────────────────────────────────────────

def fig_membership_fuzzy(x_vals, membership_vals: dict) -> go.Figure:
    """
    Curvas de membresía difusa del output (Plotly).
    membership_vals: {"Óptimo": array, "Alerta": array, "Fatiga": array, "Crítico": array}
    """
    colores_memb = {
        "Óptimo":  COLORS["optimal"],
        "Alerta":  COLORS["early_alert"],
        "Fatiga":  COLORS["accum_fatigue"],
        "Crítico": COLORS["critical"],
    }
    fig = go.Figure()
    for nombre_c, y_vals in membership_vals.items():
        color = colores_memb.get(nombre_c, COLORS["accent"])
        fill_color = color.replace(")", ",0.08)").replace("rgb(", "rgba(")
        fig.add_trace(go.Scatter(
            x=x_vals, y=y_vals, mode="lines", name=nombre_c,
            line=dict(color=color, width=2),
            fill="tozeroy", fillcolor=fill_color,
        ))
    fig.update_layout(**_DARK_LAYOUT)
    fig.update_layout(
        title=dict(text="Funciones de Membresía — Motor Mamdani v4.1",
                   font=dict(size=13), x=0.01),
        margin=dict(l=50, r=20, t=48, b=40),
    )
    fig.update_xaxes(title="Índice de Fatiga")
    fig.update_yaxes(title="Grado de Membresía", range=[0, 1.05])
    return fig

# ─────────────────────────────────────────────────────────────────────────────
# VMP RATIO THRESHOLDS — VMP, Ratio y umbrales de seguridad para atleta
# ─────────────────────────────────────────────────────────────────────────────

def fig_vmp_ratio_thresholds(df: pd.DataFrame, nombre_atleta: str) -> go.Figure:
    """
    Gráfico de líneas para VMP, Ratio y sus umbrales de seguridad.
    df columnas esperadas:
      - fecha (str o datetime): Fecha de la medición.
      - vmp (float): Valor de VMP.
      - ratio (float): Valor del Ratio (e.g., VMP/MMC28).
      - threshold_low (float): Umbral inferior de seguridad para el Ratio.
      - threshold_high (float): Umbral superior de seguridad para el Ratio.
    """
    if df.empty:
        fig = go.Figure()
        fig.update_layout(**_DARK_LAYOUT)
        fig.update_layout(title=f"Sin datos para {nombre_atleta}")
        return fig

    # Validar y limpiar NaN/NaT antes de procesar
    df = df.dropna(subset=['fecha', 'vmp', 'ratio', 'threshold_low', 'threshold_high']).copy()
    if df.empty:
        fig = go.Figure()
        fig.update_layout(**_DARK_LAYOUT)
        fig.update_layout(title=f"Sin datos válidos para {nombre_atleta}")
        return fig

    # Asegurar que las fechas estén en formato correcto para Plotly
    df['fecha'] = pd.to_datetime(df['fecha']).dt.strftime('%Y-%m-%d')

    fig = go.Figure()

    # Trace para VMP
    fig.add_trace(go.Scatter(
        x=df["fecha"], y=df["vmp"], mode="lines+markers",
        name="VMP CMJ (m/s)",
        line=dict(color=COLORS["vmp_line"], width=2),
        marker=dict(color=COLORS["vmp_line"], size=5),
        yaxis="y1" # Asignar al eje Y primario
    ))

    # Trace para Ratio
    fig.add_trace(go.Scatter(
        x=df["fecha"], y=df["ratio"], mode="lines+markers",
        name="Ratio (VMP/MMC28)",
        line=dict(color=COLORS["acute_line"], width=2, dash="dash"), # Usando un color y estilo diferente
        marker=dict(color=COLORS["acute_line"], size=5),
        yaxis="y2" # Asignar al eje Y secundario
    ))

    # Trace para umbral inferior del Ratio
    fig.add_trace(go.Scatter(
        x=df["fecha"], y=df["threshold_low"], mode="lines",
        name="Umbral Ratio Inferior",
        line=dict(color=COLORS["alert_dash"], width=1, dash="dash"),
        fill='tonexty', # Rellenar hasta el siguiente trace (umbral superior)
        fillcolor='rgba(249,115,22,0.08)', # Color de relleno para el área de alerta del ratio
        yaxis="y2"
    ))

    # Trace para umbral superior del Ratio
    fig.add_trace(go.Scatter(
        x=df["fecha"], y=df["threshold_high"], mode="lines",
        name="Umbral Ratio Superior",
        line=dict(color=COLORS["critical_dash"], width=1, dash="dash"),
        fill='tonexty', # Rellenar hasta el trace anterior (umbral inferior) o hasta cero si no hay
        fillcolor='rgba(239,68,68,0.08)', # Color de relleno para el área de alerta del ratio
        yaxis="y2"
    ))

    fig.update_layout(**_DARK_LAYOUT)
    fig.update_layout(
        title=dict(
            text=f"VMP CMJ y Ratio — {nombre_atleta}",
            font=dict(size=13, color=COLORS["text_primary"]), x=0.01,
        ),
        margin=dict(l=40, r=20, t=52, b=32),
        # Configuración de ejes Y
        yaxis=dict(
            title=dict(
                text="VMP CMJ (m/s)",
                font=dict(color=COLORS["vmp_line"])
            ),
            tickfont=dict(color=COLORS["vmp_line"]),
            color=COLORS["vmp_line"],
            rangemode='tozero' # Asegura que el eje Y empiece en 0
        ),
        yaxis2=dict(
            title=dict(
                text="Ratio (VMP/MMC28)",
                font=dict(color=COLORS["acute_line"])
            ),
            tickfont=dict(color=COLORS["acute_line"]),
            color=COLORS["acute_line"],
            overlaying="y", # Superponer sobre el eje y1
            side="right", # Colocar en el lado derecho
            rangemode='tozero'
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig
