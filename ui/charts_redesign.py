import logging
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.signal import savgol_filter
from ui.themes import COLORS, STATUS_COLOR

log = logging.getLogger(__name__)

_DARK_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor=COLORS["card_bg"],
    font=dict(family="Inter, sans-serif", color=COLORS["text_primary"]),
    margin=dict(l=12, r=12, t=40, b=12),
    xaxis=dict(gridcolor=COLORS["card_border"], zeroline=False),
    yaxis=dict(gridcolor=COLORS["card_border"], zeroline=False),
)


def _apply_savgol(df: pd.DataFrame, col: str, window: int = 5, polyorder: int = 2) -> pd.Series:
    """Aplica filtro Savitzky-Golay. Ajusta ventana si la serie es corta."""
    # Data cleaning: fill NaNs and replace Infs with 0
    data = df[col].fillna(0).replace([float('inf'), float('-inf')], 0)
    
    actual_window = min(window, len(df))
    if actual_window % 2 == 0:
        actual_window -= 1
    if actual_window < 3:
        return data
    return pd.Series(
        savgol_filter(data, window_length=actual_window, polyorder=polyorder),
        index=df.index,
    )


def fig_vmp_tendencia_redesign(df: pd.DataFrame, nombre_atleta: str, mmc28: float) -> go.Figure:
    df = df.copy()
    df["fecha"] = pd.to_datetime(df["fecha"])
    df["vmp_smooth"] = _apply_savgol(df, "vmp_hoy")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["fecha"], y=df["vmp_smooth"], mode="lines+markers", name="VMP (Suavizado)"))

    # Umbrales de seguridad (15% y 25% de caída sobre mmc28)
    fig.add_hline(y=mmc28 * 0.85, line_dash="dash", line_color="#ca8a04", annotation_text="Alerta")
    fig.add_hline(y=mmc28 * 0.75, line_dash="dash", line_color="#dc2626", annotation_text="Crítico")

    fig.update_layout(**_DARK_LAYOUT, title=f"VMP Tendencia — {nombre_atleta}")
    return fig


def fig_fatiga_tendencia(df: pd.DataFrame, titulo: str) -> go.Figure:
    df = df.copy()
    df["fecha"] = pd.to_datetime(df["fecha"])
    df["fatiga_smooth"] = _apply_savgol(df, "fatiga")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["fecha"], y=df["fatiga_smooth"], mode="lines+markers", name="Fatiga (Suavizado)"))

    fig.add_hline(y=25, line_dash="dash", line_color="#dc2626", annotation_text="Crítico")
    fig.add_hline(y=50, line_dash="dash", line_color="#ea580c", annotation_text="Fatiga")
    fig.add_hline(y=75, line_dash="dash", line_color="#ca8a04", annotation_text="Alerta")

    fig.update_layout(**_DARK_LAYOUT, title=titulo)
    return fig


_WELLNESS_METRICS = {
    "sueno":  {"label": "Sueño",  "color": "#38bdf8"},
    "fatiga": {"label": "Fatiga", "color": "#f87171"},
    "estres": {"label": "Estrés", "color": "#fb923c"},
    "dolor":  {"label": "Dolor",  "color": "#c084fc"},
    "humor":  {"label": "Humor",  "color": "#4ade80"},
}


def fig_wellness_tendencia(df: pd.DataFrame, nombre_atleta: str) -> go.Figure:
    df = df.copy()
    df["fecha"] = pd.to_datetime(df["fecha"])
    df = df.sort_values("fecha")

    fig = go.Figure()

    # Trazas individuales por sub-métrica
    for col, meta in _WELLNESS_METRICS.items():
        if col in df.columns:
            fig.add_trace(go.Scatter(
                x=df["fecha"],
                y=df[col],
                mode="lines+markers",
                name=meta["label"],
                line=dict(color=meta["color"], width=1.5),
                marker=dict(size=5),
                opacity=0.75,
            ))

    # Traza global: promedio simple de las sub-métricas disponibles
    available = [c for c in _WELLNESS_METRICS if c in df.columns]
    if available:
        df["_wellness_global"] = df[available].mean(axis=1)
        df["_wellness_smooth"] = _apply_savgol(df, "_wellness_global")
        fig.add_trace(go.Scatter(
            x=df["fecha"],
            y=df["_wellness_smooth"],
            mode="lines",
            name="Wellness Global",
            line=dict(color="#ffffff", width=3),
            opacity=1.0,
        ))

    # Construir layout evitando keyword duplicado 'yaxis'
    base_layout = {k: v for k, v in _DARK_LAYOUT.items() if k != "yaxis"}
    yaxis_base = _DARK_LAYOUT.get("yaxis", {})

    fig.update_layout(
        **base_layout,
        title=f"Wellness — {nombre_atleta}",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        yaxis=dict(**yaxis_base, range=[0, 8]),
    )
    return fig


def fig_carga_entrenamiento(df: pd.DataFrame, nombre_atleta: str) -> go.Figure:
    """
    Gráfico de barras para carga diaria (RPE * Duración) con curva de tendencia suavizada.
    """
    df = df.copy()
    df["fecha"] = pd.to_datetime(df["fecha"])
    # Asegurar que la carga está calculada
    if "carga_interna" not in df.columns:
        df["carga_interna"] = df["carga_subjetiva"] * df["duracion_min"]
    
    df = df.sort_values("fecha")
    df["carga_smooth"] = _apply_savgol(df, "carga_interna")

    fig = go.Figure()
    
    # Barras de carga diaria
    fig.add_trace(go.Bar(
        x=df["fecha"], 
        y=df["carga_interna"], 
        name="Carga Total (UA)",
        marker_color=COLORS["accent"],
        opacity=0.6
    ))
    
    # Curva de tendencia
    fig.add_trace(go.Scatter(
        x=df["fecha"], 
        y=df["carga_smooth"], 
        mode="lines", 
        name="Tendencia (Suavizado)",
        line=dict(color="#ffffff", width=2)
    ))

    fig.update_layout(**_DARK_LAYOUT, title=f"Carga de Entrenamiento — {nombre_atleta}")
    return fig
