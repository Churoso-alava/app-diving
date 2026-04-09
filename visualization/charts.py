"""
visualization/charts.py
Funciones Plotly para NMF-Optimizer (tema oscuro)
Reemplaza todos los st.pyplot(fig) del app.py actual.
"""

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from visualization.themes import COLORS, STATUS_COLOR


# ── Constantes de layout compartido ──────────────────────────────────────────
_DARK_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor=COLORS["card_bg"],
    font=dict(family="Inter, sans-serif", color=COLORS["text_primary"]),
    margin=dict(l=12, r=12, t=40, b=12),
    xaxis=dict(
        gridcolor=COLORS["card_border"],
        zeroline=False,
        showline=False,
        tickfont=dict(size=10, color=COLORS["text_muted"]),
    ),
    yaxis=dict(
        gridcolor=COLORS["card_border"],
        zeroline=False,
        showline=False,
        tickfont=dict(size=10, color=COLORS["text_muted"]),
    ),
    legend=dict(
        bgcolor="rgba(0,0,0,0)",
        bordercolor="rgba(0,0,0,0)",
        font=dict(size=10, color=COLORS["text_muted"]),
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="left",
        x=0,
    ),
    hoverlabel=dict(
        bgcolor=COLORS["card_bg"],
        bordercolor=COLORS["card_border"],
        font=dict(size=12, color=COLORS["text_primary"]),
    ),
)


def fig_vmp_tendencia(df: pd.DataFrame, nombre_atleta: str, delta_pct: float) -> go.Figure:
    """
    Gráfico de evolución VMP del CMJ para un atleta.
    
    df debe tener columnas: fecha (str), vmp_hoy (float), mma7 (float), mmc28 (float)
    delta_pct: porcentaje delta vs MMC28 para el título.
    """
    delta_str = f"+{delta_pct:.1f}%" if delta_pct >= 0 else f"{delta_pct:.1f}%"
    
    fig = go.Figure()

    # VMP fase propulsiva
    fig.add_trace(go.Scatter(
        x=df["fecha"], y=df["vmp_hoy"],
        mode="lines+markers",
        name="VMP Fase Propulsiva CMJ (m/s)",
        line=dict(color=COLORS["vmp_line"], width=2),
        marker=dict(color=COLORS["vmp_line"], size=5),
    ))

    # Fatiga aguda MMA7
    fig.add_trace(go.Scatter(
        x=df["fecha"], y=df["mma7"],
        mode="lines",
        name="Fatiga Aguda — MMA7 (7 días)",
        line=dict(color=COLORS["acute_line"], width=2),
    ))

    # Baseline crónico MMC28
    fig.add_trace(go.Scatter(
        x=df["fecha"], y=df["mmc28"],
        mode="lines",
        name="Baseline Crónico — MMC28 (28 días)",
        line=dict(color=COLORS["chronic_line"], width=1.5, dash="dot"),
    ))

    # Umbral alerta (~15% caída sobre MMC28)
    fig.add_trace(go.Scatter(
        x=df["fecha"], y=df["mmc28"] * 0.85,
        mode="lines",
        name="Umbral Alerta (~15% caída sobre MMC28)",
        line=dict(color=COLORS["alert_dash"], width=1, dash="dash"),
        showlegend=True,
    ))

    # Umbral crítico (~25% caída sobre MMC28)
    fig.add_trace(go.Scatter(
        x=df["fecha"], y=df["mmc28"] * 0.75,
        mode="lines",
        name="Umbral Crítico (~25% caída sobre MMC28)",
        line=dict(color=COLORS["critical_dash"], width=1, dash="dash"),
        showlegend=True,
    ))

    layout = dict(
        title=dict(
            text=f"Evolución VMP del CMJ — {nombre_atleta}  [Δ vs MMC28: {delta_str}]",
            font=dict(size=13, color=COLORS["text_primary"]),
            x=0.01,
        ),
        **_DARK_LAYOUT,
    )
    layout["margin"] = dict(l=40, r=20, t=52, b=32)
    fig.update_layout(**layout)
    return fig


def fig_semaforo_barras(df_estado: pd.DataFrame) -> go.Figure:
    """
    Gráfico de barras horizontales con estado de cada atleta.
    
    df_estado columnas: nombre (str), score (float 0-100), estado (str), fecha (str)
    """
    df = df_estado.sort_values("score", ascending=True).copy()
    colores = [STATUS_COLOR.get(e, COLORS["text_muted"]) for e in df["estado"]]

    fig = go.Figure(go.Bar(
        x=df["score"],
        y=df["nombre"],
        orientation="h",
        marker_color=colores,
        text=[f"{s:.0f}%" for s in df["score"]],
        textposition="outside",
        textfont=dict(size=12, color=COLORS["text_primary"]),
        customdata=df[["estado", "fecha"]].values,
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Score: %{x:.0f}%<br>"
            "Estado: %{customdata[0]}<br>"
            "Última sesión: %{customdata[1]}<extra></extra>"
        ),
    ))

    layout = dict(
        title=dict(
            text="Estado Actual — Todos los Atletas",
            font=dict(size=13, color=COLORS["text_primary"]),
            x=0.01,
        ),
        xaxis=dict(range=[0, 115], showgrid=False, showticklabels=False),
        **_DARK_LAYOUT,
    )
    layout["margin"] = dict(l=110, r=60, t=48, b=12)
    fig.update_layout(**layout)
    return fig


def fig_semaforo_historico(df_hist: pd.DataFrame) -> go.Figure:
    """
    NUEVO — Línea histórica del semáforo (no existía).
    Muestra la evolución del score de readiness para cada atleta en el tiempo.
    
    df_hist columnas: fecha (str), nombre (str), score (float), estado (str)
    """
    fig = go.Figure()

    for nombre, grupo in df_hist.groupby("nombre"):
        grupo = grupo.sort_values("fecha")
        colores_puntos = [STATUS_COLOR.get(e, COLORS["text_muted"]) for e in grupo["estado"]]

        # Línea de tendencia
        fig.add_trace(go.Scatter(
            x=grupo["fecha"],
            y=grupo["score"],
            mode="lines+markers",
            name=nombre,
            line=dict(width=1.5),
            marker=dict(
                color=colores_puntos,
                size=7,
                line=dict(width=1, color=COLORS["card_bg"]),
            ),
            hovertemplate=(
                f"<b>{nombre}</b><br>"
                "Fecha: %{x}<br>"
                "Score: %{y:.0f}%<br>"
                "<extra></extra>"
            ),
        ))

    # Bandas de referencia
    for nivel, y_val, color in [
        ("Crítico", 30,  COLORS["critical"]),
        ("Alerta",  55,  COLORS["alert_dash"]),
        ("Óptimo",  75,  COLORS["optimal"]),
    ]:
        fig.add_hline(
            y=y_val,
            line_dash="dot",
            line_color=color,
            opacity=0.35,
            annotation_text=nivel,
            annotation_font_color=color,
            annotation_font_size=9,
            annotation_position="right",
        )

    layout = dict(
        title=dict(
            text="Línea Histórica de Semáforo — Evolución del Readiness",
            font=dict(size=13, color=COLORS["text_primary"]),
            x=0.01,
        ),
        yaxis=dict(range=[0, 105], title="Score de Readiness (%)"),
        **_DARK_LAYOUT,
    )
    layout["margin"] = dict(l=50, r=80, t=52, b=32)
    fig.update_layout(**layout)
    return fig


def fig_membership_fuzzy(x_vals, membership_vals: dict) -> go.Figure:
    """
    Curvas de membresía difusa (reemplaza matplotlib).
    membership_vals: {"Óptimo": [...], "Alerta": [...], "Fatiga": [...], "Crítico": [...]}
    """
    colores_memb = {
        "Óptimo":  COLORS["optimal"],
        "Alerta":  COLORS["early_alert"],
        "Fatiga":  COLORS["accum_fatigue"],
        "Crítico": COLORS["critical"],
    }
    fig = go.Figure()
    for nombre, y_vals in membership_vals.items():
        fig.add_trace(go.Scatter(
            x=x_vals, y=y_vals,
            mode="lines",
            name=nombre,
            line=dict(color=colores_memb.get(nombre, COLORS["accent"]), width=2),
            fill="tozeroy",
            fillcolor=colores_memb.get(nombre, COLORS["accent"]).replace(")", ",0.08)").replace("rgb(", "rgba("),
        ))

    layout = dict(
        title=dict(
            text="Funciones de Membresía — Motor Difuso",
            font=dict(size=13),
            x=0.01,
        ),
        xaxis=dict(title="Índice de Fatiga"),
        yaxis=dict(title="Grado de Membresía", range=[0, 1.05]),
        **_DARK_LAYOUT,
    )
    layout["margin"] = dict(l=50, r=20, t=48, b=40)
    fig.update_layout(**layout)
    return fig
