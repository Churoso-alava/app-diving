"""
visualization/components.py
Componentes HTML/CSS reutilizables para Streamlit (st.markdown unsafe_allow_html=True)
"""

import streamlit as st
from visualization.themes import COLORS, STATUS_COLOR, STATUS_EMOJI


# ── KPI Row ───────────────────────────────────────────────────────────────────

def render_kpi_row(total: int, criticos: int, fatiga_acum: int,
                   alerta_temp: int, optimos: int) -> None:
    """
    Renderiza la fila de 5 KPI cards (Slide 1 del mockup).
    Sustituye los st.metric actuales.
    """
    def _badge(text: str, color: str) -> str:
        return (
            f'<span class="kpi-badge" '
            f'style="background:{color}22; color:{color};">{text}</span>'
        )

    cards = [
        {
            "label": "TOTAL ATHLETES",
            "value": str(total),
            "extra": "",
        },
        {
            "label": "CRITICAL",
            "value": str(criticos),
            "extra": _badge("✓ OK", COLORS["optimal"]) if criticos == 0 else _badge("⚠ VER", COLORS["critical"]),
        },
        {
            "label": "ACCUM. FATIGUE",
            "value": str(fatiga_acum),
            "extra": _badge("●", COLORS["accum_fatigue"]),
        },
        {
            "label": "EARLY ALERT",
            "value": str(alerta_temp),
            "extra": _badge("●", COLORS["early_alert"]),
        },
        {
            "label": "OPTIMAL",
            "value": str(optimos),
            "extra": _badge("●", COLORS["optimal"]),
        },
    ]

    cols = st.columns(5)
    for col, card in zip(cols, cards):
        with col:
            st.markdown(
                f"""
                <div class="kpi-card">
                  <div class="kpi-label">{card['label']}</div>
                  <div class="kpi-value">{card['value']}{card['extra']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


# ── Athlete progress bars (2-column grid) ────────────────────────────────────

def render_athlete_bars(atletas: list[dict]) -> None:
    """
    Renderiza grid de 2 columnas con barras de progreso por atleta (Slide 2).
    
    atletas: lista de dicts con keys: nombre, score (0-100), estado, fecha
    """
    cols = st.columns(2)
    for i, ath in enumerate(atletas):
        color = STATUS_COLOR.get(ath["estado"], COLORS["text_muted"])
        score = min(max(ath["score"], 0), 100)
        with cols[i % 2]:
            st.markdown(
                f"""
                <div class="athlete-card">
                  <div class="athlete-name">
                    {ath['nombre']}
                    <span class="athlete-status-label" style="color:{color};">
                      {score:.0f}% {ath['estado']}
                    </span>
                  </div>
                  <div class="progress-track">
                    <div class="progress-fill"
                         style="width:{score}%; background:{color};"></div>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )


# ── Athlete detail profile ────────────────────────────────────────────────────

def render_athlete_profile(
    nombre: str,
    posicion: str,
    disponible: bool,
    indice_fatiga: float,
    estado: str,
    recomendacion: str,
    ultima_sesion: str,
    metricas: dict,
) -> None:
    """
    Renderiza el panel completo de perfil de atleta (Slide 5).
    
    metricas: dict con keys: acwr, delta_pct, z_meso, beta7, beta28, sesiones_consec
    Cada valor puede ser un dict {"valor": x, "estado": "ÓPTIMO"} o solo el valor.
    """
    color_estado = STATUS_COLOR.get(estado, COLORS["text_muted"])
    avail_color  = COLORS["optimal"] if disponible else COLORS["critical"]
    avail_text   = "AVAILABLE" if disponible else "LIMITED"
    emoji        = STATUS_EMOJI.get(estado, "⚪")

    # Header
    st.markdown(
        f"""
        <div class="profile-header">
          <div style="font-size:9px;letter-spacing:1px;color:{COLORS['text_muted']};margin-bottom:4px;">
            ATHLETE PROFILE
          </div>
          <div style="display:flex;align-items:center;gap:12px;margin-bottom:14px;">
            <span style="font-size:20px;font-weight:700;color:{COLORS['text_primary']};">{nombre}</span>
            <span style="background:{COLORS['card_border']};color:{COLORS['text_muted']};
                         font-size:10px;font-weight:600;padding:2px 10px;border-radius:20px;">
              {posicion.upper()}
            </span>
            <span style="background:{avail_color}22;color:{avail_color};
                         font-size:10px;font-weight:600;padding:2px 10px;border-radius:20px;">
              {avail_text}
            </span>
          </div>
          <div class="fatigue-index-big" style="color:{color_estado};">{indice_fatiga:.0f}</div>
          <div class="fatigue-index-label">{emoji} {estado}</div>
          <div class="recommendation-box">{recomendacion}</div>
          <div style="font-size:10px;color:{COLORS['text_muted']};margin-top:8px;">
            Última sesión: {ultima_sesion}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Variables del modelo — grid 2x3
    st.markdown(
        f'<div class="section-title">Variables del Modelo</div>',
        unsafe_allow_html=True,
    )

    campos = [
        ("ACWR",                "acwr",           ".3f"),
        ("Δ% vs MMC28",         "delta_pct",      "+.1f%"),
        ("Z-Score Mesociclo",   "z_meso",         ".2f"),
        ("β₇ (Tend. semanal)",  "beta7",          ".4f"),
        ("β₂₈ (Tend. mens.)",   "beta28",         ".4f"),
        ("Sesiones Consecutivas","sesiones_consec","d"),
    ]

    def _fmt(val, fmt):
        try:
            return format(val, fmt) if not fmt.endswith("%") else f"{val:+.1f}%"
        except Exception:
            return str(val)

    rows = [campos[:3], campos[3:]]
    for row in rows:
        cols = st.columns(3)
        for col, (label, key, fmt) in zip(cols, row):
            with col:
                entry   = metricas.get(key, {})
                valor   = entry.get("valor", entry) if isinstance(entry, dict) else entry
                est     = entry.get("estado", "") if isinstance(entry, dict) else ""
                color_m = STATUS_COLOR.get(est, COLORS["text_primary"])
                badge   = (
                    f'<span class="metric-mini-badge" '
                    f'style="background:{color_m}22;color:{color_m};">{est}</span>'
                    if est else ""
                )
                st.markdown(
                    f"""
                    <div class="metric-mini">
                      <div class="metric-mini-label">{label}</div>
                      <div class="metric-mini-value">
                        {_fmt(valor, fmt)}{badge}
                      </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
