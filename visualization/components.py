"""
visualization/components.py
Componentes HTML/CSS reutilizables para Streamlit (st.markdown unsafe_allow_html=True)

Cambios v4.2:
  P3.2 — render_athlete_bars: ordenar score ASC (peor primero) antes de renderizar.
  P1   — render_athlete_profile: rediseño en 3 zonas (Header / Métricas primarias / Secundarias).
           Zona B expone ACWR, Δ% y DQI con umbrales de color semafórico.
           Zona C en expander colapsado para el analista.
           Elimina badges "JUGADOR/AVAILABLE" sin valor operativo.
"""

import streamlit as st
from visualization.themes import COLORS, STATUS_COLOR, STATUS_EMOJI


# ── Umbrales Zona B (según documento RSI/VBT + plan de acción) ────────────────
def _color_acwr(v: float) -> str:
    """
    Clasifica ACWR según umbrales del documento RSI/VBT + decisión de diseño.

    Definidos en doc: Verde 0.8–1.1 | Rojo >1.3 o <0.7
    Zona 0.70–0.80: no especificada en doc.
    DECISIÓN: early_alert (amarillo) — intervención conservadora.
    Ref: Weakley 2019.
    """
    if 0.8 <= v <= 1.1:
        return COLORS["optimal"]
    if v < 0.7 or v > 1.3:          # rojo: <0.7 o >1.3 (doc RSI/VBT)
        return COLORS["critical"]
    return COLORS["early_alert"]    # amarillo: 0.7–0.8 o 1.1–1.3


def _color_delta(v: float) -> str:
    if abs(v) < 5.0:
        return COLORS["optimal"]
    if abs(v) < 15.0:
        return COLORS["early_alert"]
    return COLORS["critical"]


def _color_dqi(v: float) -> str:
    if v >= 0.8:
        return COLORS["optimal"]
    if v >= 0.5:
        return COLORS["early_alert"]
    return COLORS["critical"]


def _label_acwr(v: float) -> str:
    if 0.8 <= v <= 1.1:
        return "ÓPTIMO"
    if v <= 1.3:
        return "VIGILANCIA"
    return "CRÍTICO"


def _label_delta(v: float) -> str:
    if abs(v) < 5.0:
        return "ÓPTIMO"
    if abs(v) < 15.0:
        return "VIGILANCIA"
    return "ALARMA"


def _label_dqi(v: float) -> str:
    if v >= 0.8:
        return "ALTA"
    if v >= 0.5:
        return "MEDIA"
    return "BAJA"


# ── KPI Row ───────────────────────────────────────────────────────────────────

def render_kpi_row(total: int, criticos: int, fatiga_acum: int,
                   alerta_temp: int, optimos: int) -> None:
    """Renderiza la fila de 5 KPI cards."""
    def _badge(text: str, color: str) -> str:
        return (
            f'<span class="kpi-badge" '
            f'style="background:{color}22; color:{color};">{text}</span>'
        )

    cards = [
        {"label": "TOTAL ATHLETES", "value": str(total), "extra": ""},
        {
            "label": "CRITICAL",
            "value": str(criticos),
            "extra": _badge("✓ OK", COLORS["optimal"]) if criticos == 0 else _badge("⚠ VER", COLORS["critical"]),
        },
        {"label": "ACCUM. FATIGUE", "value": str(fatiga_acum), "extra": _badge("●", COLORS["accum_fatigue"])},
        {"label": "EARLY ALERT",    "value": str(alerta_temp), "extra": _badge("●", COLORS["early_alert"])},
        {"label": "OPTIMAL",        "value": str(optimos),     "extra": _badge("●", COLORS["optimal"])},
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


# ── Athlete progress bars (P3.2 — ordenar ASC) ───────────────────────────────

def render_athlete_bars(atletas: list[dict]) -> None:
    """
    Renderiza grid de 2 columnas con barras de progreso por atleta.
    P3.2: ordena internamente por score ASC → los más fatigados aparecen primero.

    atletas: lista de dicts con keys: nombre, score (0-100), estado, fecha
    """
    # P3.2 — peor primero (score más bajo = más fatigado)
    atletas_ord = sorted(atletas, key=lambda a: a["score"])

    cols = st.columns(2)
    for i, ath in enumerate(atletas_ord):
        color = STATUS_COLOR.get(ath["estado"], COLORS["text_muted"])
        score = min(max(ath["score"], 0), 100)
        with cols[i % 2]:
            st.markdown(
                f"""
                <div class="athlete-card">
                  <div class="athlete-name">
                    {ath['nombre']}
                    <span class="athlete-status-label" style="color:{color};">
                      {score:.0f}% — {ath['estado']}
                    </span>
                  </div>
                  <!-- fondo gris = referencia al 100% -->
                  <div class="progress-track">
                    <div class="progress-fill"
                         style="width:{score}%; background:{color};"></div>
                  </div>
                  <!-- marcadores de umbral sobre la barra -->
                  <div style="position:relative;height:8px;margin-top:-2px;">
                    <div style="position:absolute;left:50%;width:1px;height:8px;
                                background:{COLORS['accum_fatigue']};opacity:0.6;"></div>
                    <div style="position:absolute;left:75%;width:1px;height:8px;
                                background:{COLORS['optimal']};opacity:0.6;"></div>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )


# ── Athlete detail profile (P1 — 3 zonas) ────────────────────────────────────

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
    Panel de perfil de atleta rediseñado en 3 zonas (P1).

    ZONA A — Header: nombre + semáforo + índice + recomendación. Sin badges posición/available.
    ZONA B — Métricas primarias: ACWR, Δ% vs MMC28, DQI con color semafórico.
    ZONA C — Métricas secundarias en expander colapsado (z_meso, β₇, β₂₈, sesiones_consec).

    metricas: dict con keys: acwr, delta_pct, z_meso, beta7, beta28, sesiones_consec, dqi
    Cada valor puede ser un dict {"valor": x, "estado": "..."} o el valor directo.
    """
    color_estado = STATUS_COLOR.get(estado, COLORS["text_muted"])
    emoji        = STATUS_EMOJI.get(estado, "⚪")

    def _get(key: str, default=0):
        entry = metricas.get(key, default)
        if isinstance(entry, dict):
            return entry.get("valor", default)
        return entry

    acwr_v      = _get("acwr",            1.0)
    delta_v     = _get("delta_pct",       0.0)
    dqi_v       = _get("dqi",             0.0)
    zmeso_v     = _get("z_meso",          0.0)
    beta7_v     = _get("beta7",           0.0)
    beta28_v    = _get("beta28",          0.0)
    sesiones_v  = _get("sesiones_consec", 0)
    dqi_label   = metricas.get("dqi", {}).get("estado", "") if isinstance(metricas.get("dqi"), dict) else ""

    # ── ZONA A — Header ───────────────────────────────────────────────────────
    st.markdown(
        f"""
        <div class="profile-header">
          <div style="font-size:9px;letter-spacing:1px;color:{COLORS['text_muted']};margin-bottom:6px;">
            ANÁLISIS INDIVIDUAL · {ultima_sesion}
          </div>
          <div style="display:flex;align-items:baseline;gap:14px;margin-bottom:10px;">
            <span style="font-size:22px;font-weight:700;color:{COLORS['text_primary']};">{nombre}</span>
            <span style="font-size:11px;color:{COLORS['text_muted']};">{posicion.upper()}</span>
          </div>
          <div style="display:flex;align-items:center;gap:16px;margin-bottom:12px;">
            <div style="font-size:52px;font-weight:900;line-height:1;color:{color_estado};">
              {indice_fatiga:.0f}
            </div>
            <div>
              <div style="font-size:13px;font-weight:700;color:{color_estado};">
                {emoji} {estado}
              </div>
              <div style="font-size:10px;letter-spacing:1px;color:{COLORS['text_muted']};margin-top:2px;">
                ÍNDICE DE READINESS
              </div>
            </div>
          </div>
          <div style="background:rgba(0,0,0,0.3);border-left:3px solid {COLORS['accent']};
                      border-radius:4px;padding:10px 14px;font-size:13px;color:{COLORS['text_primary']};">
            🎯 {recomendacion}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── ZONA B — Métricas primarias con color semafórico ─────────────────────
    st.markdown(
        f'<div class="section-title">📊 Métricas Clave del Modelo</div>',
        unsafe_allow_html=True,
    )

    c_acwr  = _color_acwr(acwr_v)
    c_delta = _color_delta(delta_v)
    c_dqi   = _color_dqi(dqi_v)
    l_acwr  = _label_acwr(acwr_v)
    l_delta = _label_delta(delta_v)
    l_dqi   = dqi_label if dqi_label else _label_dqi(dqi_v)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            f"""
            <div class="metric-primary" style="border-left:3px solid {c_acwr};">
              <div class="metric-mini-label">ACWR</div>
              <div class="metric-primary-value" style="color:{c_acwr};">{acwr_v:.3f}</div>
              <div class="metric-mini-badge-block">
                <span class="metric-mini-badge" style="background:{c_acwr}22;color:{c_acwr};">
                  {l_acwr}
                </span>
              </div>
              <div style="font-size:9px;color:{COLORS['text_muted']};margin-top:4px;">
                Verde 0.8–1.1 · Amarillo 1.1–1.3 · Rojo &gt;1.3
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        delta_fmt = f"{delta_v:+.1f}%"
        st.markdown(
            f"""
            <div class="metric-primary" style="border-left:3px solid {c_delta};">
              <div class="metric-mini-label">Δ% vs MMC28</div>
              <div class="metric-primary-value" style="color:{c_delta};">{delta_fmt}</div>
              <div class="metric-mini-badge-block">
                <span class="metric-mini-badge" style="background:{c_delta}22;color:{c_delta};">
                  {l_delta}
                </span>
              </div>
              <div style="font-size:9px;color:{COLORS['text_muted']};margin-top:4px;">
                Verde &lt;5% · Amarillo 5–15% · Rojo &gt;15%
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            f"""
            <div class="metric-primary" style="border-left:3px solid {c_dqi};">
              <div class="metric-mini-label">DQI — Confianza del Dato</div>
              <div class="metric-primary-value" style="color:{c_dqi};">{dqi_v:.2f}</div>
              <div class="metric-mini-badge-block">
                <span class="metric-mini-badge" style="background:{c_dqi}22;color:{c_dqi};">
                  {l_dqi}
                </span>
              </div>
              <div style="font-size:9px;color:{COLORS['text_muted']};margin-top:4px;">
                Verde ≥0.80 · Amarillo 0.50–0.80 · Rojo &lt;0.50
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ── ZONA C — Variables secundarias (expander para el analista) ────────────
    with st.expander("🔬 Variables secundarias del modelo", expanded=False):
        st.markdown(
            f'<div class="section-title">Variables Mamdani — Contexto Avanzado</div>',
            unsafe_allow_html=True,
        )
        campos_sec = [
            ("Z-Score Mesociclo", f"{zmeso_v:+.2f}",  "Posición relativa en el mesociclo actual"),
            ("β₇ Tendencia semanal", f"{beta7_v:+.4f}", "Pendiente regresión últimas 7 sesiones"),
            ("β₂₈ Tendencia mensual", f"{beta28_v:+.4f}", "Pendiente regresión últimas 28 sesiones"),
            ("Sesiones desc. consec.", str(int(sesiones_v)), "Sesiones VMP consecutivamente decrecientes"),
        ]
        sec_cols = st.columns(4)
        for col, (label, valor, tooltip) in zip(sec_cols, campos_sec):
            with col:
                st.markdown(
                    f"""
                    <div class="metric-mini" title="{tooltip}">
                      <div class="metric-mini-label">{label}</div>
                      <div class="metric-mini-value">{valor}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
