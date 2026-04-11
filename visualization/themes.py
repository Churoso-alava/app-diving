"""
visualization/themes.py - NMF-Optimizer Dark Theme
"""

COLORS = {
    "bg_dark": "#0D1117",
    "card_bg": "#161B22",
    "card_border": "#30363D",
    "optimal": "#00C49A",
    "early_alert": "#9B59B6",
    "accum_fatigue": "#E67E22",
    "critical": "#E74C3C",
    "vmp_line": "#7B68EE",
    "acute_line": "#E67E22",
    "chronic_line": "#A8B2D8",
    "alert_dash": "#F5A623",
    "critical_dash": "#E74C3C",
    "text_primary": "#E6EDF3",
    "text_muted": "#8B949E",
    "accent": "#58A6FF",
}

STATUS_COLOR = {
    "ÓPTIMO": COLORS["optimal"],
    "ALERTA TEMPRANA": COLORS["early_alert"],
    "FATIGA ACUMULADA": COLORS["accum_fatigue"],
    "CRÍTICO": COLORS["critical"],
    "OPTIMAL": COLORS["optimal"],
    "EARLY ALERT": COLORS["early_alert"],
    "ACCUM. FATIGUE": COLORS["accum_fatigue"],
    "CRITICAL": COLORS["critical"],
}

STATUS_EMOJI = {
    "ÓPTIMO": "🟢",
    "ALERTA TEMPRANA": "🟣",
    "FATIGA ACUMULADA": "🟠",
    "CRÍTICO": "🔴",
}

def get_global_css() -> str:
    return f"""
<style>
[data-testid="stAppViewContainer"] {{
    background-color: {COLORS['bg_dark']};
}}
[data-testid="stSidebar"] {{
    background-color: {COLORS['card_bg']};
}}
h1, h2, h3, p, label {{
    color: {COLORS['text_primary']} !important;
}}
.kpi-card {{
    background: {COLORS['card_bg']};
    border: 1px solid {COLORS['card_border']};
    border-radius: 12px;
    padding: 18px 20px 14px 20px;
    min-height: 90px;
}}
.kpi-label {{
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 1.2px;
    text-transform: uppercase;
    color: {COLORS['text_muted']};
    margin-bottom: 6px;
}}
.kpi-value {{
    font-size: 36px;
    font-weight: 800;
    line-height: 1;
    color: {COLORS['text_primary']};
}}
.kpi-badge {{
    display: inline-block;
    font-size: 11px;
    font-weight: 600;
    padding: 2px 8px;
    border-radius: 20px;
    margin-left: 6px;
    vertical-align: middle;
}}
.athlete-card {{
    background: {COLORS['card_bg']};
    border: 1px solid {COLORS['card_border']};
    border-radius: 10px;
    padding: 14px 16px;
    margin-bottom: 8px;
}}
.athlete-name {{
    font-size: 14px;
    font-weight: 700;
    color: {COLORS['text_primary']};
    margin-bottom: 4px;
}}
.athlete-status-label {{
    font-size: 11px;
    font-weight: 600;
    float: right;
}}
.progress-track {{
    background: {COLORS['card_border']};
    border-radius: 4px;
    height: 6px;
    margin-top: 6px;
    overflow: hidden;
}}
.progress-fill {{
    height: 100%;
    border-radius: 4px;
    transition: width 0.4s ease;
}}
.profile-header {{
    background: {COLORS['card_bg']};
    border: 1px solid {COLORS['card_border']};
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 12px;
}}
.fatigue-index-big {{
    font-size: 56px;
    font-weight: 900;
    line-height: 1;
    margin-bottom: 2px;
}}
.fatigue-index-label {{
    font-size: 11px;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: {COLORS['text_muted']};
}}
.recommendation-box {{
    background: rgba(0,0,0,0.3);
    border-left: 3px solid {COLORS['accent']};
    border-radius: 4px;
    padding: 10px 14px;
    margin-top: 12px;
    font-size: 13px;
    color: {COLORS['text_primary']};
}}
/* P1 — Zona B: tarjeta de métrica primaria con borde de color semafórico */
.metric-primary {{
    background: {COLORS['card_bg']};
    border: 1px solid {COLORS['card_border']};
    border-radius: 8px;
    padding: 14px 16px;
    margin-bottom: 8px;
}}
.metric-primary-value {{
    font-size: 28px;
    font-weight: 800;
    line-height: 1.1;
    margin: 4px 0;
}}
.metric-mini-badge-block {{
    margin: 4px 0;
}}
/* Tarjeta secundaria */
.metric-mini {{
    background: {COLORS['card_bg']};
    border: 1px solid {COLORS['card_border']};
    border-radius: 8px;
    padding: 12px 14px;
    margin-bottom: 8px;
}}
.metric-mini-label {{
    font-size: 9px;
    letter-spacing: 1px;
    text-transform: uppercase;
    color: {COLORS['text_muted']};
    margin-bottom: 4px;
}}
.metric-mini-value {{
    font-size: 22px;
    font-weight: 700;
    color: {COLORS['text_primary']};
}}
.metric-mini-badge {{
    font-size: 9px;
    font-weight: 600;
    padding: 1px 6px;
    border-radius: 10px;
    margin-left: 4px;
}}
.section-title {{
    font-size: 12px;
    letter-spacing: 1.2px;
    text-transform: uppercase;
    color: {COLORS['text_muted']};
    margin: 18px 0 8px 0;
    font-weight: 600;
}}
</style>
"""
