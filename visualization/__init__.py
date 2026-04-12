"""
visualization package - NMF-Optimizer
"""

from visualization.themes import (
    COLORS,
    STATUS_COLOR,
    STATUS_EMOJI,
    get_global_css,
)

from visualization.charts import (
    fig_vmp_tendencia,
    fig_semaforo_barras,
    fig_semaforo_historico,
    fig_historial_barras_atleta,
    fig_membership_fuzzy,
)

from visualization.components import (
    render_kpi_row,
    render_athlete_bars,
    render_athlete_profile,
)

__all__ = [
    "COLORS",
    "STATUS_COLOR",
    "STATUS_EMOJI",
    "get_global_css",
    "fig_vmp_tendencia",
    "fig_semaforo_barras",
    "fig_semaforo_historico",
    "fig_historial_barras_atleta",
    "fig_membership_fuzzy",
    "render_kpi_row",
    "render_athlete_bars",
    "render_athlete_profile",
]
