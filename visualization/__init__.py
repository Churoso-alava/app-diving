"""
   visualization package
   Módulo de temas, gráficos y componentes UI para NMF-Optimizer
   """

   from .themes import (
       COLORS,
       STATUS_COLOR,
       STATUS_EMOJI,
       get_global_css,
   )
   from .charts import (
       fig_vmp_tendencia,
       fig_semaforo_barras,
       fig_semaforo_historico,
       fig_membership_fuzzy,
   )
   from .components import (
       render_kpi_row,
       render_athlete_bars,
       render_athlete_profile,
   )

   __all__ = [
       # themes
       "COLORS",
       "STATUS_COLOR",
       "STATUS_EMOJI",
       "get_global_css",
       # charts
       "fig_vmp_tendencia",
       "fig_semaforo_barras",
       "fig_semaforo_historico",
       "fig_membership_fuzzy",
       # components
       "render_kpi_row",
       "render_athlete_bars",
       "render_athlete_profile",
   ]
