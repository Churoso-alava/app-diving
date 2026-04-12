"""
tests/test_ui_v43.py
NMF-Optimizer v4.3 — UI/UX Tests
"""
from __future__ import annotations

import pandas as pd
import pytest
import numpy as np
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
# Task 4 — Carga Grupal DB functions
# ─────────────────────────────────────────────────────────────────────────────

class TestCargaGrupalDB:
    """Verifica la función insertar_carga_grupal_batch antes de tocar BD."""

    def _make_ejercicios_df(self) -> pd.DataFrame:
        return pd.DataFrame({
            "tipo_plataforma": ["trampolín", "plataforma"],
            "altura_salto":    [3.0, 5.0],
            "n_saltos":        [10, 5],
            "tipo_caida":      ["pie", "mano"],
        })

    def test_insertar_carga_grupal_batch_validates_platform(self):
        """Debe rechazar tipo_plataforma inválido."""
        import db
        df_bad = pd.DataFrame({
            "tipo_plataforma": ["invalid"],
            "altura_salto":    [3.0],
            "n_saltos":        [5],
            "tipo_caida":      ["pie"],
        })
        ok, errors = db.insertar_carga_grupal_batch(
            fecha="2026-04-11",
            df_ejercicios=df_bad,
            atletas=["Atleta1"],
            notas="",
        )
        assert not ok
        assert any("plataforma" in e.lower() or "inválido" in e.lower() for e in errors)

    def test_insertar_carga_grupal_batch_validates_caida(self):
        """Debe rechazar tipo_caida inválido."""
        import db
        df_bad = pd.DataFrame({
            "tipo_plataforma": ["trampolín"],
            "altura_salto":    [3.0],
            "n_saltos":        [5],
            "tipo_caida":      ["cabeza"],
        })
        ok, errors = db.insertar_carga_grupal_batch(
            fecha="2026-04-11",
            df_ejercicios=df_bad,
            atletas=["Atleta1"],
            notas="",
        )
        assert not ok
        assert any("caida" in e.lower() or "inválido" in e.lower() for e in errors)

    def test_insertar_carga_grupal_batch_requires_atletas(self):
        """Lista de atletas vacía debe rechazarse."""
        import db
        df_ok = self._make_ejercicios_df()
        ok, errors = db.insertar_carga_grupal_batch(
            fecha="2026-04-11",
            df_ejercicios=df_ok,
            atletas=[],
            notas="",
        )
        assert not ok
        assert errors


# ─────────────────────────────────────────────────────────────────────────────
# Task 4 — Wellness Masivo template
# ─────────────────────────────────────────────────────────────────────────────

class TestWellnessMasivoUI:
    def test_wellness_masivo_template_columns(self):
        """La plantilla masiva debe tener exactamente las columnas requeridas."""
        import db
        atletas = ["A1", "A2", "A3"]
        df = db.wellness_masivo_template(atletas)
        required = {"Nombre", "Sueño", "Estrés", "Fatiga", "Humor", "Dolor"}
        assert required.issubset(set(df.columns))
        assert list(df["Nombre"]) == atletas


# ─────────────────────────────────────────────────────────────────────────────
# Task 5 — Historial individual: barras + tendencia
# ─────────────────────────────────────────────────────────────────────────────

class TestSemaforoHistoricoChart:
    def _make_df_historial(self, n: int = 10) -> pd.DataFrame:
        return pd.DataFrame({
            "fecha":  [f"2026-0{(i // 28) + 1}-{(i % 28) + 1:02d}" for i in range(n)],
            "fatiga": np.random.uniform(20, 90, n),
            "estado": ["ÓPTIMO"] * n,
            "dqi":    [0.85] * n,
        })

    def test_fig_semaforo_historico_returns_figure(self):
        """Debe retornar un go.Figure."""
        import plotly.graph_objects as go
        from visualization.charts import fig_semaforo_historico
        df = self._make_df_historial()
        fig = fig_semaforo_historico(df, titulo="Atleta Test")
        assert isinstance(fig, go.Figure)

    def test_fig_semaforo_historico_has_bar_trace(self):
        """El primer trace debe ser go.Bar (barras de fatiga)."""
        import plotly.graph_objects as go
        from visualization.charts import fig_semaforo_historico
        df = self._make_df_historial()
        fig = fig_semaforo_historico(df, titulo="Test")
        bar_traces = [t for t in fig.data if isinstance(t, go.Bar)]
        assert len(bar_traces) >= 1, "Se esperaba al menos 1 trace de tipo Bar"

    def test_fig_semaforo_historico_has_trend_line(self):
        """Debe incluir un trace Scatter (curva de tendencia)."""
        import plotly.graph_objects as go
        from visualization.charts import fig_semaforo_historico
        df = self._make_df_historial()
        fig = fig_semaforo_historico(df, titulo="Test")
        scatter_traces = [t for t in fig.data if isinstance(t, go.Scatter)]
        assert len(scatter_traces) >= 1, "Se esperaba al menos 1 trace Scatter (tendencia)"

    def test_fig_semaforo_historico_has_threshold_lines(self):
        """Debe incluir líneas horizontales en y=25, 50, 75."""
        import plotly.graph_objects as go
        from visualization.charts import fig_semaforo_historico
        df = self._make_df_historial()
        fig = fig_semaforo_historico(df, titulo="Test")
        shapes = fig.layout.shapes or []
        y_vals = {round(float(s.y0)) for s in shapes if hasattr(s, "y0")}
        assert {25, 50, 75}.issubset(y_vals), (
            f"Faltan líneas umbral. Shapes y-values encontrados: {y_vals}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Task 6 — Panel de membresías fuzzy
# ─────────────────────────────────────────────────────────────────────────────

class TestMembershipPanel:
    """Verifica la función de cálculo de membresías para el panel."""

    def test_calcular_membresias_atleta_returns_four_keys(self):
        """Debe retornar μ para los 4 conjuntos del output."""
        from app import calcular_membresias_atleta
        resultado = calcular_membresias_atleta(indice_fatiga=80.0)
        assert set(resultado.keys()) == {"optimo", "alerta_temprana", "fatiga_acumulada", "critico"}

    def test_calcular_membresias_atleta_values_in_range(self):
        """Todos los valores μ deben estar en [0, 1]."""
        from app import calcular_membresias_atleta
        for score in [10.0, 30.0, 60.0, 85.0]:
            resultado = calcular_membresias_atleta(indice_fatiga=score)
            for k, v in resultado.items():
                assert 0.0 <= v <= 1.0, f"μ fuera de rango para {k} con score={score}: {v}"


# ─────────────────────────────────────────────────────────────────────────────
# Task 7 — Dashboard cleanup (eliminar secciones duplicadas)
# ─────────────────────────────────────────────────────────────────────────────

class TestDashboardCleanup:
    def test_historial_batch_grid_removed_from_dashboard(self):
        """El dashboard no debe renderizar la cuadrícula masiva de historial."""
        src = (Path(__file__).parent.parent / "app.py").read_text()
        assert "Historial de Fatiga — Barras por Atleta" not in src, (
            "La sección de historial masivo sigue en tab_dashboard. Debe eliminarse."
        )

    def test_sesiones_expander_removed_from_dashboard(self):
        """El expander de sesiones (últimas 20) no debe existir en tab_dashboard."""
        src = (Path(__file__).parent.parent / "app.py").read_text()
        assert "Ver historial de sesiones (últimas 20)" not in src, (
            "El expander de historial de sesiones sigue en tab_dashboard."
        )
