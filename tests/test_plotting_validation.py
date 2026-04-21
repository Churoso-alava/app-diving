"""
tests/test_plotting_validation.py — Suite de validación para visualization/charts.py
Verifica que todas las funciones de plotting manejen correctamente DataFrames vacíos o con NaN.
"""
import unittest
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from visualization.charts import (
    fig_semaforo_historico,
    fig_semaforo_barras,
    fig_vmp_tendencia,
    fig_historial_barras_atleta,
    fig_membership_fuzzy,
    fig_vmp_ratio_thresholds
)

class TestPlottingValidation(unittest.TestCase):

    def test_fig_semaforo_historico_nan(self):
        # Con 2 puntos, uno es NaN, queda 1 punto. Polyfit debería manejarlo ahora.
        df = pd.DataFrame({
            'fecha': ['2026-04-01', '2026-04-02'],
            'fatiga': [50.0, np.nan]
        })
        fig = fig_semaforo_historico(df)
        self.assertIsInstance(fig, go.Figure)
        # Debe filtrar el NaN y tener 1 punto en las barras
        self.assertEqual(len(fig.data[0].x), 1)

    def test_fig_semaforo_barras_nan(self):
        df = pd.DataFrame({
            'nombre': ['Atleta 1', 'Atleta 2'],
            'score': [80.0, np.nan],
            'estado': ['🟢 ÓPTIMO', '⚪ SIN DATOS'],
            'fecha': ['2026-04-01', '2026-04-02']
        })
        fig = fig_semaforo_barras(df)
        self.assertIsInstance(fig, go.Figure)
        # Debe filtrar el NaN y tener 1 barra
        self.assertEqual(len(fig.data[0].y), 1)

    def test_fig_vmp_tendencia_nan(self):
        df = pd.DataFrame({
            'fecha': ['2026-04-01', '2026-04-02'],
            'vmp_hoy': [0.8, np.nan],
            'mmc28': [0.75, 0.75],
            'mma7': [0.78, 0.78],
            'swc_up': [0.82, 0.82],
            'swc_down': [0.72, 0.72]
        })
        fig = fig_vmp_tendencia(df, "Atleta Test", delta_pct=5.0)
        self.assertIsInstance(fig, go.Figure)
        # Debe filtrar NaN en vmp_hoy
        vmp_trace = [t for t in fig.data if t.name == "VMP CMJ (m/s)"][0]
        self.assertEqual(len(vmp_trace.x), 1)

    def test_fig_historial_barras_atleta_nan(self):
        df = pd.DataFrame({
            'fecha': ['2026-04-01', '2026-04-02'],
            'fatiga': [60.0, np.nan],
            'estado': ['🟡 ALERTA TEMPRANA', np.nan]
        })
        fig = fig_historial_barras_atleta(df, "Atleta Test")
        self.assertIsInstance(fig, go.Figure)
        # Debe filtrar el NaN y tener 1 barra
        self.assertEqual(len(fig.data[0].x), 1)

    def test_fig_vmp_ratio_thresholds_nan(self):
        df = pd.DataFrame({
            'fecha': ['2026-04-01', '2026-04-02'],
            'vmp': [0.8, np.nan],
            'ratio': [1.05, np.nan],
            'threshold_low': [0.9, 0.9],
            'threshold_high': [1.1, 1.1]
        })
        fig = fig_vmp_ratio_thresholds(df, "Atleta Test")
        self.assertIsInstance(fig, go.Figure)
        # El nombre exacto es "VMP CMJ (m/s)"
        vmp_trace = [t for t in fig.data if "VMP" in t.name][0]
        self.assertEqual(len(vmp_trace.x), 1)

    def test_empty_dataframes(self):
        df_empty = pd.DataFrame()
        funcs = [
            ("historico", lambda: fig_semaforo_historico(df_empty)),
            ("barras",    lambda: fig_semaforo_barras(df_empty)),
            ("tendencia", lambda: fig_vmp_tendencia(df_empty, "Test", 0.0)),
            ("hist_atleta", lambda: fig_historial_barras_atleta(df_empty, "Test")),
            ("ratio",     lambda: fig_vmp_ratio_thresholds(df_empty, "Test"))
        ]
        for name, f in funcs:
            fig = f()
            self.assertIsInstance(fig, go.Figure, f"Función {name} no retornó un Figure")
            title_text = fig.layout.title.text if fig.layout.title.text else ""
            self.assertTrue("Sin datos" in title_text or "Sin historial" in title_text, 
                            f"Función {name} falló con título: '{title_text}'")

if __name__ == "__main__":
    unittest.main()
