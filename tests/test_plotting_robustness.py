"""
Tests para validar que visualization/charts.py maneja datos con NaN
"""
import pandas as pd
import numpy as np
import pytest
from datetime import date, timedelta

try:
    from visualization.charts import fig_semaforo_historico
except ImportError:
    pytest.skip("visualization.charts not available", allow_module_level=True)

# Mocking the log object to prevent errors during testing
class MockLog:
    def warning(self, msg, *args): pass
    def info(self, msg, *args): pass
    def error(self, msg, *args): pass
log = MockLog()

# Define the dark layout for consistent testing
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
# Tests para fig_semaforo_historico
# ─────────────────────────────────────────────────────────────────────────────

def test_fig_semaforo_historico_maneja_dataframe_vacio():
    """Verifica que fig_semaforo_historico maneja DataFrames vacíos"""
    df = pd.DataFrame({'fecha': [], 'fatiga': []})
    fig = fig_semaforo_historico(df)
    assert fig is not None
    assert "Sin datos" in fig.layout.title.text # Check default title text

def test_fig_semaforo_historico_filtra_nan():
    """Verifica que fig_semaforo_historico filtra NaN antes de procesar"""
    df = pd.DataFrame({
        'fecha': ['2026-04-01', '2026-04-02', '2026-04-03'],
        'fatiga': [50.0, np.nan, 75.0]
    })
    fig = fig_semaforo_historico(df)
    assert fig is not None
    # The function should filter out the NaN row and still render the chart
    bar_trace = fig.data[0]
    assert len(bar_trace.x) == 2 # Should have 2 points after filtering NaN
    assert "2026-04-01" in bar_trace.x
    assert "2026-04-03" in bar_trace.x
    assert "2026-04-02" not in bar_trace.x # This was the NaN row

def test_fig_semaforo_historico_colores_correctos():
    """Verifica que los colores de zona se asignan correctamente"""
    df = pd.DataFrame({
        'fecha': pd.to_datetime(['2026-04-01', '2026-04-02', '2026-04-03', '2026-04-04']),
        'fatiga': [10.0, 40.0, 60.0, 80.0]  # crítico, alerta, temprana, óptimo
    })
    fig = fig_semaforo_historico(df)
    assert fig is not None
    
    bar_trace = fig.data[0] # The bar trace contains the fatiga values and colors
    
    # Expected colors based on implementation logic:
    # < 25: #ef4444 (crítico)
    # 25-50: #f97316 (fatiga acumulada)
    # 50-75: #eab308 (alerta temprana)
    # >= 75: #22c55e (óptimo)
    
    expected_colors = [
        "#ef4444", # 10.0
        "#f97316", # 40.0
        "#eab308", # 60.0
        "#22c55e"  # 80.0
    ]
    
    assert bar_trace.marker.color == expected_colors

def test_fig_semaforo_historico_no_valid_data_after_filter():
    """Verifica que la función maneje casos donde todos los datos son NaN/NaT después del filtrado."""
    df_input = pd.DataFrame({
        "fecha": [np.nan, pd.NaT, None],
        "fatiga": [np.nan, np.nan, np.nan]
    })
    fig = fig_semaforo_historico(df_input, titulo="Test No Valid Data")
    
    assert isinstance(fig, go.Figure)
    assert fig.layout.title.text == "Sin datos válidos" # Expected title when no valid data remains
    assert len(fig.data) == 0 # No traces should be added
