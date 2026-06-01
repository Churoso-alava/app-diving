import pytest
import plotly.graph_objects as go
from ui.charts import fig_fatiga_radial

def test_fig_fatiga_radial_returns_figure():
    fig = fig_fatiga_radial(50, "Test Atleta")
    assert isinstance(fig, go.Figure)

def test_fig_fatiga_radial_has_correct_data():
    score = 80
    fig = fig_fatiga_radial(score, "Test Atleta")
    assert fig.data[0].value == score
