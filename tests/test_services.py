"""
tests/test_services.py — Pruebas unitarias para la lógica de negocio.
"""
import pytest
import pandas as pd
import numpy as np
from core.services import calcular_metricas

def test_calcular_metricas_insuficientes():
    df = pd.DataFrame({
        "nombre": ["Atleta 1"] * 3,
        "fecha": pd.date_range("2026-01-01", periods=3),
        "vmp_hoy": [1.0, 1.1, 1.2]
    })
    res = calcular_metricas(df, "Atleta 1")
    assert isinstance(res, dict)
    assert res["estado"] == "INSUFICIENTE"
    assert res["n_sesiones"] == 3
    # Verificar que las llaves esperadas existan para evitar KeyErrors
    assert "acwr" in res
    assert "indice_fatiga" in res

def test_calcular_metricas_suficientes():
    df = pd.DataFrame({
        "nombre": ["Atleta 1"] * 10,
        "fecha": pd.date_range("2026-01-01", periods=10),
        "vmp_hoy": [1.0, 1.1, 1.0, 1.2, 1.1, 1.0, 1.1, 1.2, 1.1, 1.0]
    })
    res = calcular_metricas(df, "Atleta 1")
    assert res["n_sesiones"] == 10
    assert "acwr" in res
    assert "delta_pct" in res
