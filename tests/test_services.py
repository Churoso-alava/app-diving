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

def _df_atleta(n=15, vmp_base=1.2):
    fechas = pd.date_range("2025-01-01", periods=n, freq="D").strftime("%Y-%m-%d").tolist()
    vmps = [vmp_base + 0.01 * i for i in range(n)]
    # Añadir carga_subjetiva (RPE) default para que no falle el groupby
    return pd.DataFrame({"nombre": ["Ana"] * n, "fecha": fechas, "vmp_hoy": vmps, "carga_subjetiva": [5.0]*n})

def test_carga_integrada_no_depende_de_wellness():
    """
    carga_integrada_plan debe depender SOLO de clavados_planificados.
    """
    df = _df_atleta()
    clavados = [{"altura": 10.0, "dd": 2.5, "tipo": "PIKE"}]

    res_optimo = calcular_metricas(df, "Ana", clavados_planificados=clavados,
                                   wellness_respuestas={"sueno":1,"fatiga":1,"estres":1,"dolor":1,"humor":7})
    res_pesimo = calcular_metricas(df, "Ana", clavados_planificados=clavados,
                                   wellness_respuestas={"sueno":7,"fatiga":7,"estres":7,"dolor":7,"humor":1})

    assert abs(res_optimo["carga_integrada_plan"] - res_pesimo["carga_integrada_plan"]) < 0.01, (
        f"carga varía con wellness: {res_optimo['carga_integrada_plan']:.2f} vs "
        f"{res_pesimo['carga_integrada_plan']:.2f}. El wellness no debe modificar la carga."
    )

def test_beta_ruido_retorna_cero():
    """VMP oscilando sin tendencia → beta = 0.0 (Theil-Sen IC incluye 0)."""
    np.random.seed(99)
    vmps = [1.2 + np.random.uniform(-0.08, 0.08) for _ in range(7)]
    fechas = pd.date_range("2025-01-01", periods=7, freq="D").strftime("%Y-%m-%d").tolist()
    df = pd.DataFrame({"nombre": ["T"]*7, "fecha": fechas, "vmp_hoy": vmps, "carga_subjetiva": [5.0]*7})
    res = calcular_metricas(df, "T")
    assert res["beta_aguda"] == 0.0, f"Ruido → beta_aguda debe ser 0.0, fue {res['beta_aguda']}"

def test_beta_caida_lineal_es_negativa():
    """Caída lineal perfecta → beta negativo significativo."""
    vmps = [1.5, 1.42, 1.35, 1.28, 1.21, 1.14, 1.07]
    fechas = pd.date_range("2025-01-01", periods=7, freq="D").strftime("%Y-%m-%d").tolist()
    df = pd.DataFrame({"nombre": ["T"]*7, "fecha": fechas, "vmp_hoy": vmps, "carga_subjetiva": [5.0]*7})
    res = calcular_metricas(df, "T")
    assert res["beta_aguda"] < -0.01, f"Caída → beta_aguda debe ser <-0.01, fue {res['beta_aguda']}"

def test_beta_robusto_a_outlier():
    """Theil-Sen debe mantener la tendencia pese a un outlier extremo."""
    vmps = [1.40, 1.35, 1.30, 5.00, 1.20, 1.15, 1.10]
    fechas = pd.date_range("2025-01-01", periods=7, freq="D").strftime("%Y-%m-%d").tolist()
    df = pd.DataFrame({"nombre": ["T"]*7, "fecha": fechas, "vmp_hoy": vmps, "carga_subjetiva": [5.0]*7})
    res = calcular_metricas(df, "T")
    assert res["beta_aguda"] < 0.0, f"Theil-Sen falló ante outlier: beta={res['beta_aguda']}"

def test_acwr_robusto_a_sesion_atipica():
    """Si los datos son no-normales, ACWR debe usar mediana (robusto)."""
    # 13 estables, 1 outlier alto
    vmps = [1.2]*13 + [2.8]
    fechas = pd.date_range("2025-01-01", periods=14, freq="D").strftime("%Y-%m-%d").tolist()
    df = pd.DataFrame({"nombre": ["T"]*14, "fecha": fechas, "vmp_hoy": vmps, "carga_subjetiva": [5.0]*14})
    res = calcular_metricas(df, "T")
    # Con mediana MMA7 = 1.2. ACWR = 1.2/1.2 = 1.0. 
    # Con media MMA7 ≈ 1.43. ACWR ≈ 1.19.
    assert res["acwr"] < 1.10, f"ACWR no fue robusto ante outlier: {res['acwr']}"
