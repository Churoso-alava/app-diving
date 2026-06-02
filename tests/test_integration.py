import pytest
import pandas as pd
import numpy as np
from core.fuzzy_engine import construir_motor_fuzzy
from core.services import pipeline_diagnostico
from core.analysis import matriz_spearman, reporte_redundancias

@pytest.fixture(scope="module")
def motor_y_sim():
    return construir_motor_fuzzy()

def _generar_df(nombre, vmps, start="2025-01-01"):
    fechas = pd.date_range(start, periods=len(vmps), freq="D").strftime("%Y-%m-%d").tolist()
    return pd.DataFrame({"nombre": [nombre]*len(vmps), "fecha": fechas, "vmp_hoy": vmps})

def test_atleta_optimo_clasifica_correcto(motor_y_sim):
    """Tendencia positiva + buenas condiciones → ÓPTIMO."""
    _, sim = motor_y_sim
    vmps = [1.00, 1.05, 1.10, 1.15, 1.20, 1.25, 1.30, 1.35,
            1.40, 1.42, 1.44, 1.45, 1.46, 1.47, 1.48]
    df = _generar_df("A", vmps)
    res = pipeline_diagnostico("A", df, sim,
        wellness_respuestas={"sueno":1,"fatiga":1,"estres":2,"dolor":1,"humor":7},
        clavados_planificados=[{"altura":10.0,"dd":2.5,"tipo":"PIKE"}])
    assert res["indice_fatiga"] >= 70, f"Óptimo → índice={res['indice_fatiga']}"
    assert "ÓPTIMO" in res["estado"]

def test_atleta_critico_clasifica_correcto(motor_y_sim):
    """Caída severa + alta carga + mal wellness → CRÍTICO o FATIGA ACUMULADA."""
    _, sim = motor_y_sim
    vmps = [1.40,1.35,1.30,1.25,1.20,1.10,1.00,0.90,
            0.80,0.72,0.65,0.60,0.55,0.50,0.45]
    df = _generar_df("B", vmps)
    res = pipeline_diagnostico("B", df, sim,
        wellness_respuestas={"sueno":7,"fatiga":7,"estres":6,"dolor":6,"humor":1},
        clavados_planificados=[{"altura":10.0,"dd":3.5,"tipo":"TWIST"},
                                {"altura":7.5,"dd":3.0,"tipo":"SYNC"}])
    assert res["indice_fatiga"] <= 40, f"Crítico → índice={res['indice_fatiga']}"
    assert "CRÍTICO" in res["estado"] or "FATIGA" in res["estado"]

def test_datos_insuficientes_retorna_insuficiente(motor_y_sim):
    """Con <4 sesiones → INSUFICIENTE, no excepción."""
    _, sim = motor_y_sim
    df = _generar_df("C", [1.2, 1.1, 1.15])
    res = pipeline_diagnostico("C", df, sim)
    assert res["estado"] == "INSUFICIENTE"
    assert res["indice_fatiga"] is None

def test_sin_clavados_planificados(motor_y_sim):
    """Sin clavados planificados → carga=0, no lanza excepción."""
    _, sim = motor_y_sim
    df = _generar_df("D", [1.2]*10)
    res = pipeline_diagnostico("D", df, sim)
    assert res["carga_integrada_plan"] == 0.0

def test_critico_con_plan_recuperacion_no_sube_a_optimo(motor_y_sim):
    """Regla 25 fix: atleta crítico + plan recovery → índice ≤ 40."""
    _, sim = motor_y_sim
    vmps = [1.40,1.30,1.20,1.10,1.00,0.90,0.80,0.70,0.60,0.50,0.45,0.42,0.40,0.38,0.36]
    df = _generar_df("E", vmps)
    res = pipeline_diagnostico("E", df, sim,
        wellness_respuestas={"sueno":7,"fatiga":7,"estres":6,"dolor":5,"humor":1},
        clavados_planificados=[{"altura":1.0,"dd":1.2,"tipo":"FEET"}])  # carga mínima = RECUPERACION
    assert res["indice_fatiga"] <= 40, (
        f"Atleta crítico + plan recovery clasifica como {res['estado']} "
        f"(índice={res['indice_fatiga']}). Regla 25 no debe rescatar al atleta."
    )

def test_analysis_modulo_sobre_datos_pipeline(motor_y_sim):
    """
    El módulo de análisis bivariado funciona sobre datos reales del pipeline.
    Verifica que delta_pct y z_meso muestran alta correlación (redundancia esperada).
    """
    _, sim = motor_y_sim
    registros = []
    for seed in range(20):
        np.random.seed(seed)
        vmps = list(np.random.normal(1.2, 0.1, 15))
        df = _generar_df(f"Atleta_{seed}", vmps)
        res = pipeline_diagnostico(f"Atleta_{seed}", df, sim)
        if res.get("indice_fatiga") is not None:
            registros.append(res)

    df_metricas = pd.DataFrame(registros)
    reporte = reporte_redundancias(df_metricas, ["delta_pct", "z_meso"], threshold=0.70)
    assert len(reporte) >= 1, (
        "delta_pct y z_meso deben detectarse como redundantes (|rho|>0.70). "
        "Si no, revisar si los datos del pipeline están siendo calculados correctamente."
    )
