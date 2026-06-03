import pytest
from core.fuzzy_engine import construir_motor_fuzzy, evaluar_atleta

@pytest.fixture(scope="module")
def motor():
    _, simulador = construir_motor_fuzzy()
    return simulador

def _metricas_base(**overrides):
    base = {
        "atleta": "test", "vmp_hoy": 1.20, "acwr": 1.00,
        "delta_pct": 5.0, "z_meso": 0.0, "beta_aguda": 0.0,
        "beta_28": 0.0, "es_ruido_biologico": False,
        "caida_absoluta": 0.05, "swc_personal": 0.05,
        "n_sesiones_desc": 0, "calidad_dato": "alta",
        "dias_sin_datos": 0, "edad_atleta": 20,
    }
    return {**base, **overrides}

def test_atleta_critico_no_mejora_con_plan_recuperacion(motor):
    """
    Regla 25 bug: atleta con VMP muy baja y ACWR excesivo NO debe
    ser ÓPTIMO solo porque la carga planificada es de recuperación.
    """
    m = _metricas_base(vmp_hoy=0.45, acwr=1.65, delta_pct=30.0,
                        z_meso=-3.0, beta_aguda=-0.15, beta_28=-0.08)
    res = evaluar_atleta(motor, m, wellness_norm=0.1, carga_integrada_plan=20.0)
    assert res["indice_fatiga"] <= 40, (
        f"Atleta crítico + plan recuperación → índice={res['indice_fatiga']} "
        f"({res['estado']}). Debe ser ≤40 (CRÍTICO o FATIGA ACUMULADA)."
    )

def test_acwr_bajo_delta_alarma_no_es_critico_con_vmp_funcional(motor):
    """
    ACWR bajo + delta alarma + vmp funcional NO debe → CRÍTICO.
    La combinación es compatible con desentrenamiento, no sobrecarga.
    Máximo: FATIGA ACUMULADA (índice 25-50).
    """
    m = _metricas_base(vmp_hoy=1.05, acwr=0.60, delta_pct=25.0,
                        z_meso=-1.0, beta_aguda=-0.12, beta_28=-0.05)
    res = evaluar_atleta(motor, m, wellness_norm=0.5, carga_integrada_plan=80.0)
    assert res["indice_fatiga"] >= 25, (
        f"ACWR bajo + delta alarma + VMP funcional → índice={res['indice_fatiga']} "
        f"({res['estado']}). No debería ser CRÍTICO (<25)."
    )

def test_vmp_muy_baja_con_wellness_deficiente_es_critico(motor):
    """VMP muy baja + wellness deficiente → CRÍTICO aunque ACWR sea óptimo."""
    m = _metricas_base(vmp_hoy=0.50, acwr=1.00, delta_pct=5.0, z_meso=-0.5)
    res = evaluar_atleta(motor, m, wellness_norm=0.05, carga_integrada_plan=100.0)
    assert res["indice_fatiga"] <= 40, (
        f"VMP muy baja + wellness pésimo → {res['estado']} (índice={res['indice_fatiga']}). "
        f"Debe ser CRÍTICO o FATIGA ACUMULADA."
    )

def test_vmp_alta_con_condiciones_optimas_es_optimo(motor):
    """VMP alta + ACWR óptimo + wellness óptimo → ÓPTIMO."""
    m = _metricas_base(vmp_hoy=2.10, acwr=1.00, delta_pct=-5.0, z_meso=1.5,
                        beta_aguda=0.05, beta_28=0.02)
    res = evaluar_atleta(motor, m, wellness_norm=0.95, carga_integrada_plan=90.0)
    assert res["indice_fatiga"] >= 75, (
        f"VMP alta + óptimas condiciones → {res['estado']} (índice={res['indice_fatiga']}). "
        f"Debe ser ÓPTIMO."
    )

def test_motor_no_cae_en_fallback_50(motor):
    """
    BUG-001 Fix: El motor debe recibir todos los inputs (incluyendo carga_subjetiva/RPE)
    y retornar un valor calculado, no el fallback de 50.0.
    """
    m = _metricas_base(vmp_hoy=1.40, acwr=1.10, carga_subjetiva=3.0) 
    res = evaluar_atleta(motor, m, wellness_norm=0.9, carga_integrada_plan=50.0)
    assert res["indice_fatiga"] != 50.0, "BUG-001: El motor sigue cayendo en fallback 50.0"
    assert res["indice_fatiga"] > 70.0, f"Se esperaba un indice alto para condiciones optimas, obtenido: {res['indice_fatiga']}"
