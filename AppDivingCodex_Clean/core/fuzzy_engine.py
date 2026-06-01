"""
fuzzy.py — Motor Fuzzy Mamdani v4.2
Sin dependencias de Streamlit ni base de datos.
"""
import logging

import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

from core.schemas import VMP_MAX, VMP_MIN

log = logging.getLogger(__name__)


def construir_sistema_fuzzy():
    """Retorna la tupla de variables difusas (8 antecedentes + 1 consecuente)."""
    u_vmp = np.arange(VMP_MIN, VMP_MAX + 0.001, 0.001)
    u_acwr = np.arange(0.50, 1.81, 0.01)
    u_delta = np.arange(-20, 41, 0.5)
    u_zmeso = np.arange(-4, 4.1, 0.1)
    u_ba = np.arange(-0.25, 0.26, 0.001)
    u_b28 = np.arange(-0.25, 0.26, 0.001)
    u_well = np.arange(0.0, 1.01, 0.01)
    u_ci = np.arange(0.0, 201.0, 0.5)
    u_rpe = np.arange(1, 11, 1)
    u_fat = np.arange(0, 101, 1)

    vmp_v = ctrl.Antecedent(u_vmp, "vmp_hoy")
    acwr_v = ctrl.Antecedent(u_acwr, "acwr")
    delta_v = ctrl.Antecedent(u_delta, "delta_pct")
    zmeso_v = ctrl.Antecedent(u_zmeso, "z_meso")
    ba_v = ctrl.Antecedent(u_ba, "beta_aguda")
    b28_v = ctrl.Antecedent(u_b28, "beta_28")
    wellness_v = ctrl.Antecedent(u_well, "wellness_norm")
    ci_v = ctrl.Antecedent(u_ci, "carga_integrada_plan")
    rpe_v = ctrl.Antecedent(u_rpe, "carga_subjetiva")
    fat_v = ctrl.Consequent(u_fat, "fatiga")

    vmp_v["muy_baja"] = fuzz.trapmf(u_vmp, [0.10, 0.10, 0.55, 0.80])
    vmp_v["baja"] = fuzz.trimf(u_vmp, [0.65, 0.90, 1.10])
    vmp_v["funcional"] = fuzz.trapmf(u_vmp, [1.00, 1.18, 1.70, 1.92])
    vmp_v["alta"] = fuzz.trapmf(u_vmp, [1.80, 2.05, 2.50, 2.50])

    acwr_v["bajo"] = fuzz.trapmf(u_acwr, [0.50, 0.50, 0.70, 0.85])
    acwr_v["optimo"] = fuzz.trapmf(u_acwr, [0.78, 0.88, 1.20, 1.30])
    acwr_v["alto"] = fuzz.trimf(u_acwr, [1.25, 1.40, 1.55])
    acwr_v["excesivo"] = fuzz.trapmf(u_acwr, [1.45, 1.55, 1.80, 1.80])

    delta_v["ganancia"] = fuzz.trapmf(u_delta, [-20, -20, -5, 0])
    delta_v["tolerable"] = fuzz.trapmf(u_delta, [-2, 0, 8, 12])
    delta_v["vigilancia"] = fuzz.trimf(u_delta, [10, 15, 22])
    delta_v["alarma"] = fuzz.trapmf(u_delta, [18, 22, 40, 40])

    zmeso_v["muy_bajo"] = fuzz.trapmf(u_zmeso, [-4.0, -4.0, -2.2, -1.5])
    zmeso_v["bajo"] = fuzz.trimf(u_zmeso, [-2.0, -1.2, -0.5])
    zmeso_v["normal"] = fuzz.trimf(u_zmeso, [-0.8, 0.0, 0.8])
    zmeso_v["elevado"] = fuzz.trapmf(u_zmeso, [0.6, 1.2, 4.0, 4.0])

    ba_v["neg_fuerte"] = fuzz.trapmf(u_ba, [-0.25, -0.25, -0.06, -0.025])
    ba_v["neg_moderada"] = fuzz.trimf(u_ba, [-0.04, -0.015, -0.005])
    ba_v["estable"] = fuzz.trimf(u_ba, [-0.008, 0.0, 0.008])
    ba_v["positiva"] = fuzz.trapmf(u_ba, [0.005, 0.025, 0.25, 0.25])

    b28_v["deterioro"] = fuzz.trapmf(u_b28, [-0.25, -0.25, -0.03, -0.01])
    b28_v["estable"] = fuzz.trimf(u_b28, [-0.012, 0.0, 0.012])
    b28_v["mejora"] = fuzz.trapmf(u_b28, [0.010, 0.03, 0.25, 0.25])

    wellness_v["DEFICIENTE"] = fuzz.trapmf(u_well, [0.00, 0.00, 0.25, 0.45])
    wellness_v["ACEPTABLE"] = fuzz.trimf(u_well, [0.30, 0.50, 0.70])
    wellness_v["OPTIMO"] = fuzz.trapmf(u_well, [0.55, 0.75, 1.00, 1.00])

    ci_v["RECUPERACION"] = fuzz.trapmf(u_ci, [0, 0, 25, 55])
    ci_v["MANTENIMIENTO"] = fuzz.trapmf(u_ci, [35, 60, 90, 115])
    ci_v["DESARROLLO"] = fuzz.trapmf(u_ci, [90, 115, 145, 165])
    ci_v["SOBRECARGA"] = fuzz.trapmf(u_ci, [145, 165, 200, 200])

    fat_v["critico"] = fuzz.trapmf(u_fat, [0, 0, 18, 28])
    fat_v["fatiga_acumulada"] = fuzz.trimf(u_fat, [25, 37, 52])
    fat_v["alerta_temprana"] = fuzz.trimf(u_fat, [50, 62, 76])
    fat_v["optimo"] = fuzz.trapmf(u_fat, [75, 88, 100, 100])

    return vmp_v, acwr_v, delta_v, zmeso_v, ba_v, b28_v, wellness_v, ci_v, fat_v


def construir_sistema_fuzzy_legacy():
    """Compatibilidad retro con consumidores que esperan 6 elementos."""
    _vmp_v, acwr_v, delta_v, zmeso_v, ba_v, b28_v, _wellness_v, _ci_v, fat_v = construir_sistema_fuzzy()
    return acwr_v, delta_v, zmeso_v, ba_v, b28_v, fat_v


def construir_reglas(vmp_v, acwr_v, delta_v, zmeso_v, ba_v, b28_v, wellness_v, ci_v, rpe_v, fat_v):
    return [
        ctrl.Rule(vmp_v["muy_baja"] & (delta_v["alarma"] | zmeso_v["muy_bajo"]), fat_v["critico"]),
        ctrl.Rule(vmp_v["baja"] & ba_v["neg_fuerte"] & (delta_v["vigilancia"] | delta_v["alarma"]), fat_v["fatiga_acumulada"]),
        ctrl.Rule(vmp_v["funcional"] & acwr_v["optimo"] & ba_v["estable"] & delta_v["tolerable"], fat_v["optimo"]),
        ctrl.Rule(vmp_v["alta"] & (delta_v["ganancia"] | ba_v["positiva"]) & (b28_v["estable"] | b28_v["mejora"]), fat_v["optimo"]),

        # Reglas RPE nuevas
        ctrl.Rule(rpe_v["alta"] & wellness_v["DEFICIENTE"], fat_v["critico"]),
        ctrl.Rule(rpe_v["alta"] & ci_v["SOBRECARGA"], fat_v["critico"]),
        ctrl.Rule(rpe_v["baja"] & wellness_v["OPTIMO"], fat_v["optimo"]),

        # ... (rest of the rules)
        ctrl.Rule(vmp_v["muy_baja"] & wellness_v["DEFICIENTE"], fat_v["critico"]),
        ctrl.Rule(vmp_v["muy_baja"] & acwr_v["excesivo"], fat_v["critico"]),
        ctrl.Rule(vmp_v["muy_baja"] & acwr_v["optimo"], fat_v["fatiga_acumulada"]),
        ctrl.Rule(vmp_v["baja"] & acwr_v["excesivo"] & wellness_v["DEFICIENTE"], fat_v["fatiga_acumulada"]),
        ctrl.Rule(vmp_v["alta"] & acwr_v["optimo"] & wellness_v["OPTIMO"], fat_v["optimo"]),
        ctrl.Rule(vmp_v["alta"] & wellness_v["DEFICIENTE"], fat_v["alerta_temprana"]),

        ctrl.Rule(acwr_v["bajo"] & delta_v["alarma"] & ba_v["neg_fuerte"], fat_v["fatiga_acumulada"]),
        ctrl.Rule(acwr_v["bajo"] & b28_v["deterioro"] & ba_v["neg_fuerte"], fat_v["fatiga_acumulada"]),
        ctrl.Rule(delta_v["alarma"] & b28_v["deterioro"] & zmeso_v["muy_bajo"], fat_v["critico"]),
        ctrl.Rule(acwr_v["excesivo"] & zmeso_v["muy_bajo"] & ba_v["neg_fuerte"], fat_v["critico"]),

        ctrl.Rule(acwr_v["bajo"] & delta_v["alarma"], fat_v["fatiga_acumulada"]),
        ctrl.Rule(acwr_v["bajo"] & delta_v["vigilancia"] & b28_v["deterioro"], fat_v["fatiga_acumulada"]),
        ctrl.Rule(acwr_v["optimo"] & ba_v["neg_fuerte"] & zmeso_v["muy_bajo"], fat_v["fatiga_acumulada"]),
        ctrl.Rule(delta_v["alarma"] & ba_v["neg_moderada"], fat_v["fatiga_acumulada"]),
        ctrl.Rule(acwr_v["alto"] & delta_v["vigilancia"] & b28_v["deterioro"], fat_v["fatiga_acumulada"]),
        ctrl.Rule(acwr_v["excesivo"] & zmeso_v["normal"], fat_v["fatiga_acumulada"]),
        ctrl.Rule(b28_v["deterioro"] & (ba_v["neg_fuerte"] | ba_v["neg_moderada"]) & (zmeso_v["bajo"] | zmeso_v["muy_bajo"]), fat_v["fatiga_acumulada"]),
        ctrl.Rule(acwr_v["excesivo"] & (zmeso_v["bajo"] | zmeso_v["muy_bajo"]) & ba_v["neg_moderada"], fat_v["fatiga_acumulada"]),

        ctrl.Rule(delta_v["vigilancia"] & ba_v["neg_moderada"] & zmeso_v["normal"], fat_v["alerta_temprana"]),
        ctrl.Rule(acwr_v["optimo"] & zmeso_v["bajo"] & delta_v["vigilancia"], fat_v["alerta_temprana"]),
        ctrl.Rule(acwr_v["alto"] & ba_v["estable"] & delta_v["tolerable"], fat_v["alerta_temprana"]),
        ctrl.Rule(b28_v["deterioro"] & delta_v["tolerable"] & zmeso_v["bajo"], fat_v["alerta_temprana"]),
        ctrl.Rule(ba_v["neg_fuerte"] & b28_v["estable"] & zmeso_v["normal"], fat_v["alerta_temprana"]),
        ctrl.Rule(ba_v["positiva"] & zmeso_v["bajo"], fat_v["alerta_temprana"]),

        ctrl.Rule(acwr_v["optimo"] & delta_v["tolerable"] & ba_v["positiva"] & b28_v["mejora"], fat_v["optimo"]),
        ctrl.Rule(acwr_v["optimo"] & delta_v["ganancia"] & b28_v["estable"], fat_v["optimo"]),
        ctrl.Rule(acwr_v["optimo"] & zmeso_v["normal"] & ba_v["estable"] & delta_v["tolerable"], fat_v["optimo"]),
        ctrl.Rule((acwr_v["bajo"] | acwr_v["optimo"]) & zmeso_v["elevado"] & (ba_v["estable"] | ba_v["positiva"]), fat_v["optimo"]),
        ctrl.Rule(acwr_v["optimo"] & zmeso_v["elevado"] & b28_v["mejora"], fat_v["optimo"]),

        ctrl.Rule(ci_v["SOBRECARGA"] & wellness_v["DEFICIENTE"], fat_v["critico"]),
        ctrl.Rule(ci_v["DESARROLLO"] & wellness_v["DEFICIENTE"], fat_v["fatiga_acumulada"]),
        ctrl.Rule(ci_v["MANTENIMIENTO"] & wellness_v["OPTIMO"], fat_v["optimo"]),
        ctrl.Rule(
            ci_v["RECUPERACION"] & (vmp_v["funcional"] | vmp_v["alta"])
            & (acwr_v["optimo"] | acwr_v["bajo"]),
            fat_v["optimo"]
        ),
        ctrl.Rule(
            ci_v["RECUPERACION"] & vmp_v["baja"] & acwr_v["optimo"],
            fat_v["alerta_temprana"]
        ),
        ctrl.Rule(ci_v["SOBRECARGA"] & wellness_v["OPTIMO"], fat_v["alerta_temprana"]),
    ]


def construir_motor_fuzzy():
    vars_tuple = construir_sistema_fuzzy()
    vmp_v, acwr_v, delta_v, zmeso_v, ba_v, b28_v, wellness_v, ci_v, rpe_v, fat_v = vars_tuple
    reglas = construir_reglas(vmp_v, acwr_v, delta_v, zmeso_v, ba_v, b28_v, wellness_v, ci_v, rpe_v, fat_v)
    sistema = ctrl.ControlSystem(reglas)
    simulador = ctrl.ControlSystemSimulation(sistema)
    log.info("Motor fuzzy Mamdani v4.2 construido (%d reglas).", len(reglas))
    return vars_tuple, simulador


def evaluar_atleta(
    simulador,
    metricas: dict,
    wellness_norm: float = 0.5,
    carga_integrada_plan: float = 0.0,
) -> dict:
    if metricas.get("es_ruido_biologico", False):
        metricas_fuzzy = {**metricas, "delta_pct": 0.0}
        nota_swc = (
            f"⬇ Caída {metricas['caida_absoluta']:.3f} m/s < SWC "
            f"{metricas['swc_personal']:.3f} m/s → variabilidad biológica normal."
        )
    else:
        metricas_fuzzy = metricas
        nota_swc = ""

    try:
        simulador.input["vmp_hoy"] = metricas_fuzzy["vmp_hoy"]
        simulador.input["acwr"] = metricas_fuzzy["acwr"]
        simulador.input["delta_pct"] = metricas_fuzzy["delta_pct"]
        simulador.input["z_meso"] = metricas_fuzzy["z_meso"]
        simulador.input["beta_aguda"] = metricas_fuzzy["beta_aguda"]
        simulador.input["beta_28"] = metricas_fuzzy["beta_28"]
        simulador.input["wellness_norm"] = wellness_norm
        simulador.input["carga_integrada_plan"] = carga_integrada_plan
        simulador.compute()
        indice = float(simulador.output["fatiga"])
    except Exception as exc:
        log.warning("Motor fuzzy falló para atleta '%s': %s — usando fallback 50.0", metricas.get("atleta", "?"), exc)
        indice = 50.0

    if indice >= 75:
        estado, color = "🟢 ÓPTIMO", "#16a34a"
        accion_primaria = "Ejecutar planificación tal cual"
        contexto_cientifico = "Carga y estado pre-entrenamiento compatibles con el plan del día."
    elif indice >= 50:
        estado, color = "🟡 ALERTA TEMPRANA", "#ca8a04"
        accion_primaria = "Reducir carga planificada 10-15% o bajar altura/DD"
        contexto_cientifico = "Ajuste preventivo para mantener estímulo sin sobrerreacción aguda."
    elif indice >= 25:
        estado, color = "🟠 FATIGA ACUMULADA", "#ea580c"
        accion_primaria = "Sesión regenerativa planificada únicamente"
        contexto_cientifico = "Priorizar recuperación activa y control de carga interna."
    else:
        estado, color = "🔴 CRÍTICO", "#dc2626"
        accion_primaria = "Cancelar carga intensa, solo trabajo seco/banco"
        contexto_cientifico = "Riesgo elevado; evitar impacto y revaluar disponibilidad."

    advertencias: list[str] = []
    if metricas.get("edad_atleta", 18) < 15 and metricas["delta_pct"] > 20:
        advertencias.append("🚨 Estrés pediátrico — revisar carga semanal")

    calidad = metricas.get("calidad_dato", "alta")
    dias_sin_datos = metricas.get("dias_sin_datos", 0)
    if calidad == "insuficiente":
        indice = min(indice, 62.0)
        advertencias.append("⚠️ Datos insuficientes — decisión con baja confianza")
    elif calidad == "baja" and dias_sin_datos > 7:
        advertencias.append(f"⚠️ Sin datos hace {dias_sin_datos} días — interpretar con cautela")

    if metricas.get("n_sesiones_desc", 0) >= 3:
        advertencias.append("📉 Tendencia descendente en 3 sesiones consecutivas")

    accion = accion_primaria
    if advertencias:
        accion = advertencias[0] + " · " + accion_primaria

    return {
        **metricas,
        "wellness_norm": round(float(wellness_norm), 4),
        "carga_integrada_plan": round(float(carga_integrada_plan), 2),
        "indice_fatiga": round(indice, 1),
        "estado": estado,
        "color": color,
        "accion": accion,
        "accion_primaria": accion_primaria,
        "advertencias": advertencias,
        "contexto_cientifico": contexto_cientifico,
        "nota_swc": nota_swc,
    }
