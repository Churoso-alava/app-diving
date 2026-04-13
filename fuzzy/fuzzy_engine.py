"""
fuzzy.py — Motor Fuzzy Mamdani v4.1
Sin dependencias de Streamlit ni base de datos.
"""
import logging

import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

log = logging.getLogger(__name__)


# =============================================================================
#  UNIVERSOS Y FUNCIONES DE PERTENENCIA
# =============================================================================

def construir_sistema_fuzzy():
    """Retorna la tupla de variables difusas (antecedentes + consecuente)."""
    u_acwr  = np.arange(0.50, 1.81,  0.01)
    u_delta = np.arange(-20,  41,    0.5)
    u_zmeso = np.arange(-4,   4.1,   0.1)
    u_ba    = np.arange(-0.25, 0.26,  0.001)
    u_b28   = np.arange(-0.25, 0.26,  0.001)
    u_fat   = np.arange(0, 101, 1)

    acwr_v  = ctrl.Antecedent(u_acwr,  "acwr")
    delta_v = ctrl.Antecedent(u_delta, "delta_pct")
    zmeso_v = ctrl.Antecedent(u_zmeso, "z_meso")
    ba_v    = ctrl.Antecedent(u_ba,    "beta_aguda")
    b28_v   = ctrl.Antecedent(u_b28,   "beta_28")
    fat_v   = ctrl.Consequent(u_fat,   "fatiga")

    acwr_v["bajo"]     = fuzz.trapmf(u_acwr, [0.50, 0.50, 0.70, 0.85])
    acwr_v["optimo"]   = fuzz.trapmf(u_acwr, [0.78, 0.88, 1.20, 1.30])
    acwr_v["alto"]     = fuzz.trimf (u_acwr, [1.25, 1.40, 1.55])
    acwr_v["excesivo"] = fuzz.trapmf(u_acwr, [1.45, 1.55, 1.80, 1.80])

    delta_v["ganancia"]   = fuzz.trapmf(u_delta, [-20, -20, -5,  0])
    delta_v["tolerable"]  = fuzz.trapmf(u_delta, [ -2,   0,  8, 12])
    delta_v["vigilancia"] = fuzz.trimf (u_delta, [ 10,  15, 22])
    delta_v["alarma"]     = fuzz.trapmf(u_delta, [ 18,  22, 40, 40])

    zmeso_v["muy_bajo"] = fuzz.trapmf(u_zmeso, [-4.0, -4.0, -2.2, -1.5])
    zmeso_v["bajo"]     = fuzz.trimf (u_zmeso, [-2.0, -1.2, -0.5])
    zmeso_v["normal"]   = fuzz.trimf (u_zmeso, [-0.8,  0.0,  0.8])
    zmeso_v["elevado"]  = fuzz.trapmf(u_zmeso, [ 0.6,  1.2,  4.0,  4.0])

    ba_v["neg_fuerte"]   = fuzz.trapmf(u_ba, [-0.25, -0.25, -0.06, -0.025])
    ba_v["neg_moderada"] = fuzz.trimf (u_ba, [-0.04, -0.015, -0.005])
    ba_v["estable"]      = fuzz.trimf (u_ba, [-0.008,  0.0,   0.008])
    ba_v["positiva"]     = fuzz.trapmf(u_ba, [ 0.005,  0.025, 0.25,  0.25])

    b28_v["deterioro"] = fuzz.trapmf(u_b28, [-0.25, -0.25, -0.03, -0.01])
    b28_v["estable"]   = fuzz.trimf (u_b28, [-0.012,  0.0,   0.012])
    b28_v["mejora"]    = fuzz.trapmf(u_b28, [ 0.010,  0.03,  0.25,  0.25])

    fat_v["critico"]          = fuzz.trapmf(u_fat, [  0,  0, 18, 28])
    fat_v["fatiga_acumulada"] = fuzz.trimf (u_fat, [ 25, 37, 52])
    fat_v["alerta_temprana"]  = fuzz.trimf (u_fat, [ 50, 62, 76])
    fat_v["optimo"]           = fuzz.trapmf(u_fat, [ 75, 88, 100, 100])

    return acwr_v, delta_v, zmeso_v, ba_v, b28_v, fat_v


# =============================================================================
#  REGLAS MAMDANI v4.1 — 23 reglas
# =============================================================================

def construir_reglas(acwr_v, delta_v, zmeso_v, ba_v, b28_v, fat_v):
    return [
        # ── BLOQUE CRÍTICO (4 reglas) ─────────────────────────────────────────
        ctrl.Rule(acwr_v["bajo"] & delta_v["alarma"] & ba_v["neg_fuerte"],          fat_v["critico"]),         # R01
        ctrl.Rule(acwr_v["bajo"] & b28_v["deterioro"] & ba_v["neg_fuerte"],         fat_v["critico"]),         # R02
        ctrl.Rule(delta_v["alarma"] & b28_v["deterioro"] & zmeso_v["muy_bajo"],     fat_v["critico"]),         # R03
        ctrl.Rule(acwr_v["excesivo"] & zmeso_v["muy_bajo"] & ba_v["neg_fuerte"],    fat_v["critico"]),         # R04

        # ── BLOQUE FATIGA ACUMULADA (8 reglas) ───────────────────────────────
        ctrl.Rule(acwr_v["bajo"] & delta_v["alarma"],                               fat_v["fatiga_acumulada"]),# R05
        ctrl.Rule(acwr_v["bajo"] & delta_v["vigilancia"] & b28_v["deterioro"],      fat_v["fatiga_acumulada"]),# R06
        ctrl.Rule(acwr_v["optimo"] & ba_v["neg_fuerte"] & zmeso_v["muy_bajo"],      fat_v["fatiga_acumulada"]),# R07
        ctrl.Rule(delta_v["alarma"] & ba_v["neg_moderada"],                         fat_v["fatiga_acumulada"]),# R08
        ctrl.Rule(acwr_v["alto"] & delta_v["vigilancia"] & b28_v["deterioro"],      fat_v["fatiga_acumulada"]),# R09
        ctrl.Rule(acwr_v["excesivo"] & zmeso_v["normal"],                           fat_v["fatiga_acumulada"]),# R10
        ctrl.Rule(
            b28_v["deterioro"] & (ba_v["neg_fuerte"] | ba_v["neg_moderada"])
            & (zmeso_v["bajo"] | zmeso_v["muy_bajo"]),                              fat_v["fatiga_acumulada"]),# R11
        ctrl.Rule(
            acwr_v["excesivo"] & (zmeso_v["bajo"] | zmeso_v["muy_bajo"])
            & ba_v["neg_moderada"],                                                 fat_v["fatiga_acumulada"]),# R12

        # ── BLOQUE ALERTA TEMPRANA (6 reglas) ────────────────────────────────
        ctrl.Rule(delta_v["vigilancia"] & ba_v["neg_moderada"] & zmeso_v["normal"], fat_v["alerta_temprana"]), # R13
        ctrl.Rule(acwr_v["optimo"] & zmeso_v["bajo"] & delta_v["vigilancia"],       fat_v["alerta_temprana"]), # R14
        ctrl.Rule(acwr_v["alto"] & ba_v["estable"] & delta_v["tolerable"],          fat_v["alerta_temprana"]), # R15
        ctrl.Rule(b28_v["deterioro"] & delta_v["tolerable"] & zmeso_v["bajo"],      fat_v["alerta_temprana"]), # R16
        ctrl.Rule(ba_v["neg_fuerte"] & b28_v["estable"] & zmeso_v["normal"],        fat_v["alerta_temprana"]), # R17
        ctrl.Rule(ba_v["positiva"] & zmeso_v["bajo"],                               fat_v["alerta_temprana"]), # R18

        # ── BLOQUE ÓPTIMO (5 reglas) ──────────────────────────────────────────
        ctrl.Rule(acwr_v["optimo"] & delta_v["tolerable"] & ba_v["positiva"] & b28_v["mejora"], fat_v["optimo"]), # R19
        ctrl.Rule(acwr_v["optimo"] & delta_v["ganancia"] & b28_v["estable"],        fat_v["optimo"]),            # R20
        ctrl.Rule(acwr_v["optimo"] & zmeso_v["normal"] & ba_v["estable"] & delta_v["tolerable"], fat_v["optimo"]),# R21
        ctrl.Rule(
            (acwr_v["bajo"] | acwr_v["optimo"]) & zmeso_v["elevado"]
            & (ba_v["estable"] | ba_v["positiva"]),                                 fat_v["optimo"]),            # R22
        ctrl.Rule(acwr_v["optimo"] & zmeso_v["elevado"] & b28_v["mejora"],          fat_v["optimo"]),            # R23
    ]


def construir_motor_fuzzy():
    """Construye y retorna (vars_tuple, simulador). Cachear en app.py con @st.cache_resource."""
    vars_tuple = construir_sistema_fuzzy()
    acwr_v, delta_v, zmeso_v, ba_v, b28_v, fat_v = vars_tuple
    reglas    = construir_reglas(acwr_v, delta_v, zmeso_v, ba_v, b28_v, fat_v)
    sistema   = ctrl.ControlSystem(reglas)
    simulador = ctrl.ControlSystemSimulation(sistema)
    log.info("Motor fuzzy Mamdani v4.1 construido (%d reglas).", len(reglas))
    return vars_tuple, simulador


# =============================================================================
#  EVALUADOR — lógica de decisión post-motor
# =============================================================================

def evaluar_atleta(simulador, metricas: dict) -> dict:
    """
    Corre el motor difuso aplicando el filtro SWC pre-motor (v4.1).

    FILTRO SWC [C3]:
    Si es_ruido_biologico=True, delta_pct se envía como 0.0 al motor para
    neutralizar señales falsas dentro del ruido biológico inter-sesión normal
    (CV% CMJ 4–8%; Jukic 2022; Alba-Jiménez 2022).
    """
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
        simulador.input["acwr"]       = metricas_fuzzy["acwr"]
        simulador.input["delta_pct"]  = metricas_fuzzy["delta_pct"]
        simulador.input["z_meso"]     = metricas_fuzzy["z_meso"]
        simulador.input["beta_aguda"] = metricas_fuzzy["beta_aguda"]
        simulador.input["beta_28"]    = metricas_fuzzy["beta_28"]
        simulador.compute()
        indice = simulador.output["fatiga"]
    except Exception as exc:
        log.warning("Motor fuzzy falló para atleta '%s': %s — usando fallback 50.0",
                    metricas.get("atleta", "?"), exc)
        indice = 50.0

    if   indice >= 75:
        estado, color           = "🟢 ÓPTIMO",           "#16a34a"
        accion_primaria         = "Entrenamiento normal"
        contexto_cientifico     = "Posible progresión de carga. Mantener monitoreo de tendencia."
    elif indice >= 50:
        estado, color           = "🟡 ALERTA TEMPRANA",  "#ca8a04"
        accion_primaria         = "Reducir carga 10–15%"
        contexto_cientifico     = "Monitoreo estrecho próximas 48 h. Re-evaluar si persiste caída."
    elif indice >= 25:
        estado, color           = "🟠 FATIGA ACUMULADA", "#ea580c"
        accion_primaria         = "Sesión regenerativa únicamente"
        contexto_cientifico     = "Sin carga intensa. Priorizar recuperación activa y sueño."
    else:
        estado, color           = "🔴 CRÍTICO",           "#dc2626"
        accion_primaria         = "Descanso obligatorio"
        contexto_cientifico     = "Evaluación médica si el estado persiste >24 h."

    advertencias: list[str] = []

    if metricas.get("edad_atleta", 18) < 15 and metricas["delta_pct"] > 20:
        advertencias.append("🚨 Estrés pediátrico — revisar carga semanal")

    calidad        = metricas.get("calidad_dato", "alta")
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

    log.debug("Atleta %s → índice %.1f (%s)", metricas.get("atleta"), indice, estado)

    return {
        **metricas,
        "indice_fatiga":       round(indice, 1),
        "estado":              estado,
        "color":               color,
        "accion":              accion,
        "accion_primaria":     accion_primaria,
        "advertencias":        advertencias,
        "contexto_cientifico": contexto_cientifico,
        "nota_swc":            nota_swc,
    }
