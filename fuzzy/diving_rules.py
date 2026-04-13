"""
fuzzy_diving.py — Funciones de pertenencia y reglas difusas para carga en clavados.

Define las MFs para las variables CI (Carga Integrada) y wellness,
en el mismo formato que fuzzy.py existente, listas para alimentar
el motor Mamdani como entradas adicionales.
"""

from __future__ import annotations

import numpy as np
import skfuzzy as fuzz

# ---------------------------------------------------------------------------
# Universos de discurso
# ---------------------------------------------------------------------------

U_CI       = np.arange(0.0, 201.0, 0.5)      # Carga Integrada [0, 200]
U_WELLNESS = np.arange(0.0, 1.01, 0.01)      # Wellness normalizado [0, 1]

# ---------------------------------------------------------------------------
# Conjuntos difusos — CI (Carga Integrada)
# ---------------------------------------------------------------------------

CONJUNTOS_CI = ["RECUPERACION", "MANTENIMIENTO", "DESARROLLO", "SOBRECARGA"]

mf_ci: dict[str, callable] = {
    "RECUPERACION":  lambda x: fuzz.trapmf(np.atleast_1d(x), [0,   0,  25,  55]).item()
                               if np.isscalar(x) else fuzz.trapmf(x, [0,   0,  25,  55]),
    "MANTENIMIENTO": lambda x: fuzz.trapmf(np.atleast_1d(x), [35,  60,  90, 115]).item()
                               if np.isscalar(x) else fuzz.trapmf(x, [35,  60,  90, 115]),
    "DESARROLLO":    lambda x: fuzz.trapmf(np.atleast_1d(x), [90, 115, 145, 165]).item()
                               if np.isscalar(x) else fuzz.trapmf(x, [90, 115, 145, 165]),
    "SOBRECARGA":    lambda x: fuzz.trapmf(np.atleast_1d(x), [145, 165, 200, 200]).item()
                               if np.isscalar(x) else fuzz.trapmf(x, [145, 165, 200, 200]),
}

# Vectorizaciones sobre el universo (para visualizacion y motor Mamdani)
MF_CI_ARRAYS: dict[str, np.ndarray] = {
    "RECUPERACION":  fuzz.trapmf(U_CI, [0,   0,   25,  55]),
    "MANTENIMIENTO": fuzz.trapmf(U_CI, [35,  60,   90, 115]),
    "DESARROLLO":    fuzz.trapmf(U_CI, [90,  115, 145, 165]),
    "SOBRECARGA":    fuzz.trapmf(U_CI, [145, 165, 200, 200]),
}

# ---------------------------------------------------------------------------
# Conjuntos difusos — Wellness
# ---------------------------------------------------------------------------

CONJUNTOS_WELLNESS = ["DEFICIENTE", "ACEPTABLE", "OPTIMO"]

mf_wellness: dict[str, callable] = {
    "DEFICIENTE": lambda x: fuzz.trapmf(np.atleast_1d(x), [0.00, 0.00, 0.25, 0.45]).item()
                            if np.isscalar(x) else fuzz.trapmf(x, [0.00, 0.00, 0.25, 0.45]),
    "ACEPTABLE":  lambda x: fuzz.trimf(np.atleast_1d(x),  [0.30, 0.50, 0.70]).item()
                            if np.isscalar(x) else fuzz.trimf(x,  [0.30, 0.50, 0.70]),
    "OPTIMO":     lambda x: fuzz.trapmf(np.atleast_1d(x), [0.55, 0.75, 1.00, 1.00]).item()
                            if np.isscalar(x) else fuzz.trapmf(x, [0.55, 0.75, 1.00, 1.00]),
}

MF_WELLNESS_ARRAYS: dict[str, np.ndarray] = {
    "DEFICIENTE": fuzz.trapmf(U_WELLNESS, [0.00, 0.00, 0.25, 0.45]),
    "ACEPTABLE":  fuzz.trimf(U_WELLNESS,  [0.30, 0.50, 0.70]),
    "OPTIMO":     fuzz.trapmf(U_WELLNESS, [0.55, 0.75, 1.00, 1.00]),
}

# ---------------------------------------------------------------------------
# Reglas difusas adicionales (R24-R28)
# Formato: lista de dicts para integracion manual con el motor Mamdani existente.
# Cada regla es un dict con antecedentes y consecuente.
# ---------------------------------------------------------------------------

REGLAS_DIVING = [
    # R24: sobrecarga + wellness deficiente → fatiga critica
    {
        "id": "R24",
        "antecedentes": [("carga_integrada", "SOBRECARGA"), ("wellness", "DEFICIENTE")],
        "consecuente":  ("fatiga", "critico"),
        "operador":     "AND",
    },
    # R25: desarrollo + wellness deficiente → fatiga alta (fatiga_acumulada)
    {
        "id": "R25",
        "antecedentes": [("carga_integrada", "DESARROLLO"), ("wellness", "DEFICIENTE")],
        "consecuente":  ("fatiga", "fatiga_acumulada"),
        "operador":     "AND",
    },
    # R26: mantenimiento + wellness optimo → fatiga baja (optimo)
    {
        "id": "R26",
        "antecedentes": [("carga_integrada", "MANTENIMIENTO"), ("wellness", "OPTIMO")],
        "consecuente":  ("fatiga", "optimo"),
        "operador":     "AND",
    },
    # R27: recuperacion → fatiga muy baja (optimo maximo)
    {
        "id": "R27",
        "antecedentes": [("carga_integrada", "RECUPERACION")],
        "consecuente":  ("fatiga", "optimo"),
        "operador":     "AND",
    },
    # R28: sobrecarga + wellness optimo → fatiga moderada (alerta_temprana)
    {
        "id": "R28",
        "antecedentes": [("carga_integrada", "SOBRECARGA"), ("wellness", "OPTIMO")],
        "consecuente":  ("fatiga", "alerta_temprana"),
        "operador":     "AND",
    },
]


# ---------------------------------------------------------------------------
# Helpers de activacion (para uso en pipeline)
# ---------------------------------------------------------------------------

def activar_ci(ci_valor: float) -> dict[str, float]:
    """Retorna grado de pertenencia de ci_valor a cada conjunto CI."""
    return {c: mf_ci[c](ci_valor) for c in CONJUNTOS_CI}


def activar_wellness(w_valor: float) -> dict[str, float]:
    """Retorna grado de pertenencia de w_valor a cada conjunto wellness."""
    return {c: mf_wellness[c](w_valor) for c in CONJUNTOS_WELLNESS}


def conjunto_dominante_ci(ci_valor: float) -> str:
    """Retorna el conjunto CI con mayor grado de pertenencia."""
    grados = activar_ci(ci_valor)
    return max(grados, key=grados.get)


def conjunto_dominante_wellness(w_valor: float) -> str:
    """Retorna el conjunto wellness con mayor grado de pertenencia."""
    grados = activar_wellness(w_valor)
    return max(grados, key=grados.get)
