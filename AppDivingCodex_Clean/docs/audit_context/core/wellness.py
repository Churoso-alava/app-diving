"""
Wellness (Hooper modificado) para evaluación pre-entrenamiento.
"""

from __future__ import annotations
import pandas as pd

# Pesos basados en Saw et al. (2016) Int J Sports Physiol Perform y
# Buchheit (2014) Sport Med: sueño y fatiga tienen mayor correlación con VMP
# y recuperación del SNC que estrés, dolor y humor.
#
# Item   | Peso | Justificación
# sueno  | 0.30 | Mayor correlación con VMP y recuperación SNC
# fatiga | 0.25 | Indicador directo de acumulación neuromuscular
# estres | 0.20 | Modulador del eje HPA (carga psicológica)
# dolor  | 0.15 | Señal periférica músculo-esquelética
# humor  | 0.10 | Indicador tardío, alta varianza individual
_PESOS_HOOPER: dict[str, float] = {
    "sueno": 0.30, "fatiga": 0.25, "estres": 0.20, "dolor": 0.15, "humor": 0.10
}


def calcular_wellness(
    sueno: int, fatiga: int, estres: int, dolor: int, humor: int
) -> float:
    """
    Retorna wellness normalizado en [0.0, 1.0].

    Cada ítem Likert 1-7 se normaliza:
    - Para sueno/fatiga/estres/dolor: 1=óptimo, 7=pésimo → (7-x)/6
    - Para humor: 7=óptimo, 1=pésimo → (x-1)/6

    Los pesos diferenciados (_PESOS_HOOPER) reflejan la validez predictiva
    de cada ítem sobre el rendimiento neuromuscular.

    Nota metodológica: Los ítems son ordinales (Likert). Se tratan como
    cuasi-intervalo (aceptable para escalas de 7 niveles, Norman 2010).
    Verificar Cronbach α ≥ 0.70 con cronbach_alpha_wellness() antes de
    usar wellness_norm como variable continua en análisis externos.
    """
    items = {"sueno": sueno, "fatiga": fatiga, "estres": estres, "dolor": dolor, "humor": humor}
    for nombre, val in items.items():
        if not (1 <= val <= 7):
            raise ValueError(f"Ítem '{nombre}={val}' fuera del rango [1, 7]")

    w = {
        "sueno":  (7 - sueno)  / 6.0,
        "fatiga": (7 - fatiga) / 6.0,
        "estres": (7 - estres) / 6.0,
        "dolor":  (7 - dolor)  / 6.0,
        "humor":  (humor - 1)  / 6.0,
    }
    return sum(_PESOS_HOOPER[k] * w[k] for k in _PESOS_HOOPER)


def cronbach_alpha_wellness(items_df: pd.DataFrame) -> float:
    """
    Calcula Cronbach's alpha para los 5 ítems del Hooper.

    items_df: DataFrame con columnas ['sueno','fatiga','estres','dolor','humor'],
              valores de 1 a 7, una fila por observación.

    Retorna alpha en (-inf, 1]. Interpretación:
      α ≥ 0.70 → fiabilidad aceptable → wellness_norm como variable continua es defensible
      α < 0.70 → revisar qué ítem reduce la coherencia (eliminar o reponderar)

    Uso: ejecutar con datos históricos del equipo (mínimo 20 observaciones).
    """
    k = items_df.shape[1]
    if k < 2:
        raise ValueError("Se necesitan al menos 2 ítems para calcular alpha")
    var_items = items_df.var(axis=0, ddof=1).sum()
    var_total  = items_df.sum(axis=1).var(ddof=1)
    if var_total == 0:
        return 1.0
    return float((k / (k - 1)) * (1.0 - var_items / var_total))
