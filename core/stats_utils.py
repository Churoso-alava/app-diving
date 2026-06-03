"""
Módulo estadístico adaptativo.

Regla central: el estimador se adapta a la distribución de los datos.
- Normal (Shapiro-Wilk p > 0.05, n >= 8) → media + SD (estimadores de mínima varianza)
- No-normal o n < 8              → mediana + MAD×1.4826 (robustos, sin supuesto distribucional)

Para pendientes temporales se usa Theil-Sen (mediana de pendientes por pares)
en lugar de OLS: no requiere normalidad de residuos y es resistente a outliers.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import shapiro, theilslopes
from scipy.stats import median_abs_deviation as _mad_fn

_GAP_MIN: float = 0.5   # sesiones/día mínimo para escalado de pendiente
_GAP_MAX: float = 3.0   # sesiones/día máximo


def estimar_centro_dispersion(
    series: pd.Series,
    min_n_shapiro: int = 8,
) -> tuple[float, float]:
    """
    Retorna (centro, dispersión) adaptado a la distribución de `series`.

    Parameters
    ----------
    series       : pd.Series con valores numéricos (NaN ignorados)
    min_n_shapiro: mínimo de puntos para ejecutar Shapiro-Wilk (default 8)

    Returns
    -------
    (centro, dispersión):
        - Normal     → (media, SD)
        - No-normal  → (mediana, MAD * 1.4826)
        El factor 1.4826 hace MAD consistente con σ para distribución normal.
    """
    arr = series.dropna().values
    n = len(arr)

    if n < 2:
        return float(arr[0]) if n == 1 else 0.0, 0.0

    # Decidir normalidad
    if n >= min_n_shapiro:
        # Some arrays might be constant, which throws a UserWarning in shapiro and returns an invalid p-value.
        # But we let it pass. If constant, mean=median, std=mad=0.
        try:
            _, p_sw = shapiro(arr)
            es_normal = p_sw > 0.05
        except Exception:
            es_normal = False
    else:
        es_normal = False  # n pequeño → no confiar en el test, usar robusto

    if es_normal:
        return float(np.mean(arr)), float(np.std(arr, ddof=1))
    else:
        mad_scaled = float(_mad_fn(arr, scale="normal"))  # × 1.4826 internamente
        return float(np.median(arr)), mad_scaled


def pendiente_theil_sen(
    vmp_series: pd.Series,
    min_n: int = 4,
    alpha_ic: float = 0.90,
    gap_min: float = _GAP_MIN,
    gap_max: float = _GAP_MAX,
) -> float:
    """
    Pendiente robusta de VMP en el tiempo usando el estimador Theil-Sen.

    Theil-Sen = mediana de todas las pendientes por pares posibles.
    Es el equivalente no-paramétrico del slope OLS: sin supuesto de normalidad,
    resistente a outliers (breakdown point 29%).

    Retorna 0.0 si:
      - n < min_n, o
      - el IC (1-alpha_ic) del slope incluye 0  →  tendencia no distinguible de ruido

    Unidades: m/s por sesión (escalado por gap promedio en días entre sesiones).
    """
    win = vmp_series.dropna()
    if len(win) < min_n:
        return 0.0

    x = (win.index - win.index[0]).days.values.astype(float)
    # theilslopes returns: slope, intercept, low_slope, high_slope
    result = theilslopes(win.values, x, alpha=alpha_ic)

    # IC incluye 0 → no significativo
    if result[2] <= 0.0 <= result[3]:
        return 0.0

    avg_gap = float(np.clip(x[-1] / max(len(x) - 1, 1), gap_min, gap_max))
    return float(np.clip(result[0] * avg_gap, -0.25, 0.25))


def calcular_ewma(series: pd.Series, span: int) -> pd.Series:
    """
    Calcula el promedio móvil ponderado exponencialmente (EWMA).
    
    Se usa para ACWR de carga y rendimiento por ser más sensible a 
    cambios agudos que el promedio rodante simple (RA).
    
    Formula: alpha = 2 / (span + 1)
    """
    # Usar adjust=False para formula recursiva pura (S_t = a*Y_t + (1-a)*S_{t-1})
    # y ignore_na=False para que nulos propaguen o sean manejados por fillna previo.
    return series.ewm(span=span, adjust=False, ignore_na=False).mean()
