"""
Módulo de análisis bivariado para calibrar variables y membresías del motor difuso.

Funciones principales:
  - matriz_spearman(): correlaciones entre variables (Spearman, no Pearson)
  - cross_correlation_lag(): ACWR[t-lag] → VMP[t] para detectar retrasos temporales
  - umbrales_por_percentil(): calibración empírica de membresías difusas
  - reporte_redundancias(): identifica pares con alta correlación

Por qué Spearman y no Pearson:
  - VMP puede ser no-normal (distribuciones asimétricas en bloques de carga)
  - wellness_norm es técnicamente ordinal (Likert 1-7)
  - Spearman captura relaciones monotónicas no lineales
  - Para variables ordinales Pearson viola el supuesto de escala de intervalo
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from statsmodels.tsa.stattools import ccf as _ccf


_COLS_DEFAULT = [
    "vmp_hoy", "acwr", "delta_pct", "z_meso",
    "beta_aguda", "beta_28", "wellness_norm",
]


def matriz_spearman(
    df: pd.DataFrame,
    columnas: list[str] | None = None,
) -> pd.DataFrame:
    """
    Matriz de correlación de Spearman entre columnas numéricas.

    Usa Spearman porque las variables pueden ser no normales u ordinales.
    Filas/columnas con NaN son excluidas pairwise.

    Returns
    -------
    pd.DataFrame simétrico con rho redondeado a 3 decimales.
    """
    cols = columnas or [c for c in _COLS_DEFAULT if c in df.columns]
    df_num = df[cols].dropna()
    n = len(cols)
    matrix = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            rho, _ = spearmanr(df_num.iloc[:, i], df_num.iloc[:, j])
            matrix[i, j] = round(float(rho), 3)
            matrix[j, i] = round(float(rho), 3)
    return pd.DataFrame(matrix, index=cols, columns=cols)


def cross_correlation_lag(
    serie_x: pd.Series,
    serie_y: pd.Series,
    max_lag: int = 7,
) -> dict[int, float]:
    """
    Correlación cruzada entre serie_x (predictor) y serie_y (respuesta)
    para lags 0..max_lag.

    Uso típico: cross_correlation_lag(acwr_series, vmp_series, max_lag=7)
    → muestra si ACWR de hace k sesiones predice el VMP actual.

    Returns
    -------
    dict {lag: correlacion} con valores en [-1, 1].
    """
    x = serie_x.dropna().values
    y = serie_y.dropna().values
    min_n = min(len(x), len(y))
    x, y = x[:min_n], y[:min_n]
    raw = _ccf(x, y, nlags=max_lag, adjusted=True)
    return {lag: round(float(raw[lag]), 3) for lag in range(max_lag + 1)}


def umbrales_por_percentil(
    series: pd.Series,
    percentiles: list[float] = (10, 25, 75, 90),
) -> dict[float, float]:
    """
    Umbrales empíricos para calibrar funciones de membresía difusa.

    Sustituye umbrales arbitrarios por valores derivados de los datos del equipo.

    Ejemplo de uso para calibrar vmp_v["baja"]:
        u = umbrales_por_percentil(df["vmp_hoy"])
        # Usar u[10] como límite inferior, u[25] como núcleo inferior,
        # u[75] como núcleo superior, u[90] como límite superior.

    Returns
    -------
    {percentil: valor} con los puntos de corte empíricos.
    """
    arr = series.dropna().values
    return {float(p): float(np.percentile(arr, p)) for p in percentiles}


def reporte_redundancias(
    df: pd.DataFrame,
    columnas: list[str] | None = None,
    threshold: float = 0.70,
) -> list[dict]:
    """
    Identifica pares de variables con |rho_Spearman| > threshold.

    Alta correlación entre dos antecedentes del motor difuso indica redundancia:
    ambas variables activan las mismas reglas con información casi idéntica,
    inflando el peso implícito del constructo que representan.

    Returns
    -------
    Lista de dicts: [{"par": (var1, var2), "rho": valor, "interpretacion": str}]
    Ordenada de mayor a menor |rho|.
    """
    mat = matriz_spearman(df, columnas)
    cols = mat.columns.tolist()
    redundancias = []
    for i, c1 in enumerate(cols):
        for j in range(i + 1, len(cols)):
            c2 = cols[j]
            rho = mat.loc[c1, c2]
            if abs(rho) >= threshold:
                nivel = "MUY ALTA" if abs(rho) >= 0.85 else "ALTA"
                redundancias.append({
                    "par": (c1, c2),
                    "rho": rho,
                    "interpretacion": (
                        f"{nivel} correlación Spearman ({rho:+.3f}). "
                        f"Considerar eliminar '{c2}' si '{c1}' ya está en el modelo."
                    ),
                })
    return sorted(redundancias, key=lambda x: -abs(x["rho"]))
