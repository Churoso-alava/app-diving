import pytest
import numpy as np
import pandas as pd
from core.analysis import (
    matriz_spearman,
    cross_correlation_lag,
    umbrales_por_percentil,
    reporte_redundancias,
)

def _df_correlacionado(n=30, seed=0):
    """DataFrame con variables correlacionadas de manera conocida."""
    np.random.seed(seed)
    vmp = np.random.normal(1.2, 0.1, n)
    # delta_pct correlacionado negativamente con vmp (r ≈ -0.9)
    delta = -50 * (vmp - 1.2) + np.random.normal(0, 2, n)
    return pd.DataFrame({"vmp_hoy": vmp, "delta_pct": delta,
                          "acwr": np.random.uniform(0.8, 1.3, n),
                          "z_meso": np.random.normal(0, 1, n),
                          "wellness_norm": np.random.uniform(0.3, 1.0, n)})

def test_spearman_detecta_alta_correlacion():
    """delta_pct correlacionado con vmp → |rho| alto."""
    df = _df_correlacionado()
    mat = matriz_spearman(df, ["vmp_hoy", "delta_pct"])
    rho = mat.loc["vmp_hoy", "delta_pct"]
    assert abs(rho) > 0.70, f"Correlación alta esperada, obtenida rho={rho:.3f}"

def test_spearman_diagonal_es_uno():
    """Diagonal de la matriz Spearman siempre = 1.0."""
    df = _df_correlacionado()
    mat = matriz_spearman(df)
    for col in mat.columns:
        assert mat.loc[col, col] == pytest.approx(1.0)

def test_cross_correlation_retorna_lags_correctos():
    """Retorna dict con lags 0..max_lag."""
    np.random.seed(1)
    s1 = pd.Series(np.random.normal(0, 1, 50))
    s2 = pd.Series(np.random.normal(0, 1, 50))
    cc = cross_correlation_lag(s1, s2, max_lag=5)
    assert set(cc.keys()) == {0, 1, 2, 3, 4, 5}
    for v in cc.values():
        assert -1.0 <= v <= 1.0

def test_umbrales_por_percentil_retorna_valores_esperados():
    """Percentiles 10/25/75/90 de una serie conocida."""
    arr = pd.Series(range(1, 101))  # 1..100
    umbrales = umbrales_por_percentil(arr, [10, 25, 75, 90])
    assert umbrales[10] == pytest.approx(10.9, abs=0.5)
    assert umbrales[90] == pytest.approx(90.1, abs=0.5)

def test_reporte_redundancias_identifica_alta_correlacion():
    """
    Dos variables con |rho| > threshold → aparecen en el reporte de redundancias.
    """
    df = _df_correlacionado()
    reporte = reporte_redundancias(df, ["vmp_hoy", "delta_pct"], threshold=0.70)
    assert len(reporte) >= 1, "Debe detectar el par (vmp_hoy, delta_pct) como redundante"
    par = reporte[0]
    assert "vmp_hoy" in par["par"] and "delta_pct" in par["par"]
