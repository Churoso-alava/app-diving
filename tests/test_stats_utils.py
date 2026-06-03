import pytest
import numpy as np
import pandas as pd
from core.stats_utils import estimar_centro_dispersion, pendiente_theil_sen

class TestEstimadorCentroDispersion:
    def test_datos_normales_usa_media_sd(self):
        """Con n>=8 y datos normales (Shapiro p>0.05), retorna media y SD."""
        np.random.seed(0)
        arr = pd.Series(np.random.normal(1.2, 0.05, 20))
        centro, dispersion = estimar_centro_dispersion(arr)
        assert abs(centro - arr.mean()) < 1e-9, "Normal → debe usar media"
        assert abs(dispersion - arr.std(ddof=1)) < 1e-9, "Normal → debe usar SD"

    def test_datos_no_normales_usa_mediana_mad(self):
        """Con distribución muy asimétrica (Shapiro falla), retorna mediana y MAD."""
        # Serie bimodal / muy asimétrica — Shapiro rechazará normalidad
        arr = pd.Series([1.2]*10 + [2.5]*2 + [0.3]*2)
        centro, dispersion = estimar_centro_dispersion(arr)
        assert abs(centro - arr.median()) < 1e-9, "No-normal → debe usar mediana"
        # MAD scaled != SD para distribución bimodal
        from scipy.stats import median_abs_deviation
        mad_ref = median_abs_deviation(arr.values, scale="normal")
        assert abs(dispersion - mad_ref) < 1e-9, "No-normal → debe usar MAD"

    def test_n_menor_8_usa_mediana_mad_sin_shapiro(self):
        """Con n<8 no se puede hacer Shapiro; usa mediana/MAD directamente."""
        arr = pd.Series([1.1, 1.2, 1.3, 1.0, 1.15])  # n=5
        centro, dispersion = estimar_centro_dispersion(arr)
        assert abs(centro - arr.median()) < 1e-9

    def test_dispersion_nunca_negativa(self):
        """MAD y SD siempre >= 0."""
        arr = pd.Series([1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2])
        _, dispersion = estimar_centro_dispersion(arr)
        assert dispersion >= 0.0

class TestPendienteTheilSen:
    def test_tendencia_negativa_clara_es_negativa(self):
        """Caída lineal perfecta → pendiente negativa significativa."""
        vmps = pd.Series([1.5, 1.42, 1.35, 1.28, 1.21, 1.14, 1.07],
                          index=pd.date_range("2025-01-01", periods=7))
        slope = pendiente_theil_sen(vmps, min_n=4)
        assert slope < -0.01

    def test_datos_ruidosos_retorna_cero(self):
        """Oscilación sin tendencia → IC incluye 0 → retorna 0.0."""
        np.random.seed(99) # Seed estable para evitar fallos aleatorios
        vals = 1.2 + np.random.uniform(-0.05, 0.05, 7)
        vmps = pd.Series(vals, index=pd.date_range("2025-01-01", periods=7))
        slope = pendiente_theil_sen(vmps, min_n=4)
        assert slope == 0.0

    def test_n_insuficiente_retorna_cero(self):
        """Con menos de min_n puntos, retorna 0.0."""
        vmps = pd.Series([1.2, 1.1], index=pd.date_range("2025-01-01", periods=2))
        slope = pendiente_theil_sen(vmps, min_n=4)
        assert slope == 0.0

    def test_outlier_no_distorsiona_pendiente(self):
        """Theil-Sen es resistente a outliers; un outlier extremo no cambia la dirección."""
        # Tendencia bajando, con un outlier extremo en el día 4
        vmps = pd.Series([1.40, 1.35, 1.30, 5.00, 1.20, 1.15, 1.10],
                          index=pd.date_range("2025-01-01", periods=7))
        slope_ts = pendiente_theil_sen(vmps, min_n=4)
        # OLS se dispararía hacia arriba; Theil-Sen debe seguir siendo negativo
        assert slope_ts < 0.0, f"Theil-Sen debe ser negativo pese al outlier, fue {slope_ts}"
