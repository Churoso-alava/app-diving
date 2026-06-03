# Fuzzy Model — Auditoría y Corrección v2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Corregir los problemas estructurales del motor difuso Mamdani v4.2: doble contabilización del wellness, estimadores paramétricos incondicionales (Shapiro-Wilk no controla el flujo), slopes OLS sin robustez a no-normalidad, reglas contradictorias, subponderación de vmp_hoy, y ausencia total de análisis bivariado que justifique las membresías.

**Architecture:** Cuatro capas de corrección en orden de dependencia: (0) `stats_utils.py` — módulo estadístico adaptativo central que todos los demás consumen; (1) `services.py` — cálculo de métricas usando stats_utils; (2) `fuzzy_engine.py` — reglas y membresías; (3) `wellness.py` + `analysis.py` — fiabilidad del índice subjetivo y análisis bivariado empírico para calibrar membresías.

**Tech Stack:** Python 3.11+, scikit-fuzzy 0.4, scipy 1.11, numpy, pandas, statsmodels, pytest 7

---

## Mapa de archivos

| Archivo | Rol | Tareas |
|---|---|---|
| `stats_utils.py` | NUEVO — estimador adaptativo (Shapiro → mean/median, SD/MAD, Theil-Sen) | 0 |
| `services.py` | ACWR rolling, delta_pct, z_meso, beta, DQI, carga_integrada | 1, 2, 3, 6 |
| `fuzzy_engine.py` | Membresías y reglas Mamdani | 4, 5 |
| `wellness.py` | Índice Hooper: pesos + Cronbach's alpha | 7 |
| `analysis.py` | NUEVO — Spearman matrix, cross-correlation, umbrales empíricos | 8 |
| `tests/test_stats_utils.py` | Tests del módulo estadístico | 0 |
| `tests/test_services.py` | Tests unitarios de servicios | 1, 2, 3, 6 |
| `tests/test_fuzzy_engine.py` | Tests del motor difuso | 4, 5 |
| `tests/test_wellness.py` | Tests del índice de wellness y Cronbach | 7 |
| `tests/test_analysis.py` | Tests del módulo de análisis bivariado | 8 |
| `tests/test_integration.py` | Pipeline end-to-end | 9 |

---

## Task 0: Crear stats_utils.py — módulo estadístico adaptativo

**Problema raíz:** El Shapiro-Wilk ya se calcula en `services.py` pero nunca controla qué estimador se usa. Toda la lógica estadística usa media/SD incondicional. Este módulo centraliza la decisión paramétrico/no-paramétrico para que todos los demás módulos lo consuman con una sola llamada.

**Decisiones de diseño:**
- Normalidad → `(media, SD)` porque son estimadores de mínima varianza para datos normales
- No-normalidad o n<8 → `(mediana, MAD×1.4826)` donde el factor 1.4826 hace el MAD consistente con σ para distribuciones normales
- Theil-Sen como reemplazo de OLS slope: es el estimador de pendiente equivalente a la mediana, sin supuesto de normalidad de residuos

**Files:**
- Create: `stats_utils.py`
- Create: `tests/test_stats_utils.py`

- [ ] **Step 1: Escribir los tests antes del módulo**

```python
# tests/test_stats_utils.py
import pytest
import numpy as np
import pandas as pd
from stats_utils import estimar_centro_dispersion, pendiente_theil_sen

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
        np.random.seed(42)
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
```

- [ ] **Step 2: Correr tests — todos deben FALLAR (módulo no existe)**

```bash
pytest tests/test_stats_utils.py -v
```

- [ ] **Step 3: Crear stats_utils.py**

```python
# stats_utils.py
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
        _, p_sw = shapiro(arr)
        es_normal = p_sw > 0.05
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
    result = theilslopes(win.values, x, alpha=alpha_ic)

    # IC incluye 0 → no significativo
    if result.low_slope <= 0.0 <= result.high_slope:
        return 0.0

    avg_gap = float(np.clip(x[-1] / max(len(x) - 1, 1), gap_min, gap_max))
    return float(np.clip(result.slope * avg_gap, -0.25, 0.25))
```

- [ ] **Step 4: Correr tests — todos deben PASAR**

```bash
pytest tests/test_stats_utils.py -v
```

- [ ] **Step 5: Commit**

```bash
git add stats_utils.py tests/test_stats_utils.py
git commit -m "feat: crear stats_utils.py — módulo estadístico adaptativo

estimar_centro_dispersion(): Shapiro-Wilk controla mean/SD vs mediana/MAD.
pendiente_theil_sen(): slope no-paramétrico, IC 90% para significancia.
Reemplaza el uso incondicional de mean/SD y OLS en todo el pipeline."
```

---

## Task 1: Eliminar doble contabilización del wellness en carga_integrada_plan

**Problema:** El factor `(2.0 - wellness_norm)` en `services.py` bake el wellness dentro de `carga_integrada_plan` antes de entrar al motor. Luego `wellness_norm` entra también como antecedente independiente. El efecto del wellness se aplica dos veces; para wellness=0.0 la carga se duplica.

**Files:**
- Modify: `services.py` (bloque `if clavados_planificados:`)
- Create: `tests/test_services.py`

- [ ] **Step 1: Escribir el test que expone el bug**

```python
# tests/test_services.py
import pytest
import pandas as pd
import numpy as np
from services import calcular_metricas

def _df_atleta(n=15, vmp_base=1.2):
    fechas = pd.date_range("2025-01-01", periods=n, freq="D").strftime("%Y-%m-%d").tolist()
    vmps = [vmp_base + 0.01 * i for i in range(n)]
    return pd.DataFrame({"nombre": ["Ana"] * n, "fecha": fechas, "vmp_hoy": vmps})

def test_carga_integrada_no_depende_de_wellness():
    """
    carga_integrada_plan debe depender SOLO de clavados_planificados.
    El wellness entra al motor difuso por su propio canal.
    Con el bug: wellness=pésimo → carga ≈2× que wellness=óptimo.
    Con el fix: carga idéntica para ambos wellness con mismos clavados.
    """
    df = _df_atleta()
    clavados = [{"altura": 10.0, "dd": 2.5, "tipo": "PIKE"}]

    res_optimo = calcular_metricas(df, "Ana", clavados_planificados=clavados,
                                   wellness_respuestas={"sueno":1,"fatiga":1,"estres":1,"dolor":1,"humor":7})
    res_pesimo = calcular_metricas(df, "Ana", clavados_planificados=clavados,
                                   wellness_respuestas={"sueno":7,"fatiga":7,"estres":7,"dolor":7,"humor":1})

    assert abs(res_optimo["carga_integrada_plan"] - res_pesimo["carga_integrada_plan"]) < 0.01, (
        f"carga varía con wellness: {res_optimo['carga_integrada_plan']:.2f} vs "
        f"{res_pesimo['carga_integrada_plan']:.2f}. El wellness no debe modificar la carga."
    )
```

- [ ] **Step 2: Correr el test — debe FALLAR**

```bash
pytest tests/test_services.py::test_carga_integrada_no_depende_de_wellness -v
```

- [ ] **Step 3: Aplicar el fix en services.py**

```python
# ANTES:
if clavados_planificados:
    carga_bruta_plan = carga_bruta_sesion(clavados_planificados)
    carga_integrada_plan = normalizar_carga(carga_bruta_plan) * (2.0 - float(wellness_norm))
else:
    carga_integrada_plan = 0.0

# DESPUÉS:
if clavados_planificados:
    carga_bruta_plan = carga_bruta_sesion(clavados_planificados)
    # wellness_norm entra al motor difuso por su propio antecedente.
    # Eliminado: factor (2.0 - wellness) que causaba doble contabilización.
    carga_integrada_plan = normalizar_carga(carga_bruta_plan)
else:
    carga_integrada_plan = 0.0
```

- [ ] **Step 4: Correr el test — debe PASAR**

```bash
pytest tests/test_services.py::test_carga_integrada_no_depende_de_wellness -v
```

- [ ] **Step 5: Commit**

```bash
git add services.py tests/test_services.py
git commit -m "fix: eliminar doble contabilización del wellness en carga_integrada_plan

Factor (2.0 - wellness_norm) removido. Wellness entra al motor difuso
solo como antecedente independiente wellness_v[]. El índice de fatiga
dejaba de representar el estado real para atletas con wellness bajo."
```

---

## Task 2: Reemplazar OLS con Theil-Sen para beta_aguda y beta_28

**Problema (nuevo, incorporado de la auditoría):** `_pendiente_calendar` usa `np.polyfit` (OLS). OLS asume normalidad de los residuos. Si el VMP no es normal (lo cual no se verifica), el slope puede estar distorsionado por outliers o distribuciones asimétricas. Además, el OLS no tiene criterio de significancia → cualquier slope no-cero activa reglas.

**Solución:** Theil-Sen estimator = mediana de todas las pendientes por pares. No asume distribución, breakdown point 29% (resistente a outliers). El IC del slope reemplaza el p-value: si el IC incluye 0, la pendiente no es distinguible de ruido.

**Por qué no Spearman:** Spearman mide la fuerza de asociación monotónica (adimensional, [-1,1]). No da la velocidad de cambio en m/s·sesión⁻¹ que el motor difuso necesita para sus membresías de beta. Theil-Sen y OLS comparten unidades; Spearman no.

**Files:**
- Modify: `services.py` (función `_pendiente_calendar`, agregar import de `stats_utils`)
- Modify: `tests/test_services.py`

**Prerequisito:** Task 0 completado (stats_utils.py existe).

- [ ] **Step 1: Agregar tests de beta**

```python
# Agregar a tests/test_services.py

def test_beta_ruido_retorna_cero():
    """
    VMP oscilando sin tendencia → IC del Theil-Sen incluye 0 → beta = 0.0.
    OLS hubiera retornado un slope pequeño pero no-cero.
    """
    np.random.seed(42)
    vmps = [1.2 + np.random.uniform(-0.08, 0.08) for _ in range(7)]
    fechas = pd.date_range("2025-01-01", periods=7, freq="D").strftime("%Y-%m-%d").tolist()
    df = pd.DataFrame({"nombre": ["T"]*7, "fecha": fechas, "vmp_hoy": vmps})
    res = calcular_metricas(df, "T")
    assert res["beta_aguda"] == 0.0, f"Ruido → beta_aguda debe ser 0.0, fue {res['beta_aguda']}"

def test_beta_caida_lineal_es_negativa():
    """Caída lineal perfecta → beta negativo significativo."""
    vmps = [1.5, 1.42, 1.35, 1.28, 1.21, 1.14, 1.07]
    fechas = pd.date_range("2025-01-01", periods=7, freq="D").strftime("%Y-%m-%d").tolist()
    df = pd.DataFrame({"nombre": ["T"]*7, "fecha": fechas, "vmp_hoy": vmps})
    res = calcular_metricas(df, "T")
    assert res["beta_aguda"] < -0.01, f"Caída → beta_aguda debe ser <-0.01, fue {res['beta_aguda']}"

def test_beta_robusto_a_outlier():
    """
    Un outlier extremo (error de medición) no debe invertir la dirección del slope.
    Theil-Sen: robusto. OLS: vulnerable.
    """
    vmps = [1.40, 1.35, 1.30, 5.00, 1.20, 1.15, 1.10]  # 5.00 = outlier
    fechas = pd.date_range("2025-01-01", periods=7, freq="D").strftime("%Y-%m-%d").tolist()
    df = pd.DataFrame({"nombre": ["T"]*7, "fecha": fechas, "vmp_hoy": vmps})
    res = calcular_metricas(df, "T")
    assert res["beta_aguda"] < 0.0, (
        f"Con outlier extremo, Theil-Sen debe mantener pendiente negativa. "
        f"OLS fallaría. Obtenido: {res['beta_aguda']}"
    )
```

- [ ] **Step 2: Correr los tests — deben FALLAR**

```bash
pytest tests/test_services.py::test_beta_ruido_retorna_cero 
       tests/test_services.py::test_beta_caida_lineal_es_negativa 
       tests/test_services.py::test_beta_robusto_a_outlier -v
```

- [ ] **Step 3: Reemplazar _pendiente_calendar en services.py**

Agregar el import al inicio del archivo:
```python
from stats_utils import pendiente_theil_sen
```

Localizar la función anidada `_pendiente_calendar` dentro de `calcular_metricas` y reemplazarla por completo:

```python
# ANTES — eliminar esta función completa:
def _pendiente_calendar(dias_back: int, min_n: int) -> float:
    cutoff = vmp_daily.index[-1] - pd.Timedelta(days=dias_back - 1)
    win = vmp_daily[vmp_daily.index >= cutoff].dropna()
    if len(win) < min_n:
        return 0.0
    x = (win.index - win.index[0]).days.values.astype(float)
    slope_d = np.polyfit(x, win.values, 1)[0]
    avg_gap = ...
    return float(slope_d * avg_gap)

# DESPUÉS — reemplazar con llamada a stats_utils:
def _pendiente_calendar(dias_back: int, min_n: int) -> float:
    """
    Pendiente robusta de VMP usando Theil-Sen (no-paramétrico).
    Retorna 0.0 si n < min_n o el IC 90% del slope incluye 0.
    Unidades: m/s por sesión.
    """
    cutoff = vmp_daily.index[-1] - pd.Timedelta(days=dias_back - 1)
    win = vmp_daily[vmp_daily.index >= cutoff].dropna()
    return pendiente_theil_sen(win, min_n=min_n)
```

- [ ] **Step 4: Correr los tres tests — deben PASAR**

```bash
pytest tests/test_services.py::test_beta_ruido_retorna_cero 
       tests/test_services.py::test_beta_caida_lineal_es_negativa 
       tests/test_services.py::test_beta_robusto_a_outlier -v
```

- [ ] **Step 5: Commit**

```bash
git add services.py tests/test_services.py
git commit -m "fix: reemplazar OLS slope con Theil-Sen para beta_aguda y beta_28

Theil-Sen (mediana de pendientes por pares) no asume normalidad de
residuos, tiene breakdown point 29% contra outliers, y usa el IC
del slope para significancia en lugar de p-value OLS.
Elimina la distorsión del slope por sesiones atípicas (pico de test,
error de encoder, regreso de lesión)."
```

---

## Task 3: Hacer que Shapiro-Wilk controle el estimador para ACWR, delta_pct y z_meso

**Problema (nuevo):** El Shapiro-Wilk se calcula en `services.py` pero el resultado NUNCA se usa para seleccionar el estimador. Todo el pipeline usa `mean()` y `std()` incondicionalmente. Para VMP no-normal (asimétrico por bloques de carga, tapering, valores atípicos), la media y SD son estimadores sesgados del centro y dispersión reales.

**Variables afectadas:**
- `mma7` / `mmc28` → base del ACWR: rolling mean → rolling median si no-normal
- `mmc28` como denominador de `delta_pct` → misma decisión
- `z_meso` → `(vmp - mean) / std` → `(vmp - mediana) / MAD` si no-normal

**Por qué la mediana/MAD es mejor con no-normalidad:** Un solo día de VMP=2.0 (test máximo) cuando el rango habitual es 1.0-1.3 eleva el rolling mean en ~15%, lo que deprime el ACWR artificialmente (parece que el atleta está en detraining cuando no lo está). La rolling median ignora ese punto.

**Files:**
- Modify: `services.py` (bloque rolling windows + z_meso, agregar import stats_utils)
- Modify: `tests/test_services.py`

**Prerequisito:** Task 0 completado.

- [ ] **Step 1: Escribir los tests**

```python
# Agregar a tests/test_services.py

def test_acwr_robusto_a_sesion_atipica():
    """
    Un único día de VMP muy alta (test máximo) no debe elevar MMA7 artificialmente.
    Rolling median es resistente; rolling mean no lo es.
    Cuando los datos no son normales (Shapiro falla por el outlier),
    el pipeline debe usar mediana → ACWR permanece cercano a 1.0.
    """
    # 14 sesiones estables en 1.2, día 14 con outlier 2.8 (test máximo)
    vmps = [1.2]*13 + [2.8]
    fechas = pd.date_range("2025-01-01", periods=14, freq="D").strftime("%Y-%m-%d").tolist()
    df = pd.DataFrame({"nombre": ["T"]*14, "fecha": fechas, "vmp_hoy": vmps})
    res = calcular_metricas(df, "T")
    # Con rolling mean: MMA7 ≈ (6×1.2 + 2.8)/7 ≈ 1.43, ACWR ≈ 1.19 (inflado)
    # Con rolling median (robusto): MMA7 = 1.2, ACWR ≈ 1.0
    assert res["acwr"] < 1.15, (
        f"Con outlier y datos no-normales, ACWR={res['acwr']:.3f} debería ser ~1.0. "
        f"Rolling mean infla artificialmente el ACWR."
    )

def test_zmeso_usa_mediana_mad_con_no_normalidad():
    """
    Con distribución claramente no-normal (Shapiro falla),
    z_meso se calcula con mediana y MAD en lugar de mean y SD.
    Un valor normal al final de una serie con outlier debe tener z_meso ≈ 0.
    """
    # 9 sesiones ≈1.2, outlier extremo en posición 5, último valor normal
    vmps = [1.20, 1.21, 1.19, 1.22, 1.20, 0.10, 1.21, 1.20, 1.22]
    fechas = pd.date_range("2025-01-01", periods=9, freq="D").strftime("%Y-%m-%d").tolist()
    df = pd.DataFrame({"nombre": ["T"]*9, "fecha": fechas, "vmp_hoy": vmps})
    res = calcular_metricas(df, "T")
    # Con SD clásica: outlier infla SD → z_meso del valor 1.22 ≈ 0 (correcto por azar)
    # pero para valores normales del equipo la SD estaría distorsionada
    # Con MAD robusto: z_meso del valor normal debe ser cercano a 0 o ligeramente positivo
    assert res["z_meso"] > -1.0, (
        f"Con MAD robusto, VMP=1.22 en serie ≈1.2 debe tener z_meso≈0, "
        f"obtenido {res['z_meso']:.2f}"
    )

def test_shapiro_normal_usa_media(self):
    """Con datos genuinamente normales, debe usar mean (más eficiente)."""
    np.random.seed(1)
    vmps = list(np.random.normal(1.2, 0.03, 20))
    fechas = pd.date_range("2025-01-01", periods=20, freq="D").strftime("%Y-%m-%d").tolist()
    df = pd.DataFrame({"nombre": ["T"]*20, "fecha": fechas, "vmp_hoy": vmps})
    res = calcular_metricas(df, "T")
    # Para datos normales, ACWR debe ser calculado correctamente (≈ 1.0 para serie estable)
    assert 0.90 < res["acwr"] < 1.10, f"ACWR con datos normales estables debe ser ≈1.0"
```

- [ ] **Step 2: Correr los tests — deben FALLAR**

```bash
pytest tests/test_services.py::test_acwr_robusto_a_sesion_atipica 
       tests/test_services.py::test_zmeso_usa_mediana_mad_con_no_normalidad -v
```

- [ ] **Step 3: Modificar services.py — añadir import y lógica adaptativa**

Agregar al bloque de imports de `services.py`:
```python
from stats_utils import estimar_centro_dispersion
from scipy.stats import shapiro as _shapiro
```

Localizar el bloque de rolling windows (donde se calcula `mma7`, `mmc28`) y reemplazar:

```python
# ANTES:
mma7  = vmp_daily.rolling(7,  min_periods=_MIN_N7).mean()
mmc28 = vmp_daily.rolling(28, min_periods=_MIN_N28).mean()

# DESPUÉS — rolling adaptativo según normalidad de la serie completa:
_arr_completo = vmp_daily.dropna().values
if len(_arr_completo) >= 8:
    _, _p_sw = _shapiro(_arr_completo)
    _usar_mediana = _p_sw <= 0.05
else:
    _usar_mediana = True  # n pequeño → robusto por defecto

if _usar_mediana:
    mma7  = vmp_daily.rolling(7,  min_periods=_MIN_N7).median()
    mmc28 = vmp_daily.rolling(28, min_periods=_MIN_N28).median()
else:
    mma7  = vmp_daily.rolling(7,  min_periods=_MIN_N7).mean()
    mmc28 = vmp_daily.rolling(28, min_periods=_MIN_N28).mean()
```

Localizar el bloque `# ── Z-Score mesociclo` y reemplazar:

```python
# ANTES:
cutoff_meso = last_date - pd.Timedelta(days=ventana_meso - 1)
win_meso = vmp_daily[vmp_daily.index >= cutoff_meso].dropna()
if len(win_meso) >= 4 and float(win_meso.std()) > 0:
    z_meso = (last_vmp - float(win_meso.mean())) / float(win_meso.std())
else:
    z_meso = 0.0

# DESPUÉS — estimar_centro_dispersion decide mean/SD vs mediana/MAD:
cutoff_meso = last_date - pd.Timedelta(days=ventana_meso - 1)
win_meso = vmp_daily[vmp_daily.index >= cutoff_meso].dropna()
if len(win_meso) >= 4:
    centro_meso, disp_meso = estimar_centro_dispersion(win_meso)
    z_meso = (last_vmp - centro_meso) / disp_meso if disp_meso > 0 else 0.0
else:
    z_meso = 0.0
z_meso = float(np.clip(z_meso, -4.0, 4.0))
```

- [ ] **Step 4: Correr los tests — deben PASAR**

```bash
pytest tests/test_services.py -v
```

- [ ] **Step 5: Commit**

```bash
git add services.py tests/test_services.py
git commit -m "fix: Shapiro-Wilk controla el estimador para ACWR, delta_pct y z_meso

El test de normalidad ya existía pero nunca ramificaba el código.
Ahora:
- Shapiro p>0.05 → rolling mean, SD (eficiente para normales)
- Shapiro p≤0.05 o n<8 → rolling median, MAD×1.4826 (robusto)
Elimina la distorsión por outliers en ventanas 7d y 28d."
```

---

## Task 4: Corregir reglas problemáticas en fuzzy_engine.py

**4a — Regla 25:** `ci_v["RECUPERACION"] → fat_v["optimo"]` clasifica como ÓPTIMO siempre que el plan sea de recuperación, sin importar VMP, ACWR ni wellness. Un atleta en estado CRÍTICO con sesión de recuperación planificada queda reclasificado como ÓPTIMO.

**4b — Reglas 5-6:** `acwr["bajo"] & delta["alarma"] & beta["neg_fuerte"] → critico`. ACWR bajo indica carga reciente menor que la crónica. Pero delta alarma indica VMP caída >18%. La combinación existe en síndrome de sobreentrenamiento fase 2 (decremento de rendimiento sin carga reciente alta), pero disparar CRÍTICO es excesivo sin confirmación de VMP baja directa. Rebajamos a `fatiga_acumulada`.

**Files:**
- Modify: `fuzzy_engine.py` (función `construir_reglas`)
- Create: `tests/test_fuzzy_engine.py`

- [ ] **Step 1: Escribir los tests**

```python
# tests/test_fuzzy_engine.py
import pytest
from fuzzy_engine import construir_motor_fuzzy, evaluar_atleta

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
```

- [ ] **Step 2: Correr los tests — deben FALLAR**

```bash
pytest tests/test_fuzzy_engine.py -v
```

- [ ] **Step 3: Corregir las reglas en fuzzy_engine.py**

En `construir_reglas()`, localizar y modificar:

```python
# ANTES — Regla 25 (línea ~127):
ctrl.Rule(ci_v["RECUPERACION"], fat_v["optimo"]),

# DESPUÉS — requiere estado basal aceptable para clasificar como óptimo:
ctrl.Rule(
    ci_v["RECUPERACION"] & (vmp_v["funcional"] | vmp_v["alta"])
    & (acwr_v["optimo"] | acwr_v["bajo"]),
    fat_v["optimo"]
),
ctrl.Rule(
    ci_v["RECUPERACION"] & vmp_v["baja"] & acwr_v["optimo"],
    fat_v["alerta_temprana"]
),
```

```python
# ANTES — Reglas 5-6 (líneas ~97-98):
ctrl.Rule(acwr_v["bajo"] & delta_v["alarma"] & ba_v["neg_fuerte"], fat_v["critico"]),
ctrl.Rule(acwr_v["bajo"] & b28_v["deterioro"] & ba_v["neg_fuerte"], fat_v["critico"]),

# DESPUÉS — rebajado a fatiga_acumulada (ACWR bajo ≠ confirmación de sobrecarga):
ctrl.Rule(acwr_v["bajo"] & delta_v["alarma"] & ba_v["neg_fuerte"], fat_v["fatiga_acumulada"]),
ctrl.Rule(acwr_v["bajo"] & b28_v["deterioro"] & ba_v["neg_fuerte"], fat_v["fatiga_acumulada"]),
```

- [ ] **Step 4: Correr los tests — deben PASAR**

```bash
pytest tests/test_fuzzy_engine.py -v
```

- [ ] **Step 5: Commit**

```bash
git add fuzzy_engine.py tests/test_fuzzy_engine.py
git commit -m "fix: corregir reglas contradictorias — Mamdani v4.2 → v4.3

Regla 25: RECUPERACION → optimo ahora requiere vmp[funcional|alta]
+ acwr[optimo|bajo]. Plan de recuperación no reclasifica atleta crítico.

Reglas 5-6: acwr[bajo] + tendencias negativas → fatiga_acumulada
(antes: critico). ACWR bajo indica subestímulo agudo, no sobrecarga."
```

---

## Task 5: Aumentar peso implícito de vmp_hoy en las reglas

**Problema:** `vmp_hoy` aparece en 4 reglas directas. Es declarado "indicador principal de readiness neuromuscular" en el fuzzy_model_report.md, pero ACWR tiene 12 reglas y delta_pct tiene 10. El peso implícito contradice la jerarquía declarada.

**Files:**
- Modify: `fuzzy_engine.py` (función `construir_reglas`)
- Modify: `tests/test_fuzzy_engine.py`

- [ ] **Step 1: Agregar tests de vmp_hoy**

```python
# Agregar a tests/test_fuzzy_engine.py

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
```

- [ ] **Step 2: Correr — registrar resultado actual (pueden pasar o fallar)**

```bash
pytest tests/test_fuzzy_engine.py::test_vmp_muy_baja_con_wellness_deficiente_es_critico 
       tests/test_fuzzy_engine.py::test_vmp_alta_con_condiciones_optimas_es_optimo -v
```

- [ ] **Step 3: Agregar reglas de vmp_hoy en fuzzy_engine.py**

Dentro de `construir_reglas()`, después del bloque de reglas de `vmp_hoy` existente, agregar:

```python
# ── Reglas adicionales vmp_hoy — aumentar peso del indicador primario ──────
ctrl.Rule(vmp_v["muy_baja"] & wellness_v["DEFICIENTE"], fat_v["critico"]),
ctrl.Rule(vmp_v["muy_baja"] & acwr_v["excesivo"], fat_v["critico"]),
ctrl.Rule(vmp_v["muy_baja"] & acwr_v["optimo"], fat_v["fatiga_acumulada"]),
ctrl.Rule(vmp_v["baja"] & acwr_v["excesivo"] & wellness_v["DEFICIENTE"], fat_v["fatiga_acumulada"]),
ctrl.Rule(vmp_v["alta"] & acwr_v["optimo"] & wellness_v["OPTIMO"], fat_v["optimo"]),
ctrl.Rule(vmp_v["alta"] & wellness_v["DEFICIENTE"], fat_v["alerta_temprana"]),
```

- [ ] **Step 4: Correr todos los tests del motor**

```bash
pytest tests/test_fuzzy_engine.py -v
```

- [ ] **Step 5: Commit**

```bash
git add fuzzy_engine.py tests/test_fuzzy_engine.py
git commit -m "feat: aumentar representación de vmp_hoy en reglas difusas (4 → 10 reglas)

vmp_hoy declarado indicador primario pero tenía el menor número de reglas.
6 reglas adicionales que le dan poder de decisión independiente de ACWR
y delta_pct, alineando el peso implícito con la jerarquía fisiológica."
```

---

## Task 6: Rebalancear pesos DQI hacia ventana aguda (7 días)

**Problema:** `DQI = 0.40×(n_7d/3) + 0.60×(n_28d/12)`. El histórico (28d) pesa más que los datos recientes (7d). Para detectar fatiga aguda, la disponibilidad de datos recientes es más crítica operacionalmente.

**Files:**
- Modify: `services.py` (constantes `_DQI_W7`, `_DQI_W28`)
- Modify: `tests/test_services.py`

- [ ] **Step 1: Escribir el test**

```python
# Agregar a tests/test_services.py

def test_dqi_penaliza_ausencia_de_datos_recientes():
    """
    Muchos datos en 28d pero ninguno en los últimos 7d → DQI bajo.
    Con pesos 55/45, la falta de datos recientes pesa más que antes (40/60).
    """
    # 12 sesiones en días 1-21, ninguna en días 22-28
    fechas = pd.date_range("2025-01-01", periods=12, freq="2D").strftime("%Y-%m-%d").tolist()
    df = pd.DataFrame({"nombre": ["T"]*12, "fecha": fechas, "vmp_hoy": [1.2]*12})
    res = calcular_metricas(df, "T")
    # n_28d=12 (máx), n_7d=0: DQI = 0.55*0 + 0.45*1.0 = 0.45
    assert res["dqi"] < 0.55, f"Sin datos recientes, DQI={res['dqi']:.2f} debe ser <0.55"
    assert res["calidad_dato"] in ("baja", "insuficiente")
```

- [ ] **Step 2: Correr — debe FALLAR**

```bash
pytest tests/test_services.py::test_dqi_penaliza_ausencia_de_datos_recientes -v
```

- [ ] **Step 3: Cambiar las constantes en services.py**

```python
# ANTES:
_DQI_W7: float = 0.40
_DQI_W28: float = 0.60

# DESPUÉS:
_DQI_W7: float = 0.55   # mayor peso a calidad reciente — detección de fatiga aguda
_DQI_W28: float = 0.45  # menor peso al histórico crónico
```

- [ ] **Step 4: Correr todos los tests de services**

```bash
pytest tests/test_services.py -v
```

- [ ] **Step 5: Commit**

```bash
git add services.py tests/test_services.py
git commit -m "fix: rebalancear DQI a 55/45 (ventana 7d/28d)

Ventana 7d sube de 0.40 a 0.55: la calidad de los datos recientes
es más crítica para detectar fatiga aguda que la densidad histórica."
```

---

## Task 7: Wellness — verificación de Cronbach's alpha + pesos evidenciados

**Problema (nuevo, incorporado de la auditoría):** Los 5 ítems del Hooper son ordinales (Likert 1-7). Tratar `wellness_norm` como variable continua requiere verificar que los ítems midan el mismo constructo (Cronbach α ≥ 0.70). Sin esto, la suma/media puede ser incoherente. Además, los pesos iguales (0.20 cada ítem) ignoran que sueño y fatiga tienen mayor validez predictiva de rendimiento neuromuscular (Saw et al. 2016).

**Nota metodológica:** Los ítems siguen siendo ordinales para análisis bivariado externo (usar Spearman con VMP, no Pearson). Dentro del motor difuso, `wellness_norm` como número compuesto [0,1] es tratado correctamente por la fuzzificación.

**Files:**
- Modify: `wellness.py`
- Create: `tests/test_wellness.py`

- [ ] **Step 1: Escribir los tests**

```python
# tests/test_wellness.py
import pytest
import pandas as pd
from wellness import calcular_wellness, cronbach_alpha_wellness

def test_sueno_deficiente_pesa_mas_que_dolor_leve():
    """
    Sueño muy malo + dolor leve debe dar peor wellness que sueño perfecto + dolor severo.
    El sueño tiene mayor peso predictivo de rendimiento (Saw et al. 2016).
    """
    w_sueño_malo = calcular_wellness(sueno=7, fatiga=2, estres=2, dolor=2, humor=5)
    w_dolor_malo  = calcular_wellness(sueno=1, fatiga=2, estres=2, dolor=7, humor=5)
    assert w_sueño_malo < w_dolor_malo, (
        f"Sueño malo (w={w_sueño_malo:.3f}) debe ser < dolor malo (w={w_dolor_malo:.3f})"
    )

def test_extremos_en_rango_cero_uno():
    assert calcular_wellness(1, 1, 1, 1, 7) == pytest.approx(1.0, abs=0.001)
    assert calcular_wellness(7, 7, 7, 7, 1) == pytest.approx(0.0, abs=0.001)

def test_valores_invalidos_lanzan_error():
    with pytest.raises(ValueError):
        calcular_wellness(sueno=0, fatiga=4, estres=4, dolor=4, humor=4)
    with pytest.raises(ValueError):
        calcular_wellness(sueno=4, fatiga=8, estres=4, dolor=4, humor=4)

def test_cronbach_alpha_muestra_coherente():
    """
    Con 5 ítems que varían en la misma dirección (coherentes),
    alpha debe ser > 0.70.
    """
    import pandas as pd
    # Atletas con respuestas coherentes (todos mal o todos bien)
    datos = pd.DataFrame({
        "sueno":  [1, 2, 6, 7, 3, 5, 2, 6],
        "fatiga": [1, 2, 5, 7, 3, 5, 2, 6],
        "estres": [2, 2, 5, 6, 4, 4, 3, 5],
        "dolor":  [1, 3, 4, 7, 3, 4, 2, 5],
        "humor":  [7, 6, 2, 1, 5, 3, 6, 2],
    })
    alpha = cronbach_alpha_wellness(datos)
    assert alpha >= 0.70, f"Alpha={alpha:.3f} debe ser >=0.70 para items coherentes"

def test_cronbach_alpha_muestra_incoherente_es_bajo():
    """Ítems aleatorios sin relación → alpha bajo."""
    import numpy as np
    np.random.seed(7)
    datos = pd.DataFrame({
        col: np.random.randint(1, 8, 20)
        for col in ["sueno", "fatiga", "estres", "dolor", "humor"]
    })
    alpha = cronbach_alpha_wellness(datos)
    assert alpha < 0.70, f"Alpha={alpha:.3f} debe ser <0.70 para items incoherentes"
```

- [ ] **Step 2: Correr los tests — deben FALLAR**

```bash
pytest tests/test_wellness.py -v
```

- [ ] **Step 3: Modificar wellness.py**

```python
# wellness.py — reemplazar calcular_wellness y agregar cronbach_alpha_wellness

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
```

- [ ] **Step 4: Correr los tests — deben PASAR**

```bash
pytest tests/test_wellness.py -v
```

- [ ] **Step 5: Commit**

```bash
git add wellness.py tests/test_wellness.py
git commit -m "feat: wellness — pesos Hooper evidenciados + Cronbach's alpha

Pesos: sueno(0.30)>fatiga(0.25)>estres(0.20)>dolor(0.15)>humor(0.10)
basados en Saw et al. 2016 y Buchheit 2014.
Agrega cronbach_alpha_wellness() para verificar fiabilidad antes de
usar wellness_norm como variable continua en análisis bivariados.
Extremos [0.0, 1.0] conservados."
```

---

## Task 8: Crear analysis.py — análisis bivariado para calibrar membresías

**Problema (nuevo, incorporado de la auditoría):** Los umbrales de las funciones de membresía (ej. `vmp_v["baja"] = trimf([0.65, 0.90, 1.10])`) fueron definidos por intuición sin datos empíricos del equipo. Las reglas del motor deberían reflejar las correlaciones reales entre variables, no supuestas.

**Qué resuelve este módulo:**
1. **Matriz Spearman** entre todas las variables → confirma/refuta redundancias (delta_pct vs z_meso ≈ −0.90)
2. **Cross-correlación con lag** ACWR→VMP → informa si el ACWR de hace N sesiones predice el VMP actual mejor que el ACWR de hoy
3. **Umbrales por percentil** → sustituye los umbrales arbitrarios de membresía por valores empíricos del equipo

**Nota sobre Spearman vs Pearson:** Se usa Spearman porque: (a) el VMP puede ser no-normal, (b) wellness_norm es técnicamente ordinal, (c) Spearman captura relaciones monotónicas no lineales. Pearson requiere linealidad y normalidad conjunta.

**Files:**
- Create: `analysis.py`
- Create: `tests/test_analysis.py`

- [ ] **Step 1: Escribir los tests**

```python
# tests/test_analysis.py
import pytest
import numpy as np
import pandas as pd
from analysis import (
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
```

- [ ] **Step 2: Correr tests — deben FALLAR (módulo no existe)**

```bash
pytest tests/test_analysis.py -v
```

- [ ] **Step 3: Crear analysis.py**

```python
# analysis.py
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
```

- [ ] **Step 4: Correr los tests — deben PASAR**

```bash
pytest tests/test_analysis.py -v
```

- [ ] **Step 5: Commit**

```bash
git add analysis.py tests/test_analysis.py
git commit -m "feat: crear analysis.py — análisis bivariado para calibrar membresías

Funciones:
- matriz_spearman(): correlaciones con Spearman (no Pearson) por no-normalidad
  y escala ordinal del wellness
- cross_correlation_lag(): ACWR[t-lag] → VMP[t] para detectar retrasos causales
- umbrales_por_percentil(): calibra membresías con percentiles empíricos del equipo
- reporte_redundancias(): identifica pares con |rho| > threshold

Permite reemplazar umbrales arbitrarios (ej. vmp_v['baja']=[0.65,0.90,1.10])
por valores derivados de los datos reales del equipo."
```

---

## Task 9: Test de integración del pipeline completo

**Files:**
- Create: `tests/test_integration.py`

- [ ] **Step 1: Crear el test de integración**

```python
# tests/test_integration.py
import pytest
import pandas as pd
import numpy as np
from fuzzy_engine import construir_motor_fuzzy
from services import pipeline_diagnostico
from analysis import matriz_spearman, reporte_redundancias

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
```

- [ ] **Step 2: Correr el test de integración**

```bash
pytest tests/test_integration.py -v
```

- [ ] **Step 3: Correr la suite completa**

```bash
pytest tests/ -v --tb=short
```
Resultado esperado: todos los tests pasan. Si alguno falla, revisar si la firma de funciones del Task 0-8 correspondiente tiene incompatibilidades.

- [ ] **Step 4: Commit final**

```bash
git add tests/test_integration.py
git commit -m "test: suite de integración end-to-end del pipeline v2

Cubre: atleta óptimo/crítico, datos insuficientes, sin clavados,
regla 25 (crítico + recovery no sube a óptimo), y verificación de
que analysis.py detecta redundancia delta_pct↔z_meso en datos reales."
```

---

## Autoreview del plan

### 1. Cobertura spec → tarea

| Problema / Consideración estadística | Tarea | Estado |
|---|---|---|
| Doble contabilización wellness | 1 | ✅ |
| OLS slope sin robustez → Theil-Sen | 2 | ✅ |
| Shapiro-Wilk no controla el estimador | 3 | ✅ |
| ACWR rolling mean → rolling adaptativo | 3 | ✅ |
| z_meso paramétrico frágil → mediana/MAD | 3 | ✅ |
| Regla 25 (recovery invalida estado real) | 4 | ✅ |
| Reglas 5-6 contradictorias (ACWR bajo → critico) | 4 | ✅ |
| vmp_hoy subrepresentado en reglas | 5 | ✅ |
| DQI sobrepesa histórico | 6 | ✅ |
| Wellness pesos sin evidencia | 7 | ✅ |
| Wellness ordinal → Cronbach alpha antes de tratar como continua | 7 | ✅ |
| Por qué Spearman y no Pearson para correlaciones | 8 | ✅ |
| Por qué Theil-Sen y no Spearman para slopes | 2 | ✅ |
| Análisis bivariado para calibrar membresías | 8 | ✅ |
| Cross-correlación con lag ACWR→VMP | 8 | ✅ |
| Umbrales de membresía por percentiles empíricos | 8 | ✅ |
| Tests de integración incluyendo regla 25 y analysis | 9 | ✅ |

### 2. Consistencia de tipos y firmas

- `stats_utils.py` expone `estimar_centro_dispersion(pd.Series) → (float, float)` y `pendiente_theil_sen(pd.Series, int, ...) → float`. Ambas usadas en `services.py` Task 2 y 3.
- `services.py` no cambia su firma pública `calcular_metricas(df, nombre, ...)`. Compatible con todos los tests.
- `wellness.py` no cambia la firma de `calcular_wellness(int×5) → float`. Agrega `cronbach_alpha_wellness(pd.DataFrame) → float`.
- `analysis.py` funciones independientes; no hay dependencias cruzadas entre tasks 8 y 1-7.
- `fuzzy_engine.py` no cambia firmas. `construir_reglas()` y `evaluar_atleta()` compatibles.

### 3. Impacto en el índice de fatiga por tarea

| Tarea | Atleta crítico | Atleta óptimo |
|---|---|---|
| T1 (wellness fix) | Índice sube (menos penalización duplicada) | Baja levemente |
| T2 (Theil-Sen) | Menos reglas activas por ruido; pendientes reales conservadas | Sin cambio |
| T3 (estimador adaptativo) | Más estable ante outliers y no-normalidad | Más estable |
| T4 (reglas) | Sube si ACWR es bajo; baja si VMP muy baja | Sin cambio |
| T5 (vmp_hoy) | Baja si VMP es muy baja (refuerza crítico) | Sube si VMP alta |
| T6 (DQI) | Más advertencias con datos irregulares | Sin cambio |
| T7 (wellness pesos) | Sueño/fatiga pesan más; más sensible cuando son malos | Igual |
| T8 (analysis) | No afecta el índice — informa calibración futura | Igual |
