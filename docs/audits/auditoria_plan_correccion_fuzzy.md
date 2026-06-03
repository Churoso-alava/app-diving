# Auditoría Exhaustiva del Plan de Corrección Fuzzy v2
## AppDivingCodex — Motor Fuzzy Mamdani v4.2→v4.3

**Fecha:** 2026-06-03  
**Auditor:** Análisis automatizado + revisión manual  
**Repositorio:** https://github.com/Churoso-alava/app-diving  
**Commit base:** `4997796` (reestructuración completa)  
**Estado del plan:** Parcialmente implementado, con **bugs críticos en producción**

---

## TL;DR — Diagnóstico Ejecutivo

El plan de corrección ha sido implementado en **~80%**, pero contiene **un bug crítico (BUG-001) que inutiliza el motor fuzzy en producción**: el antecedente `rpe_v` (carga subjetiva) se define en el sistema y se usa en 3 reglas, pero nunca recibe un valor de input, lo que provoca que scikit-fuzzy lance `"All antecedents must have input values!"` y caiga al fallback de 50.0 para **todos los atletas**. Además, se identificaron **11 deudas, bugs y gaps** que deben abordarse antes de que el sistema sea operacionalmente confiable.

**Recomendación prioritaria:** No desplegar en producción hasta resolver BUG-001, BUG-002 y GAP-002. Las deudas técnicas TD-009 y TD-010 son críticas para la validez conceptual del modelo.

---

## 1. Metodología de Auditoría

La auditoría siguió un protocolo de cuatro fases diseñado para garantizar cobertura exhaustiva:

| Fase | Actividad | Resultado |
|---|---|---|
| **1. Reconocimiento** | Clonado completo del repositorio, mapeo de 1,732 archivos, identificación de 14 módulos Python core + 15 test files | Estructura `app/→core/` con packages anidados |
| **2. Ejecución de tests** | `pytest tests/` excluyendo dependencias de DB (streamlit, supabase). **44 tests ejecutados: 36 pasaron, 8 fallaron** | 18% de tasa de fallo en tests independientes de infraestructura |
| **3. Lectura de código** | Revisión línea por línea de `services.py` (604 LOC), `fuzzy_engine.py` (250 LOC), `stats_utils.py` (100 LOC), `wellness.py` (76 LOC), `analysis.py` (137 LOC), `biomechanics.py` (66 LOC), `schemas.py` (206 LOC), y 8 archivos de tests | Identificación de 3 bugs, 2 gaps funcionales, 6 deudas técnicas |
| **4. Contraste con plan y auditorías previas** | Comparación task-by-task del plan v2 contra implementación; integración de hallazgos de auditorías `1.txt` y `2.txt` | 6 discrepancias estructurales, 3 hallazgos no documentados en el plan |

Las pruebas se ejecutaron en un entorno Python 3.12 con las dependencias del proyecto (`requirements.txt`), excluyendo únicamente los tests que requieren conexión activa a Supabase (`test_db.py`, `test_wellness_insertion.py`, `test_auth_session.py`, `test_ui_charts.py`, `test_injury_db.py`, `test_injury_integration.py`, `test_injury_services.py`). Los tests excluidos por dependencias de infraestructura **no** forman parte del alcance de esta auditoría, que se centra exclusivamente en la lógica del motor fuzzy y sus correcciones.

---

## 2. Gap Analysis Task por Task

### 2.1 Task 0: `stats_utils.py` — Módulo Estadístico Adaptativo

**Estado de implementación:** ✅ **COMPLETA — CON BUG MENOR EN TEST**

El módulo `core/stats_utils.py` existe con ambas funciones especificadas en el plan. La implementación coincide prácticamente carácter por carácter con la especificación. `estimar_centro_dispersion()` implementa correctamente la lógica de Shapiro-Wilk con fallback a mediana/MAD, incluyendo el manejo de arrays constantes mediante `try/except` alrededor del test (línea 52-55) — una mejora defensiva que no estaba en el plan original pero que es correcta. `pendiente_theil_sen()` utiliza `theilslopes` de scipy con el factor de escala de gap entre sesiones.

**Bug identificado en test (BUG-003):** El test `test_datos_ruidosos_retorna_cero` falla porque `np.random.uniform(-0.05, 0.05, 7)` puede generar, por azar, una serie donde el intervalo de confianza del 90% de Theil-Sen **no** incluye el cero. La función `pendiente_theil_sen()` retorna un valor no-cero (ej. `-0.017`) cuando el test espera exactamente `0.0`. Este es un **test estadísticamente frágil**: con solo 7 puntos y ruido uniforme, la probabilidad de que el IC no capture el cero es mayor que cero. La implementación de la función es correcta; el test necesita una semilla garantizada o una serie construida para que el IC necesariamente incluya cero.

| Aspecto | Plan | Implementación | Estado |
|---|---|---|---|
| `estimar_centro_dispersion()` | Shapiro → mean/SD vs mediana/MAD | ✅ Idéntico + manejo arrays constantes | Cumple |
| `pendiente_theil_sen()` | Theil-Sen con IC 90% | ✅ Idéntico + clip a [-0.25, 0.25] | Cumple |
| `stats_utils.py` en `core/` | No especificaba ruta | En `core/stats_utils.py` con imports relativos | Ajustado |
| Tests TDD | 4 tests + 4 tests Theil-Sen | 4 + 4, 1 falla por fragilidad estadística | Parcial |

**Recomendación:** Corregir el test `test_datos_ruidosos_retorna_cero` usando una serie con semilla garantizada donde el IC sí incluya 0 (ej. `np.random.seed(99)` y verificar), o reescribir para probar el comportamiento probabilístico (ej. "en 100 iteraciones, ≥80% retornan 0.0").

---

### 2.2 Task 1: Eliminar Doble Contabilización del Wellness

**Estado de implementación:** ✅ **COMPLETA — SIN TESTS DIRECTOS**

En `services.py`, líneas 318-322, el factor `(2.0 - wellness_norm)` ha sido eliminado correctamente:

```python
# Líneas 318-322 de services.py (ACTUAL)
if clavados_planificados:
    carga_bruta_plan = carga_bruta_sesion(clavados_planificados)
    carga_integrada_plan = normalizar_carga(carga_bruta_plan)
else:
    carga_integrada_plan = 0.0
```

El wellness ya no modifica la carga integrada; entra al motor únicamente por su canal independiente `wellness_norm`. Este fix es **correcto y completo**. Sin embargo, el test especificado en el plan (`test_carga_integrada_no_depende_de_wellness`) **no existe** en `tests/test_services.py`. El archivo de tests de services solo contiene 2 tests básicos que no verifican este comportamiento.

**Implicación operativa:** Sin test directo, una regresión futura (reintroducción accidental del factor) no sería detectada por la CI. El fix es bueno pero sin cobertura de tests es vulnerable a regresiones.

---

### 2.3 Task 2: Reemplazar OLS con Theil-Sen para `beta_aguda` y `beta_28`

**Estado de implementación:** ✅ **COMPLETA — SIN TESTS DIRECTOS**

La función `_pendiente_calendar` en `services.py` (líneas 252-260) ahora delega completamente a `pendiente_theil_sen`:

```python
def _pendiente_calendar(dias_back: int, min_n: int) -> float:
    cutoff = vmp_daily.index[-1] - pd.Timedelta(days=dias_back - 1)
    win = vmp_daily[vmp_daily.index >= cutoff].dropna()
    return pendiente_theil_sen(win, min_n=min_n)
```

La implementación de Theil-Sen es correcta y la firma mantiene compatibilidad. No obstante, al igual que el Task 1, **no existen tests específicos** que verifiquen que `beta_aguda` y `beta_28` usan Theil-Sen y no OLS. Los tests `test_beta_ruido_retorna_cero`, `test_beta_caida_lineal_es_negativa`, y `test_beta_robusto_a_outlier` especificados en el plan **no están presentes** en `tests/test_services.py`.

---

### 2.4 Task 3: Shapiro-Wilk Controla el Estimador para ACWR, `delta_pct` y `z_meso`

**Estado de implementación:** ✅ **COMPLETA — SIN TESTS DEL ACWR ADAPTATIVO**

La implementación en `services.py` (líneas 214-249) es correcta:

```python
# Líneas 214-229: ACWR con estimador adaptativo
_arr_completo = vmp_daily.dropna().values
if len(_arr_completo) >= 8:
    try:
        _, _p_sw = _shapiro(_arr_completo)
        _usar_mediana = _p_sw <= 0.05
    except Exception:
        _usar_mediana = False
else:
    _usar_mediana = True

if _usar_mediana:
    mma7_s  = vmp_daily.rolling("7D",  min_periods=mp_7d).median()
    mmc28_s = vmp_daily.rolling("28D", min_periods=mp_28d).median()
else:
    mma7_s  = vmp_daily.rolling("7D",  min_periods=mp_7d).mean()
    mmc28_s = vmp_daily.rolling("28D", min_periods=mp_28d).mean()
```

El `z_meso` (líneas 242-249) también usa `estimar_centro_dispersion()` correctamente. Sin embargo, los tests especificados en el plan (`test_acwr_robusto_a_sesion_atipica`, `test_zmeso_usa_mediana_mad_con_no_normalidad`, `test_shapiro_normal_usa_media`) **no existen** en el test suite. Solo se verifica indirectamente a través de los tests de integración.

**Observación de calidad:** El código usa `rolling("7D")` con `min_periods` adaptativo basado en frecuencia de entrenamiento (líneas 208-211), lo cual es una mejora respecto al plan que asumía `rolling(7)`. Esta es una **mejora legítima** no documentada en el plan.

---

### 2.5 Task 4: Corregir Reglas Problemáticas

**Estado de implementación:** ⚠️ **IMPLEMENTADA PERO INOPERANTE POR BUG-001**

Las correcciones de reglas están implementadas en `fuzzy_engine.py`:

| Regla | Estado en plan | Estado en código | Líneas |
|---|---|---|---|
| **Regla 25** (Recuperación→óptimo) | Requiere `vmp[funcional\|alta]` + `acwr[optimo\|bajo]` | ✅ Implementada (146-150) | 146-150 |
| Regla 25b (Recuperación + vmp baja) | `alerta_temprana` | ✅ Implementada (151-154) | 151-154 |
| **Reglas 5-6** (ACWR bajo→crítico) | Rebajado a `fatiga_acumulada` | ✅ Implementadas (116-117) | 116-117 |

Las reglas adicionales de vmp_hoy del Task 5 también están presentes (líneas 109-114). El problema es que **ninguna de estas reglas se ejecuta nunca** debido a BUG-001 (falta de input para `rpe_v`).

---

### 2.6 Task 5: Aumentar Peso Implícito de `vmp_hoy`

**Estado de implementación:** ✅ **COMPLETA — INOPERANTE POR BUG-001**

Las 6 reglas adicionales especificadas en el plan están implementadas en `fuzzy_engine.py`, líneas 109-114:

```python
# Líneas 109-114
ctrl.Rule(vmp_v["muy_baja"] & wellness_v["DEFICIENTE"], fat_v["critico"]),
ctrl.Rule(vmp_v["muy_baja"] & acwr_v["excesivo"], fat_v["critico"]),
ctrl.Rule(vmp_v["muy_baja"] & acwr_v["optimo"], fat_v["fatiga_acumulada"]),
ctrl.Rule(vmp_v["baja"] & acwr_v["excesivo"] & wellness_v["DEFICIENTE"], fat_v["fatiga_acumulada"]),
ctrl.Rule(vmp_v["alta"] & acwr_v["optimo"] & wellness_v["OPTIMO"], fat_v["optimo"]),
ctrl.Rule(vmp_v["alta"] & wellness_v["DEFICIENTE"], fat_v["alerta_temprana"]),
```

El conteo de reglas pasó de 4 a 10 reglas que incluyen `vmp_hoy`, cumpliendo el objetivo del Task. Nuevamente, estas reglas no se ejecutan debido al bug de `rpe_v`.

---

### 2.7 Task 6: Rebalancear Pesos DQI

**Estado de implementación:** ✅ **COMPLETA — SIN TEST**

Los pesos han sido cambiados correctamente en `services.py`, líneas 105-106:

```python
_DQI_W7:  float = 0.55   # mayor peso a calidad reciente
_DQI_W28: float = 0.45  # menor peso al histórico crónico
```

El test `test_dqi_penaliza_ausencia_de_datos_recientes` especificado en el plan **no existe** en el test suite.

---

### 2.8 Task 7: Wellness — Cronbach's Alpha + Pesos Evidenciados

**Estado de implementación:** ✅ **COMPLETA CON TESTS**

Este es el Task con mejor cobertura de tests. `core/wellness.py` tiene:
- Pesos Hooper diferenciados (0.30/0.25/0.20/0.15/0.10) — líneas 17-23
- `calcular_wellness()` con validación de rango [1,7] — líneas 34-55
- `cronbach_alpha_wellness()` con fórmula estándar — líneas 58-76

**Todos los 5 tests pasan:**
- `test_sueno_deficiente_pesa_mas_que_dolor_leve` ✅
- `test_extremos_en_rango_cero_uno` ✅
- `test_valores_invalidos_lanzan_error` ✅
- `test_cronbach_alpha_muestra_coherente` ✅
- `test_cronbach_alpha_muestra_incoherente_es_bajo` ✅

---

### 2.9 Task 8: `analysis.py` — Análisis Bivariado

**Estado de implementación:** ⚠️ **COMPLETO CON BUG EN `cross_correlation_lag`**

Las cuatro funciones están implementadas. Tres de cinco tests pasan. El bug está en `cross_correlation_lag()`:

```python
# Línea 76 de analysis.py (BUG)
raw = _ccf(x, y, nlags=max_lag, adjusted=True)
return {lag: round(float(raw[lag]), 3) for lag in range(max_lag + 1)}
```

El parámetro `nlags` de `statsmodels.tsa.stattools.ccf` especifica el **número** de lags a retornar, no el índice máximo. Si `nlags=5`, retorna 5 elementos (índices 0-4), pero el código intenta acceder `raw[5]` cuando `lag=5` en el `range(max_lag + 1)`.

**Fix requerido:** Cambiar `nlags=max_lag` a `nlags=max_lag + 1`.

---

### 2.10 Task 9: Test de Integración

**Estado de implementación:** ❌ **5 DE 6 TESTS FALLAN POR BUG-001**

| Test | Resultado | Causa |
|---|---|---|
| `test_atleta_optimo_clasifica_correcto` | ❌ Falla (50.0 vs ≥70) | BUG-001: RPE sin input |
| `test_atleta_critico_clasifica_correcto` | ❌ Falla (50.0 vs ≤40) | BUG-001: RPE sin input |
| `test_datos_insuficientes_retorna_insuficiente` | ✅ Pasa | No usa motor fuzzy |
| `test_sin_clavados_planificados` | ✅ Pasa | No usa motor fuzzy |
| `test_critico_con_plan_recuperacion_no_sube_a_optimo` | ❌ Falla (50.0 vs ≤40) | BUG-001: RPE sin input |
| `test_analysis_modulo_sobre_datos_pipeline` | ✅ Pasa | No usa motor fuzzy para el assert |

---

## 3. Bug Crítico: Análisis de BUG-001 (RPE sin Input)

![Flujo del BUG-001](fig_bug_rpe_flujo.png)

### 3.1 Descripción Técnica

El sistema define **10 antecedentes** en `construir_sistema_fuzzy()` (líneas 29-38), incluyendo `rpe_v = ctrl.Antecedent(u_rpe, "carga_subjetiva")`. Luego, `construir_reglas()` crea **3 reglas** que dependen de este antecedente:

| Regla | Condición | Consecuente |
|---|---|---|
| RPE-1 | `rpe_v["alta"] & wellness_v["DEFICIENTE"]` | `fat_v["critico"]` |
| RPE-2 | `rpe_v["alta"] & ci_v["SOBRECARGA"]` | `fat_v["critico"]` |
| RPE-3 | `rpe_v["baja"] & wellness_v["OPTIMO"]` | `fat_v["optimo"]` |

Cuando `evaluar_atleta()` o `pipeline_diagnostico()` intentan ejecutar el motor, asignan valores a **8 de 9 antecedentes** (falta `carga_subjetiva`). Scikit-fuzzy detecta que no todos los antecedentes usados en reglas activas tienen input, lanza una excepción, y el código captura esta excepción retornando el fallback de `50.0`.

### 3.2 Impacto en Producción

**Todas las evaluaciones del motor fuzzy retornan 50.0** ("ALERTA TEMPRANA"), sin importar el estado real del atleta. El motor fuzzy está **completamente inoperante**. Los entrenadores verían "🟡 ALERTA TEMPRANA" para todos los atletas, todo el tiempo.

### 3.3 Causa Raíz

El `rpe_v` fue añadido al sistema fuzzy como parte de las "Reglas RPE nuevas" (líneas 103-106 de `fuzzy_engine.py`), probablemente durante una iteración de desarrollo posterior al plan original, **sin actualizar** las funciones `evaluar_atleta()` y `pipeline_diagnostico()` para asignarle un valor. Es un caso clásico de inconsistencia entre la definición del sistema y su consumo.

### 3.4 Opciones de Corrección

| Opción | Descripción | Complejidad | Riesgo |
|---|---|---|---|
| **A. Eliminar `rpe_v`** | Quitar las 3 reglas RPE y el antecedente del sistema | Baja | Medio — pierde funcionalidad RPE |
| **B. Añadir input RPE** | Pasar `carga_subjetiva` como parámetro a `evaluar_atleta()` y `pipeline_diagnostico()` | Baja | Bajo — requiere cambios en firma y llamadas |
| **C. Hacer RPE opcional** | Asignar un valor por defecto (ej. `simulador.input["carga_subjetiva"] = 5.0`) cuando no se proporcione | Baja | Bajo — mantiene compatibilidad |

**Recomendación:** Opción C como fix inmediato (asignar default neutral), seguido de Opción B para propósito completo. Las reglas RPE deben recibir la carga subjetiva real del atleta para ser útiles.

---

## 4. Deudas Técnicas: Integración con Hallazgos de Auditorías 1 y 2

Las auditorías adjuntas (`1.txt` y `2.txt`) identifican **6 deudas técnicas (TD-006 a TD-011)** que no aparecen en el plan de corrección v2. A continuación se evalúa cada una y se propone cómo integrarla.

![Catálogo de Deudas Técnicas](fig_deudas_tecnicas.png)

### 4.1 TD-006: Migrar ACWR de Rolling Average a EWMA

**Origen:** Auditoría 2, Sección 2  
**Prioridad:** Media  
**Justificación:** Griffin et al. (2019) y Wang et al. (2020) documentan que EWMA supera a RA en sensibilidad para detectar cambios agudos. El plan v2 reconoce esto como "deuda técnica registrable" pero no lo incluye como task.

**Propuesta de integración:** Añadir como **Task 10**. Implementar función `ewma_acwr()` en `stats_utils.py` con factor de decaimiento λ adaptado a la frecuencia de entrenamiento del equipo. El EWMA debe coexistir con RA durante una fase de transición para comparar resultados.

### 4.2 TD-007: Calibración Empírica de Pesos Wellness

**Origen:** Auditoría 2, Sección 3  
**Prioridad:** Media  
**Justificación:** Los pesos 0.30/0.25/0.20/0.15/0.10 son una formalización razonable pero no derivados de datos propios. Campbell et al. (2021) muestran que el wellness tiene baja predictibilidad desde carga objetiva.

**Propuesta de integración:** Añadir como **Task 11**. Ejecutar `cronbach_alpha_wellness()` + análisis de correlación Spearman de cada ítem con `VMP-delta` sobre datos históricos del equipo. Recomendar pesos empíricos sin cambiar los actuales hasta tener ≥50 observaciones.

### 4.3 TD-008: Protocolo de Validación Interna

**Origen:** Auditoría 2, Sección 5  
**Prioridad:** Alta  
**Justificación:** La validez del sistema completo depende de que el índice de fatiga prediga el VMP del día siguiente. Sin validación interna, no hay evidencia de que el modelo funcione para clavadistas.

**Propuesta de integración:** Añadir como **Task 12**. Implementar `cross_correlation_lag(acwr_series, vmp_series, max_lag=7)` sobre datos históricos del equipo para verificar el desfase temporal real. Corregir primero BUG-002 (CCF index out of bounds).

### 4.4 TD-009: Renombrar `acwr` como `vmp_ratio` + Añadir `acwr_carga`

**Origen:** Auditoría 1, Problemas 1 y 2  
**Prioridad:** Alta  
**Justificación:** El `acwr` actual es un **ratio de rendimiento** (VMP 7d/28d), no un ACWR de carga. Los umbrales 0.8-1.3 están calibrados sobre unidades de carga (sRPE), no sobre m/s. Además, el modelo carece del verdadero ACWR de carga (`sRPE_load_7d / sRPE_load_28d`), que es el que la literatura vincula con riesgo de lesión.

**Propuesta de integración:** Añadir como **Task 13**. Renombrar la variable interna `acwr` → `vmp_ratio` para claridad conceptual. Calcular `acwr_carga` desde el historial de `carga_subjetiva × duracion_min`. Añadir `acwr_carga` como nuevo antecedente del motor con sus propias membresías. Esto es un cambio **arquitectónico mayor** que afecta el nombre de la variable en toda la base de datos y la UI.

### 4.5 TD-010: Modelar Desfase Temporal de `carga_integrada_plan`

**Origen:** Auditoría 1, Problema 3  
**Prioridad:** Alta  
**Justificación:** Crewther et al. (2024) muestran que el efecto de la carga sobre bienestar tiene pico en lag 1-2 días. La carga del día actual impacta el VMP en t+1 a t+3 días, pero el motor la trata como si fuera simultánea.

**Propuesta de integración:** Añadir como **Task 14**. Implementar `cross_correlation_lag(carga_series, vmp_series)` para determinar el lag predominante. Separar el índice de fatiga actual (retrospectivo) de un "índice de riesgo prospectivo" que use la carga planificada. Esto requiere un consecuente adicional en el motor o un módulo separado.

### 4.6 TD-011: Revisar Reglas con `acwr_v["excesivo"]`

**Origen:** Auditoría 1, Problema 1 y Análisis de Reglas Grupo 3  
**Prioridad:** Media  
**Justificación:** Las reglas que combinan VMP-ACWR "excesivo" (alto rendimiento) con señales de fatiga son fisiológicamente inconsistentes. Un ACWR excesivo en VMP significa supercompensación/peak de forma, no sobrecarga.

**Reglas a revisar en `fuzzy_engine.py`:**
- Línea 119: `acwr_v["excesivo"] & zmeso_v["muy_bajo"] & ba_v["neg_fuerte"] → fat_v["critico"]`
- Línea 126: `acwr_v["excesivo"] & zmeso_v["normal"] → fat_v["fatiga_acumulada"]`
- Línea 128: `acwr_v["excesivo"] & (zmeso_v["bajo"] | zmeso_v["muy_bajo"]) & ba_v["neg_moderada"] → fat_v["fatiga_acumulada"]`

**Propuesta de integración:** Añadir como **Task 15**. Reclasificar estas reglas como patrón de "colapso post-pico" con consecuente `alerta_temprana` en lugar de `critico`/`fatiga_acumulada`. Validar primero si este patrón ocurre en datos del equipo.

---

## 5. Gaps Adicionales No Cubiertos por el Plan

### 5.1 GAP-001: Tests Insuficientes en `services.py`

El archivo `tests/test_services.py` contiene solo **2 tests**, ambos muy básicos (datos insuficientes y suficientes). Faltan los tests especificados en el plan para:

- `test_carga_integrada_no_depende_de_wellness` (Task 1)
- `test_beta_ruido_retorna_cero` (Task 2)
- `test_beta_caida_lineal_es_negativa` (Task 2)
- `test_beta_robusto_a_outlier` (Task 2)
- `test_acwr_robusto_a_sesion_atipica` (Task 3)
- `test_zmeso_usa_mediana_mad_con_no_normalidad` (Task 3)
- `test_shapiro_normal_usa_media` (Task 3)
- `test_dqi_penaliza_ausencia_de_datos_recientes` (Task 6)

**Impacto:** Sin estos tests, los fixes de los Tasks 1-3 y 6 no tienen cobertura de regresión.

### 5.2 GAP-002: `delta_pct` con Signo Invertido

En `services.py`, línea 238:

```python
delta_pct = float(np.clip(((mmc28 - last_vmp) / mmc28) * 100 if mmc28 > 0 else 0.0, -20, 40))
```

La fórmula calcula `(MMC28 - VMP_hoy) / MMC28 × 100`. Si el VMP de hoy es **mayor** que el promedio crónico (supercompensación), `delta_pct` será **negativo** (señal de "ganancia"). Si el VMP de hoy es **menor** (fatiga), `delta_pct` será **positivo** (señal de "alarma").

Esto es **contraintuitivo** pero **consistente** con las membresías del motor difuso, donde:
- `delta_v["ganancia"]` cubre valores negativos [-20, 0]
- `delta_v["alarma"]` cubre valores positivos [18, 40]

Sin embargo, la documentación del plan no menciona esta inversión de signo. Un desarrollador nuevo podría interpretar `delta_pct = 25.0` como "25% de mejora" cuando en realidad significa "25% de caída respecto al baseline". Se recomienda documentar explícitamente esta convención.

---

## 6. Matriz de Discrepancias: Plan vs. Implementación

![Matriz de Discrepancias](fig_matriz_discrepancias.png)

La matriz anterior resume el estado de cada Task del plan a través de seis dimensiones de calidad. Las celdas en rojo (✗) indican problemas que deben resolverse antes del despliegue. Las celdas en amarillo (~) indican áreas que funcionan pero con advertencias. Las celdas grises (—) representan dimensiones no aplicables a ese Task específico.

---

## 7. Propuesta de Plan de Corrección v3

### 7.1 Fase 1: Bugs Críticos (Bloqueantes para Producción)

| ID | Descripción | Archivos | Esfuerzo estimado |
|---|---|---|---|
| **BUG-001** | Añadir `simulador.input["carga_subjetiva"]` en `evaluar_atleta()` y `pipeline_diagnostico()`. Valor por defecto: 5.0 (RPE media) cuando no se proporcione | `fuzzy_engine.py`, `services.py`, `app.py` | 2h |
| **BUG-002** | Corregir `cross_correlation_lag`: cambiar `nlags=max_lag` a `nlags=max_lag + 1` | `analysis.py` | 15min |
| **BUG-003** | Corregir test Theil-Sen: usar semilla garantizada o test probabilístico | `tests/test_stats_utils.py` | 30min |

### 7.2 Fase 2: Tests Faltantes

| ID | Descripción | Archivo de test |
|---|---|---|
| TEST-T1 | Verificar que wellness no modifica carga_integrada | `tests/test_services.py` |
| TEST-T2a,b,c | Verificar Theil-Sen en beta (ruido, caída, outlier) | `tests/test_services.py` |
| TEST-T3a,b,c | Verificar ACWR adaptativo y z_meso robusto | `tests/test_services.py` |
| TEST-T6 | Verificar DQI rebalanceado | `tests/test_services.py` |

### 7.3 Fase 3: Deudas Técnicas (Integración con Auditorías)

| ID | Descripción | Prioridad | Origen |
|---|---|---|---|
| **Task 10** | Implementar EWMA para ACWR (coexistir con RA) | Media | TD-006 |
| **Task 11** | Calibrar pesos wellness con datos del equipo | Media | TD-007 |
| **Task 12** | Protocolo de validación interna (cross-correlation) | Alta | TD-008 |
| **Task 13** | Renombrar `acwr`→`vmp_ratio` + añadir `acwr_carga` | Alta | TD-009 |
| **Task 14** | Modelar desfase temporal carga→VMP | Alta | TD-010 |
| **Task 15** | Revisar/reclasificar reglas con `acwr["excesivo"]` | Media | TD-011 |

### 7.4 Fase 4: Mejoras Arquitectónicas

| ID | Descripción | Justificación |
|---|---|---|
| ARCH-001 | Centralizar todos los inputs del motor en un diccionario/typed dict | Previere bugs como BUG-001 donde se olvida un input |
| ARCH-002 | Separar `evaluar_atleta()` en: (a) preprocesamiento, (b) ejecución motor, (c) postprocesamiento | Facilita testing unitario de cada fase |
| ARCH-003 | Añadir validación de antecedentes antes de `simulador.compute()` | Detectar "antecedentes sin input" antes de que falle scikit-fuzzy |

---

## 8. Conclusiones y Recomendaciones

### 8.1 Hallazgo Principal

El plan de corrección v2 representa un **buen diseño técnico**: las decisiones estadísticas (Theil-Sen, MAD, Shapiro-Wilk adaptativo) están bien fundamentadas, las reglas corregidas son fisiológicamente coherentes, y la arquitectura de módulos es limpia. El problema no está en el diseño, sino en la **ejecución incompleta y la falta de cobertura de tests**.

### 8.2 Problema Crítico

El **BUG-001 (RPE sin input)** es un error de integración que inutiliza completamente el motor fuzzy. Es un recordatorio de que incluso los mejores diseños pueden fallar por inconsistencias entre la definición de un sistema y su consumo. La lección clave es que **toda adición de antecedente al motor debe ir acompañada de su correspondiente asignación de input** y un test de integración que verifique el comportamiento end-to-end.

### 8.3 Recomendaciones para el Plan v3

1. **Incluir un "checklist de integridad del motor"** que verifique que todos los antecedentes definidos en `construir_sistema_fuzzy()` tienen asignación de input en `evaluar_atleta()` y `pipeline_diagnostico()`.

2. **Establecer un mínimo de cobertura de tests por Task**: cada Task debe incluir al menos un test unitario y un test de integración.

3. **Integrar las 6 deudas técnicas de las auditorías previas** como Tasks adicionales, con priorización basada en el análisis de riesgo presentado.

4. **Implementar validación pre-compute**: antes de llamar `simulador.compute()`, verificar que todos los antecedentes requeridos tienen valores asignados, y si no, lanzar un error informativo en lugar de caer al fallback silencioso.

5. **Documentar la convención de signo de `delta_pct`** para evitar confusiones futuras.

---

*Este informe fue generado mediante análisis automatizado del repositorio, ejecución de tests, y revisión manual del código fuente. Todos los hallazgos son verificables mediante los comandos y referencias de línea proporcionadas.*
