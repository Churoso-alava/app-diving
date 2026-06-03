# Fuzzy Model — Auditoría y Corrección v3 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Resolver los bloqueantes críticos de producción (BUG-001), corregir errores estadísticos (BUG-002, BUG-003), cerrar brechas de testing (GAP-001, GAP-002) e integrar las deudas técnicas arquitectónicas y fisiológicas (TD-006 a TD-011) identificadas en la auditoría del 2026-06-03.

**Architecture:** 
1. **Estabilidad Inmediata:** Corrección de inputs del motor y validación pre-compute para evitar fallos silenciosos.
2. **Integridad Estadística:** Ajuste de CCF y tests de Theil-Sen.
3. **Claridad Conceptual:** Renombrado de `acwr` (rendimiento) a `vmp_ratio` e introducción del verdadero `acwr_carga`.
4. **Refinamiento Fisiológico:** EWMA para ratios, modelado de desfase temporal y recalibración de reglas inconsistentes.

**Tech Stack:** Python 3.12, scikit-fuzzy 0.4, scipy, numpy, pandas, statsmodels, pytest.

---

## Task 1: Resolver BUG-001 (RPE sin Input) y ARCH-003 (Validación Pre-compute)

**Problema:** El motor define `carga_subjetiva` pero no recibe el valor en `evaluar_atleta`, lo que causa que todas las evaluaciones caigan al fallback de 50.0.
**Solución:** Mapear el input de RPE y añadir un validador que asegure que todos los antecedentes tengan valor antes de computar.

**Files:**
- Modify: `core/fuzzy_engine.py`
- Modify: `core/services.py`
- Test: `tests/test_fuzzy_engine.py`

- [ ] **Step 1: Escribir test de integración que exponga el fallo de 50.0**

```python
def test_motor_no_cae_en_fallback_50(motor):
    # Metricas normales que deberian dar algo distinto a 50.0 (ej. optimo > 70)
    m = _metricas_base(vmp_hoy=1.40, acwr=1.10) 
    res = evaluar_atleta(motor, m, wellness_norm=0.9, carga_integrada_plan=50.0)
    assert res["indice_fatiga"] != 50.0, "BUG-001: El motor sigue cayendo en fallback 50.0"
```

- [ ] **Step 2: Ejecutar test y verificar que falla (retorna 50.0)**

- [ ] **Step 3: Implementar validación y mapeo en `core/fuzzy_engine.py`**

```python
# En evaluar_atleta()
simulador.input["carga_subjetiva"] = metricas.get("rpe_sesion", 5.0) # Default neutral

# Añadir validación antes de simulador.compute()
for ant in sistema.antecedents:
    if ant.label not in simulador.input:
        raise ValueError(f"Falta input para antecedente: {ant.label}")
```

- [ ] **Step 4: Actualizar `pipeline_diagnostico` en `core/services.py` para extraer RPE**

- [ ] **Step 5: Verificar que el test pasa y el índice ya no es 50.0 constante**

---

## Task 2: BUG-002 (CCF Index Error) y BUG-003 (Test Theil-Sen frágil)

**Problema:** `cross_correlation_lag` intenta acceder a un índice inexistente. El test de Theil-Sen falla aleatoriamente por ruido uniforme.

**Files:**
- Modify: `core/analysis.py`
- Modify: `tests/test_stats_utils.py`

- [ ] **Step 1: Corregir BUG-002 en `core/analysis.py`**
Cambiar `nlags=max_lag` por `nlags=max_lag + 1` en la llamada a `_ccf`.

- [ ] **Step 2: Corregir BUG-003 en `tests/test_stats_utils.py`**
Usar `np.random.seed(99)` o una serie fija para asegurar que el ruido sea capturado por el IC de 0.

- [ ] **Step 3: Ejecutar `pytest tests/test_analysis.py tests/test_stats_utils.py` y verificar éxito**

---

## Task 3: Cerrar GAP-001 (Tests de Regresión para Services)

**Problema:** Faltan tests unitarios para los fixes implementados en Plan v2 (Theil-Sen, ACWR adaptativo, etc.).

**Files:**
- Modify: `tests/test_services.py`

- [ ] **Step 1: Implementar `test_carga_integrada_no_depende_de_wellness`**
- [ ] **Step 2: Implementar `test_beta_ruido_retorna_cero` y `test_beta_robusto_a_outlier`**
- [ ] **Step 3: Implementar `test_acwr_robusto_a_sesion_atipica`**
- [ ] **Step 4: Ejecutar tests y asegurar cobertura de regresión**

---

## Task 4: TD-009 (Renombrar `acwr` → `vmp_ratio` e introducir `acwr_carga`)

**Problema:** El ACWR actual mide rendimiento (VMP), no carga. Los umbrales de la literatura no aplican.
**Solución:** Renombrar la variable actual y calcular el verdadero ACWR de carga (sRPE_load).

**Files:**
- Modify: `core/services.py`
- Modify: `core/fuzzy_engine.py` (nombres de antecedentes)
- Modify: `core/schemas.py` (si aplica a la salida)

- [ ] **Step 1: Renombrar variable interna en `services.py`**
`acwr` -> `vmp_ratio`.

- [ ] **Step 2: Implementar cálculo de `acwr_carga`**
Calcular sRPE_load = RPE * duracion. Luego `MMA7_carga / MMC28_carga`.

- [ ] **Step 3: Actualizar motor fuzzy en `fuzzy_engine.py`**
Añadir antecedente `acwr_carga_v` con umbrales 0.8 - 1.3 (óptimo), > 1.5 (excesivo).

- [ ] **Step 4: Testear que ambas señales entran al motor correctamente**

---

## Task 5: TD-006 (Migrar VMP-Ratio a EWMA)

**Problema:** El promedio rodante (RA) es menos sensible que EWMA para detectar cambios agudos.

**Files:**
- Modify: `core/stats_utils.py`
- Modify: `core/services.py`

- [ ] **Step 1: Crear función `ewma_acwr` en `stats_utils.py`**
- [ ] **Step 2: Reemplazar `.rolling().mean()` por la llamada a EWMA en `services.py`**
- [ ] **Step 3: Verificar estabilidad con test `test_acwr_robusto_a_sesion_atipica`**

---

## Task 6: TD-011 (Reclasificar reglas de `acwr_v["excesivo"]`)

**Problema:** VMP_ACWR > 1.5 significa supercompensación, no fatiga crítica.

**Files:**
- Modify: `core/fuzzy_engine.py`

- [ ] **Step 1: Identificar reglas de "excesivo" en `construir_reglas()`**
- [ ] **Step 2: Cambiar consecuente de `critico` a `optimo` o `alerta_temprana` (post-pico)**
- [ ] **Step 3: Testear con un "atleta en pico" que el índice sea alto (>70)**

---

## Task 7: TD-010 (Modelar Desfase Temporal Carga → VMP)

**Problema:** La carga de hoy impacta el VMP en t+1 o t+2. El motor los mezcla en t=0.

**Files:**
- Modify: `core/analysis.py`
- Modify: `core/services.py`

- [ ] **Step 1: Usar `cross_correlation_lag` para reportar el lag óptimo en los logs**
- [ ] **Step 2: Implementar `indice_riesgo_prospectivo` que use la carga planificada de hoy vs el estado de ayer**
- [ ] **Step 3: Separar visualmente o en el schema la fatiga actual del riesgo futuro**

---

## Task 8: TD-007 (Calibración Empírica Wellness)

**Problema:** Los pesos del wellness (0.30, 0.25...) son teóricos.

**Files:**
- Modify: `core/wellness.py`
- Modify: `core/analysis.py`

- [ ] **Step 1: Implementar script/función que ejecute Spearman entre ítems de wellness y delta_VMP**
- [ ] **Step 2: Actualizar `_PESOS_HOOPER` si los datos del equipo muestran divergencias significativas**
- [ ] **Step 3: Testear con Cronbach α que la escala es fiable**

---

## Task 9: Refactor Arquitectónico y Validación Final

**Files:**
- Modify: `core/services.py`
- Modify: `tests/test_integration.py`

- [ ] **Step 1: ARCH-001: Centralizar inputs en un TypedDict/Mapeador único**
- [ ] **Step 2: ARCH-002: Separar preprocesamiento de ejecución en `evaluar_atleta`**
- [ ] **Step 3: Ejecución de `pytest tests/` completa (0 fallos esperados)**
- [ ] **Step 4: Commit final y reporte de cierre**

---
