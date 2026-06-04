# Corrección Registro Grupal: Normalización de Inicialización

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Modificar los valores por defecto en el registro grupal de RPE y VMP para evitar registros fantasma y asegurar que solo se guarden las entradas explícitas del usuario.

**Architecture:** Modificar los DataFrames de inicialización en `app.py` para establecer valores de RPE/VMP en 0 y Duración en 240, confiando en la lógica de filtrado existente (`if value > 0`) para prevenir guardados erróneos.

**Tech Stack:** Python, Pandas, Streamlit.

---

### Task 1: Ajustar inicialización de Carga Grupal (RPE/Duración)

**Files:**
- Modify: `app.py` (dentro de la sección de registro grupal de carga)
- Test: Manual (verificar UI al abrir registro grupal)

- [ ] **Step 1: Localizar la inicialización del DataFrame `df_carga_grupal` en `app.py`**
Aproximadamente en la línea 304.

- [ ] **Step 2: Modificar los valores por defecto**

```python
# app.py, línea 305 aprox.
# Cambiar:
"RPE (1-10)": [5] * len(atletas_carga),
"Duración (min)": [60] * len(atletas_carga)

# Por:
"RPE (1-10)": [0] * len(atletas_carga),
"Duración (min)": [240] * len(atletas_carga)
```

- [ ] **Step 3: Verificar que el `data_editor` renderice los nuevos ceros y 240**
Correr la aplicación `streamlit run app.py` y navegar a la pestaña "Carga de Entrenamiento", seleccionar "Grupal" y verificar los valores.

- [ ] **Step 4: Commit**
```bash
git add app.py
git commit -m "feat: normalizar inicialización registro carga grupal (RPE=0, Dur=240)"
```

### Task 2: Ajustar inicialización de Registro Grupal VMP

**Files:**
- Modify: `app.py` (dentro de la sección de registro grupal de VMP)
- Test: Manual (verificar UI al abrir registro grupal VMP)

- [ ] **Step 1: Localizar la inicialización del DataFrame `df_grupal` en `app.py`**
Aproximadamente en la línea 233.

- [ ] **Step 2: Modificar los valores por defecto**

```python
# app.py, línea 233 aprox.
# Cambiar:
"VMP Hoy": [1.0] * len(atletas_list)

# Por:
"VMP Hoy": [0.0] * len(atletas_list)
```

- [ ] **Step 3: Verificar que el `data_editor` renderice los nuevos valores 0.0**
Correr la aplicación `streamlit run app.py` y navegar a la pestaña correspondiente (SALTO CMJ), seleccionar "Grupal" y verificar los valores.

- [ ] **Step 4: Commit**
```bash
git add app.py
git commit -m "feat: normalizar inicialización registro grupal VMP (VMP=0.0)"
```

### Task 3: Verificación final de integridad de guardado

**Files:**
- Modify: N/A
- Test: Intentar guardar registros grupales (tanto carga como VMP) asegurando que solo los modificados se guardan.

- [ ] **Step 1: Probar guardado de Carga (RPE/Duración)**
En la UI, dejar algunos atletas en 0 y modificar otros. Guardar y verificar en la BD/logs que solo se guardaron los atletas modificados.

- [ ] **Step 2: Probar guardado de VMP**
En la UI, dejar algunos atletas en 0.0 y modificar otros. Guardar y verificar en la BD/logs que solo se guardaron los atletas modificados.
