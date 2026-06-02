# Módulo de Gestión de Lesiones - Plan de Implementación

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactorizar el módulo de lesiones para incluir hitos biológicos del retorno a la competición (RTP) y una clasificación biomecánica más precisa.

**Architecture:** Se extenderá el modelo de datos (SQL + Python Dataclasses) y se reorganizará el formulario de registro en la UI utilizando un layout de 3 columnas para mejorar la usabilidad.

**Tech Stack:** Python (Streamlit), Supabase (PostgreSQL), Dataclasses.

---

- [completed] Task 1: Migración de Base de Datos

**Files:**
- Create: `docs/migrations/add_lesiones_refactor.sql`

- [x] **Step 1: Crear archivo de migración SQL**

```sql
-- Agregar campos de clasificación biomecánica y hitos RTP
ALTER TABLE lesiones 
ADD COLUMN IF NOT EXISTS tipo_tejido TEXT,
ADD COLUMN IF NOT EXISTS mecanismo TEXT,
ADD COLUMN IF NOT EXISTS recurrencia TEXT,
ADD COLUMN IF NOT EXISTS mecanismo_contacto BOOLEAN DEFAULT FALSE,
ADD COLUMN IF NOT EXISTS fecha_evento DATE,
ADD COLUMN IF NOT EXISTS fecha_alta_medica DATE,
ADD COLUMN IF NOT EXISTS fecha_rtt DATE,
ADD COLUMN IF NOT EXISTS fecha_rtp DATE;
```

---

- [completed] Task 2: Actualización de Esquemas (Core)

**Files:**
- Modify: `core/schemas.py`

- [x] **Step 1: Definir nuevos Enums**

```python
from enum import Enum

class TipoTejido(str, Enum):
    MUSCULO = "musculo"
    TENDON = "tendon"
    LIGAMENTO = "ligamento"
    OTRO = "otro"

class MecanismoInicio(str, Enum):
    AGUDA = "aguda"
    SOBREUSO = "sobreuso"

class HistorialRecurrencia(str, Enum):
    NUEVA = "nueva"
    RECURRENCIA = "recurrencia"
```

- [x] **Step 2: Actualizar InjuryInput**

Modificar `InjuryInput` en `core/schemas.py` para incluir los nuevos campos y actualizar las validaciones en `__post_init__`.

---

- [completed] Task 3: Actualización de persistencia y servicios

**Files:**
- Modify: `data/db.py`
- Modify: `core/services.py`

- [x] **Step 1: Actualizar `insertar_lesion` en `data/db.py`**
Actualizar la función para mapear los nuevos campos desde el dict `InjuryInput`.

- [x] **Step 2: Actualizar `cargar_lesiones_activas` y `cargar_historial_lesiones`**
Asegurar que los nuevos campos se seleccionen y mapeen correctamente.

---

- [completed] Task 4: Refactorización UI (Streamlit)

**Files:**
- Modify: `components/tab_lesiones.py`

- [x] **Step 1: Reorganizar Formulario de Registro (`_render_form_registro`)**
Implementar el layout de 3 columnas propuesto en el diseño.

- [x] **Step 2: Actualizar vista de Seguimiento (`_render_seguimiento_activo`)**
Asegurar que los nuevos campos sean visibles o editables si aplica.

---

- [completed] Task 5: Testing

**Files:**
- Modify: `tests/test_injury_schemas.py`
- Modify: `tests/test_injury_services.py`

- [x] **Step 1: Actualizar tests de validación de esquemas**
Añadir casos de prueba para los nuevos campos de `InjuryInput`.

- [x] **Step 2: Añadir tests de integración para persistencia**
Verificar que la inserción y carga de lesiones incluya correctamente los nuevos campos.
