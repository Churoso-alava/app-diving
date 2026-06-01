# Refactor Módulo Lesiones Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement new RTP (Return to Play) milestones and biomechanical classification fields in the injury management module.

**Architecture:** Database extension via SQL ALTER statements, schema updates in `core/schemas.py`, and UI refactor in `components/tab_lesiones.py` using a 3-column layout.

**Tech Stack:** Python, Streamlit, SQL (SQLite).

---

### Task 1: Update Database Schema

**Files:**
- Modify: `docs/migrations/add_carga_entrenamiento.sql` (Add new columns)
- Modify: `data/db.py` (Ensure DB connection handles migration or update script)

- [ ] **Step 1: Write SQL migration**

```sql
-- Add new columns to lesiones table
ALTER TABLE lesiones ADD COLUMN tipo_tejido TEXT;
ALTER TABLE lesiones ADD COLUMN mecanismo TEXT;
ALTER TABLE lesiones ADD COLUMN recurrencia TEXT;
ALTER TABLE lesiones ADD COLUMN mecanismo_contacto BOOLEAN DEFAULT FALSE;
ALTER TABLE lesiones ADD COLUMN fecha_evento DATE;
ALTER TABLE lesiones ADD COLUMN fecha_alta_medica DATE;
ALTER TABLE lesiones ADD COLUMN fecha_rtt DATE;
ALTER TABLE lesiones ADD COLUMN fecha_rtp DATE;
```

- [ ] **Step 2: Commit**

```bash
git add docs/migrations/add_carga_entrenamiento.sql
git commit -m "feat: add schema migration for injury module"
```

### Task 2: Update Data Models

**Files:**
- Modify: `core/schemas.py`

- [ ] **Step 1: Write test for new schemas**

```python
import pytest
from core.schemas import InjuryInput, TipoTejido, MecanismoInicio, HistorialRecurrencia
from datetime import date

def test_injury_input_validation():
    data = {
        "tipo_tejido": "musculo",
        "mecanismo": "aguda",
        "recurrencia": "nueva",
        "fecha_evento": date(2026, 6, 1)
    }
    injury = InjuryInput(**data)
    assert injury.tipo_tejido == TipoTejido.musculo
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_injury_schemas.py`
Expected: FAIL

- [ ] **Step 3: Update `core/schemas.py`**

```python
from enum import Enum
from pydantic import BaseModel, Field
from datetime import date
from typing import Optional

class TipoTejido(str, Enum):
    musculo = "musculo"
    tendon = "tendon"
    ligamento = "ligamento"
    otro = "otro"

class MecanismoInicio(str, Enum):
    aguda = "aguda"
    sobreuso = "sobreuso"

class HistorialRecurrencia(str, Enum):
    nueva = "nueva"
    recurrencia = "recurrencia"

class InjuryInput(BaseModel):
    # ... existing fields
    tipo_tejido: Optional[TipoTejido] = None
    mecanismo: Optional[MecanismoInicio] = None
    recurrencia: Optional[HistorialRecurrencia] = None
    mecanismo_contacto: bool = False
    fecha_evento: Optional[date] = None
    fecha_alta_medica: Optional[date] = None
    fecha_rtt: Optional[date] = None
    fecha_rtp: Optional[date] = None
    
    class Config:
        use_enum_values = True
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_injury_schemas.py`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add core/schemas.py tests/test_injury_schemas.py
git commit -m "feat: add injury schema models"
```

### Task 3: Update UI Form

**Files:**
- Modify: `components/tab_lesiones.py`

- [ ] **Step 1: Update UI layout to 3 columns**

```python
import streamlit as st

def render_lesiones_tab():
    with st.form("lesiones_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("General")
            # ... inputs for atleta, fecha, zona
        
        with col2:
            st.subheader("Clasificación")
            # ... inputs for tipo_tejido, mecanismo, recurrencia, contacto
        
        with col3:
            st.subheader("Hitos RTP")
            # ... inputs for fechas
            
        submit = st.form_submit_button("Registrar")
```

- [ ] **Step 2: Commit**

```bash
git add components/tab_lesiones.py
git commit -m "feat: update injury tab UI"
```

### Task 4: Integration Tests

**Files:**
- Modify: `tests/test_injury_integration.py`

- [ ] **Step 1: Add integration test for new fields**

```python
def test_injury_registration_with_new_fields(db_session):
    # ... setup
    # ... call registration service
    # ... assert new fields in db
```

- [ ] **Step 2: Commit**

```bash
git add tests/test_injury_integration.py
git commit -m "feat: add integration tests for injury refactor"
```
