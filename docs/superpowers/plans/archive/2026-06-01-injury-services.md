# Injury Services Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement injury management service functions in `core/services.py` and ensure they are tested.

**Architecture:** Add service-layer functions in `core/services.py` that act as a bridge between the business logic and the data layer (`data.db`). Each function will handle input/output and simple validation.

**Tech Stack:** Python, Pandas.

---

### Task 1: Initialize Tests

**Files:**
- Create: `tests/test_injury_services.py`

- [ ] **Step 1: Create the test file**

```python
import pytest
import pandas as pd
from datetime import date
from core.services import (
    registrar_lesion_servicio,
    obtener_lesiones_activas_servicio,
    obtener_historial_lesiones_servicio,
    actualizar_estado_lesion_servicio
)
# We will mock the database functions in future tasks.
```

- [ ] **Step 2: Commit**

```bash
git add tests/test_injury_services.py
git commit -m "feat: initialize test_injury_services.py"
```

### Task 2: Implement registrar_lesion_servicio

**Files:**
- Modify: `core/services.py`
- Test: `tests/test_injury_services.py`

- [ ] **Step 1: Write the failing test**

```python
from unittest.mock import patch, MagicMock
from core.schemas import InjuryInput

@patch('core.services.insertar_lesion')
def test_registrar_lesion_servicio_success(mock_insertar):
    mock_insertar.return_value = (True, "OK")
    data = InjuryInput(atleta="Test Athlete", tipo="Muscle", severidad=1, descripcion="Test")
    success, msg = registrar_lesion_servicio(data)
    assert success is True
    assert msg == "OK"
    mock_insertar.assert_called_once()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_injury_services.py`
Expected: FAIL with "ImportError" or "NameError" (function not defined)

- [ ] **Step 3: Write implementation**

Modify `core/services.py`:
```python
from core.schemas import InjuryInput
from data.db import insertar_lesion
from typing import Tuple

def registrar_lesion_servicio(data: InjuryInput) -> Tuple[bool, str]:
    return insertar_lesion(data)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_injury_services.py`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add core/services.py tests/test_injury_services.py
git commit -m "feat: implement registrar_lesion_servicio"
```

### Task 3: Implement obtener_lesiones_activas_servicio

**Files:**
- Modify: `core/services.py`
- Test: `tests/test_injury_services.py`

- [ ] **Step 1: Write the failing test**

```python
@patch('core.services.cargar_lesiones_activas')
def test_obtener_lesiones_activas_servicio(mock_cargar):
    mock_df = pd.DataFrame({"id": [1], "atleta": ["Test"]})
    mock_cargar.return_value = mock_df
    result = obtener_lesiones_activas_servicio(atleta="Test")
    pd.testing.assert_frame_equal(result, mock_df)
    mock_cargar.assert_called_once_with(atleta="Test")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_injury_services.py`
Expected: FAIL (function not defined)

- [ ] **Step 3: Write implementation**

Modify `core/services.py`:
```python
from typing import Optional
from data.db import cargar_lesiones_activas

def obtener_lesiones_activas_servicio(atleta: Optional[str] = None) -> pd.DataFrame:
    return cargar_lesiones_activas(atleta=atleta)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_injury_services.py`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add core/services.py tests/test_injury_services.py
git commit -m "feat: implement obtener_lesiones_activas_servicio"
```

### Task 4: Implement obtener_historial_lesiones_servicio

**Files:**
- Modify: `core/services.py`
- Test: `tests/test_injury_services.py`

- [ ] **Step 1: Write the failing test**

```python
@patch('core.services.cargar_historial_lesiones')
def test_obtener_historial_lesiones_servicio(mock_cargar):
    mock_df = pd.DataFrame({"id": [1], "atleta": ["Test"]})
    mock_cargar.return_value = mock_df
    result = obtener_historial_lesiones_servicio(atleta="Test")
    pd.testing.assert_frame_equal(result, mock_df)
    mock_cargar.assert_called_once_with(atleta="Test")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_injury_services.py`
Expected: FAIL (function not defined)

- [ ] **Step 3: Write implementation**

Modify `core/services.py`:
```python
from data.db import cargar_historial_lesiones

def obtener_historial_lesiones_servicio(atleta: str) -> pd.DataFrame:
    return cargar_historial_lesiones(atleta=atleta)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_injury_services.py`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add core/services.py tests/test_injury_services.py
git commit -m "feat: implement obtener_historial_lesiones_servicio"
```

### Task 5: Implement actualizar_estado_lesion_servicio

**Files:**
- Modify: `core/services.py`
- Test: `tests/test_injury_services.py`

- [ ] **Step 1: Write the failing test**

```python
from datetime import date
@patch('core.services.actualizar_estado_lesion')
def test_actualizar_estado_lesion_servicio(mock_actualizar):
    mock_actualizar.return_value = (True, "Updated")
    success, msg = actualizar_estado_lesion_servicio(lesion_id="1", nuevo_estado="Recuperado", fecha_alta=date(2026, 6, 1))
    assert success is True
    assert msg == "Updated"
    mock_actualizar.assert_called_once_with(lesion_id="1", nuevo_estado="Recuperado", fecha_alta=date(2026, 6, 1))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_injury_services.py`
Expected: FAIL (function not defined)

- [ ] **Step 3: Write implementation**

Modify `core/services.py`:
```python
from datetime import date
from data.db import actualizar_estado_lesion

def actualizar_estado_lesion_servicio(lesion_id: str, nuevo_estado: str, fecha_alta: Optional[date] = None) -> Tuple[bool, str]:
    return actualizar_estado_lesion(lesion_id=lesion_id, nuevo_estado=nuevo_estado, fecha_alta=fecha_alta)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_injury_services.py`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add core/services.py tests/test_injury_services.py
git commit -m "feat: implement actualizar_estado_lesion_servicio"
```
