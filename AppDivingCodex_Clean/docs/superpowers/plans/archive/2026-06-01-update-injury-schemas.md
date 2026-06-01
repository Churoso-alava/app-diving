# Update Injury Schemas Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Update `core/schemas.py` and `tests/test_injury_schemas.py` to reflect the new database schema for injury records.

**Architecture:** Add Enums for new data fields, update `InjuryInput` dataclass, and ensure strict validation.

**Tech Stack:** Python, Dataclasses, Enum.

---

### Task 1: Update core/schemas.py

**Files:**
- Modify: `core/schemas.py`

- [ ] **Step 1: Import Enum and define new Enums**

```python
from enum import Enum

class TipoTejido(Enum):
    MUSCULO = "musculo"
    TENDON = "tendon"
    LIGAMENTO = "ligamento"
    OTRO = "otro"

class MecanismoInicio(Enum):
    AGUDA = "aguda"
    SOBREUSO = "sobreuso"

class HistorialRecurrencia(Enum):
    NUEVA = "nueva"
    RECURRENCIA = "recurrencia"
```

- [ ] **Step 2: Modify InjuryInput dataclass**

```python
@dataclass
class InjuryInput:
    atleta: str
    fecha_evento: str
    zona_corporal: str
    tipo_tejido: TipoTejido
    mecanismo: MecanismoInicio
    recurrencia: HistorialRecurrencia
    mecanismo_contacto: bool
    notas: str = ""
    fecha_alta_medica: Optional[str] = None
    fecha_rtt: Optional[str] = None
    fecha_rtp: Optional[str] = None

    def __post_init__(self) -> None:
        errors: list[str] = []
        if not self.atleta or not self.atleta.strip():
            errors.append("atleta no puede estar vacío")
        
        # Validate Enums
        if not isinstance(self.tipo_tejido, TipoTejido):
            errors.append("tipo_tejido inválido")
        if not isinstance(self.mecanismo, MecanismoInicio):
            errors.append("mecanismo inválido")
        if not isinstance(self.recurrencia, HistorialRecurrencia):
            errors.append("recurrencia inválida")
            
        # Validate dates
        for field_name in ["fecha_evento", "fecha_alta_medica", "fecha_rtt", "fecha_rtp"]:
            value = getattr(self, field_name)
            if value is not None:
                try:
                    pd.Timestamp(value)
                except Exception:
                    errors.append(f"{field_name} inválida: '{value}'")
        
        if errors:
            raise ValueError("InjuryInput inválido: " + "; ".join(errors))

    def to_dict(self) -> dict:
        d = {
            "atleta": self.atleta.strip(),
            "fecha_evento": self.fecha_evento,
            "zona_corporal": self.zona_corporal,
            "tipo_tejido": self.tipo_tejido.value,
            "mecanismo": self.mecanismo.value,
            "recurrencia": self.recurrencia.value,
            "mecanismo_contacto": self.mecanismo_contacto,
            "notas": self.notas.strip(),
        }
        for field_name in ["fecha_alta_medica", "fecha_rtt", "fecha_rtp"]:
            value = getattr(self, field_name)
            if value is not None:
                d[field_name] = value
        return d
```

- [ ] **Step 3: Commit changes**

### Task 2: Update tests/test_injury_schemas.py

**Files:**
- Modify: `tests/test_injury_schemas.py`

- [ ] **Step 1: Rewrite tests for new InjuryInput structure**

- [ ] **Step 2: Add validation tests for Enums and dates**

- [ ] **Step 3: Commit changes**
