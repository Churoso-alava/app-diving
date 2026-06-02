# Módulo de Gestión de Lesiones — Plan de Implementación

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implementar el módulo completo de Gestión de Lesiones (Tab 6) en AppDivingCodex v4.5, desde la migración SQL hasta la UI en Streamlit, con integración en el pipeline de fatiga.

**Architecture:** Esquema de datos validado en `core/schemas.py` → CRUD puro en `data/db.py` (Supabase directo, sin RPC) → lógica de servicio en `core/services.py` → componente UI en `components/tab_lesiones.py` → integración en `app.py`. El módulo es aditivo: no rompe firmas existentes; `pipeline_diagnostico` recibe `df_lesiones` como parámetro opcional.

**Tech Stack:** Python 3.12+, Streamlit, Supabase (PostgreSQL + RLS), Plotly, pytest + unittest.mock.

---

## Estructura de Archivos

| Acción | Ruta | Responsabilidad |
|--------|------|-----------------|
| Crear | `tests/test_injury_schemas.py` | Tests unitarios de `InjuryInput` |
| Crear | `tests/test_injury_db.py` | Tests de CRUD (cliente mockeado) |
| Crear | `tests/test_injury_services.py` | Tests de lógica de servicio pura |
| Modificar | `core/schemas.py` | Añadir constantes + `InjuryInput` + `InjuryRecord` |
| Modificar | `data/db.py` | Añadir 4 funciones CRUD de lesiones |
| Modificar | `core/services.py` | Añadir 3 funciones de servicio + param `df_lesiones` en pipeline |
| Crear | `components/tab_lesiones.py` | Componente UI completo (3 sub-tabs) |
| Modificar | `app.py` | Reemplazar placeholder Tab 6 |

---

## Prerrequisito: Migración SQL

**Ejecutar en Supabase SQL Editor antes de cualquier tarea.**

```sql
-- ─── 1. Tabla ────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS lesiones (
    id            UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    atleta        TEXT        NOT NULL,
    fecha_lesion  DATE        NOT NULL,
    zona_corporal TEXT        NOT NULL
        CHECK (zona_corporal IN ('Hombro','Rodilla','Espalda','Tobillo','Muñeca','Cadera','Cuello','Otro')),
    tipo          TEXT        NOT NULL
        CHECK (tipo IN ('Aguda','Sobreuso')),
    gravedad      TEXT        NOT NULL
        CHECK (gravedad IN ('Leve','Moderada','Grave')),
    estado        TEXT        NOT NULL DEFAULT 'Activa'
        CHECK (estado IN ('Activa','Recuperación','Alta')),
    fecha_alta    DATE,
    notas         TEXT        NOT NULL DEFAULT '',
    created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at    TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ─── 2. Índices ───────────────────────────────────────────────────────────────
CREATE INDEX IF NOT EXISTS idx_lesiones_atleta_estado ON lesiones(atleta, estado);
CREATE INDEX IF NOT EXISTS idx_lesiones_atleta        ON lesiones(atleta);

-- ─── 3. Trigger updated_at ───────────────────────────────────────────────────
CREATE OR REPLACE FUNCTION update_lesiones_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_lesiones_updated_at
    BEFORE UPDATE ON lesiones
    FOR EACH ROW EXECUTE FUNCTION update_lesiones_updated_at();

-- ─── 4. RLS ───────────────────────────────────────────────────────────────────
ALTER TABLE lesiones ENABLE ROW LEVEL SECURITY;

-- Staff: acceso total
CREATE POLICY "staff_all_lesiones"
    ON lesiones FOR ALL
    USING    (is_staff_or_admin())
    WITH CHECK (is_staff_or_admin());

-- Deportista: sólo lectura de sus propias lesiones
CREATE POLICY "deportista_own_lesiones"
    ON lesiones FOR SELECT
    USING (atleta = (
        SELECT nombre FROM perfiles
        WHERE usuario_acceso = current_setting('request.jwt.claims', true)::jsonb->>'sub'
        LIMIT 1
    ));
```

Verificar que la tabla existe: `SELECT COUNT(*) FROM lesiones;` debe devolver 0 filas, sin error.

---

## Tarea 1: Esquemas (`core/schemas.py`)

**Files:**
- Modify: `core/schemas.py` (añadir al final del archivo)
- Create: `tests/test_injury_schemas.py`

### Step 1.1 — Escribir tests fallidos

- [ ] Crear `tests/test_injury_schemas.py` con el contenido exacto:

```python
# tests/test_injury_schemas.py
"""Tests unitarios para InjuryInput. Sin dependencias de DB ni Streamlit."""
import pytest
from core.schemas import (
    InjuryInput,
    ZONA_CORPORAL_OPTIONS,
    TIPO_LESION_OPTIONS,
    GRAVEDAD_OPTIONS,
    ESTADO_LESION_OPTIONS,
)


class TestInjuryInputValidacion:
    def test_instancia_valida_defaults(self):
        record = InjuryInput(
            atleta="Carlos",
            fecha_lesion="2026-01-15",
            zona_corporal="Rodilla",
            tipo="Aguda",
            gravedad="Moderada",
        )
        assert record.estado == "Activa"    # default
        assert record.notas == ""           # default
        assert record.fecha_alta is None    # default

    def test_to_dict_no_incluye_fecha_alta_none(self):
        record = InjuryInput(
            atleta="Ana",
            fecha_lesion="2026-01-15",
            zona_corporal="Hombro",
            tipo="Sobreuso",
            gravedad="Leve",
        )
        d = record.to_dict()
        assert "fecha_alta" not in d
        assert d["atleta"] == "Ana"
        assert d["estado"] == "Activa"

    def test_to_dict_incluye_fecha_alta_si_presente(self):
        record = InjuryInput(
            atleta="Luis",
            fecha_lesion="2026-01-01",
            zona_corporal="Tobillo",
            tipo="Aguda",
            gravedad="Leve",
            estado="Alta",
            fecha_alta="2026-01-20",
        )
        d = record.to_dict()
        assert d["fecha_alta"] == "2026-01-20"
        assert d["estado"] == "Alta"

    def test_atleta_vacio_lanza_error(self):
        with pytest.raises(ValueError, match="atleta"):
            InjuryInput(
                atleta="",
                fecha_lesion="2026-01-15",
                zona_corporal="Hombro",
                tipo="Sobreuso",
                gravedad="Leve",
            )

    def test_zona_corporal_invalida_lanza_error(self):
        with pytest.raises(ValueError, match="zona_corporal"):
            InjuryInput(
                atleta="Carlos",
                fecha_lesion="2026-01-15",
                zona_corporal="Codo",  # no está en la lista
                tipo="Aguda",
                gravedad="Moderada",
            )

    def test_tipo_invalido_lanza_error(self):
        with pytest.raises(ValueError, match="tipo"):
            InjuryInput(
                atleta="Ana",
                fecha_lesion="2026-01-15",
                zona_corporal="Rodilla",
                tipo="Crónica",  # no está en la lista
                gravedad="Leve",
            )

    def test_gravedad_invalida_lanza_error(self):
        with pytest.raises(ValueError, match="gravedad"):
            InjuryInput(
                atleta="Luis",
                fecha_lesion="2026-01-15",
                zona_corporal="Espalda",
                tipo="Sobreuso",
                gravedad="Extrema",  # no está en la lista
            )

    def test_estado_invalido_lanza_error(self):
        with pytest.raises(ValueError, match="estado"):
            InjuryInput(
                atleta="Ana",
                fecha_lesion="2026-01-15",
                zona_corporal="Hombro",
                tipo="Aguda",
                gravedad="Grave",
                estado="Pendiente",  # no está en la lista
            )

    def test_fecha_lesion_invalida_lanza_error(self):
        with pytest.raises(ValueError, match="fecha_lesion"):
            InjuryInput(
                atleta="Carlos",
                fecha_lesion="not-a-date",
                zona_corporal="Rodilla",
                tipo="Aguda",
                gravedad="Leve",
            )

    def test_fecha_alta_invalida_lanza_error(self):
        with pytest.raises(ValueError, match="fecha_alta"):
            InjuryInput(
                atleta="Luis",
                fecha_lesion="2026-01-01",
                zona_corporal="Tobillo",
                tipo="Aguda",
                gravedad="Leve",
                fecha_alta="mañana",  # inválida
            )

    def test_multiples_errores_en_un_mensaje(self):
        with pytest.raises(ValueError) as exc_info:
            InjuryInput(
                atleta="",
                fecha_lesion="bad",
                zona_corporal="Codo",
                tipo="Aguda",
                gravedad="Leve",
            )
        msg = str(exc_info.value)
        # Todos los errores están en el mismo mensaje
        assert "atleta" in msg
        assert "zona_corporal" in msg

    def test_constantes_completas(self):
        assert "Hombro" in ZONA_CORPORAL_OPTIONS
        assert "Otro" in ZONA_CORPORAL_OPTIONS
        assert "Aguda" in TIPO_LESION_OPTIONS
        assert "Sobreuso" in TIPO_LESION_OPTIONS
        assert "Grave" in GRAVEDAD_OPTIONS
        assert "Alta" in ESTADO_LESION_OPTIONS
```

### Step 1.2 — Ejecutar para verificar que fallan

- [ ] Correr:

```bash
pytest tests/test_injury_schemas.py -v 2>&1 | head -20
```

Salida esperada: `ImportError: cannot import name 'InjuryInput' from 'core.schemas'`

### Step 1.3 — Implementar en `core/schemas.py`

- [ ] Añadir al **final** de `core/schemas.py` (después de `class DiagnosticResult`):

```python
# ─────────────────────────────────────────────────────────────────────────────
# LESIONES — Constantes y Esquemas
# ─────────────────────────────────────────────────────────────────────────────

ZONA_CORPORAL_OPTIONS: list[str] = [
    "Hombro", "Rodilla", "Espalda", "Tobillo",
    "Muñeca", "Cadera", "Cuello", "Otro",
]
TIPO_LESION_OPTIONS:    list[str] = ["Aguda", "Sobreuso"]
GRAVEDAD_OPTIONS:       list[str] = ["Leve", "Moderada", "Grave"]
ESTADO_LESION_OPTIONS:  list[str] = ["Activa", "Recuperación", "Alta"]


@dataclass
class InjuryInput:
    """
    Valida y normaliza los datos de una lesión antes de insertar en la DB.
    Sigue el mismo patrón que SessionInput: validación en __post_init__,
    serialización en to_dict().
    """
    atleta:        str
    fecha_lesion:  str           # ISO 8601: "YYYY-MM-DD"
    zona_corporal: str
    tipo:          str           # Aguda | Sobreuso
    gravedad:      str           # Leve | Moderada | Grave
    estado:        str = "Activa"
    notas:         str = ""
    fecha_alta:    Optional[str] = None  # ISO 8601, sólo si estado == "Alta"

    def __post_init__(self) -> None:
        errors: list[str] = []
        if not self.atleta or not self.atleta.strip():
            errors.append("atleta no puede estar vacío")
        if self.zona_corporal not in ZONA_CORPORAL_OPTIONS:
            errors.append(
                f"zona_corporal '{self.zona_corporal}' inválida. "
                f"Opciones: {ZONA_CORPORAL_OPTIONS}"
            )
        if self.tipo not in TIPO_LESION_OPTIONS:
            errors.append(
                f"tipo '{self.tipo}' inválido. Opciones: {TIPO_LESION_OPTIONS}"
            )
        if self.gravedad not in GRAVEDAD_OPTIONS:
            errors.append(
                f"gravedad '{self.gravedad}' inválida. Opciones: {GRAVEDAD_OPTIONS}"
            )
        if self.estado not in ESTADO_LESION_OPTIONS:
            errors.append(
                f"estado '{self.estado}' inválido. Opciones: {ESTADO_LESION_OPTIONS}"
            )
        try:
            pd.Timestamp(self.fecha_lesion)
        except Exception:
            errors.append(f"fecha_lesion inválida: '{self.fecha_lesion}'")
        if self.fecha_alta is not None:
            try:
                pd.Timestamp(self.fecha_alta)
            except Exception:
                errors.append(f"fecha_alta inválida: '{self.fecha_alta}'")
        if errors:
            raise ValueError("InjuryInput inválido: " + "; ".join(errors))

    def to_dict(self) -> dict:
        d: dict = {
            "atleta":        self.atleta.strip(),
            "fecha_lesion":  self.fecha_lesion,
            "zona_corporal": self.zona_corporal,
            "tipo":          self.tipo,
            "gravedad":      self.gravedad,
            "estado":        self.estado,
            "notas":         self.notas.strip(),
        }
        if self.fecha_alta is not None:
            d["fecha_alta"] = self.fecha_alta
        return d


class InjuryRecord(TypedDict, total=False):
    """Registro de lesión leído desde la DB (shape del dict de Supabase)."""
    id:            str
    atleta:        str
    fecha_lesion:  str
    zona_corporal: str
    tipo:          str
    gravedad:      str
    estado:        str
    fecha_alta:    Optional[str]
    notas:         str
    created_at:    str
    updated_at:    str
```

### Step 1.4 — Verificar que los tests pasan

- [ ] Correr:

```bash
pytest tests/test_injury_schemas.py -v
```

Salida esperada: `11 passed`

### Step 1.5 — Commit

- [ ] Commit:

```bash
git add core/schemas.py tests/test_injury_schemas.py
git commit -m "feat(schemas): add InjuryInput, InjuryRecord and domain constants"
```

---

## Tarea 2: CRUD de Lesiones (`data/db.py`)

**Files:**
- Modify: `data/db.py` (añadir al final del archivo)
- Create: `tests/test_injury_db.py`

### Step 2.1 — Escribir tests fallidos

- [ ] Crear `tests/test_injury_db.py`:

```python
# tests/test_injury_db.py
"""
Tests de CRUD de lesiones. El cliente Supabase se mockea completamente;
estos tests validan la lógica de validación y transformación, no el contrato de red.
"""
import pytest
from datetime import date
from unittest.mock import MagicMock, patch, call
import pandas as pd


class TestInsertarLesion:
    def test_atleta_vacio_rechazado_antes_de_db(self):
        """La validación de InjuryInput ocurre antes del get_client()."""
        from data.db import insertar_lesion
        ok, msg = insertar_lesion(
            atleta="",
            fecha_lesion=date(2026, 1, 15),
            zona_corporal="Rodilla",
            tipo="Aguda",
            gravedad="Leve",
        )
        assert not ok
        assert "atleta" in msg.lower()

    def test_zona_invalida_rechazada_antes_de_db(self):
        from data.db import insertar_lesion
        ok, msg = insertar_lesion(
            atleta="Carlos",
            fecha_lesion=date(2026, 1, 15),
            zona_corporal="Codo",
            tipo="Aguda",
            gravedad="Leve",
        )
        assert not ok
        assert "zona_corporal" in msg.lower()

    @patch("data.db.get_client")
    def test_insercion_exitosa(self, mock_get_client):
        from data.db import insertar_lesion
        mock_client = MagicMock()
        mock_client.table.return_value.insert.return_value.execute.return_value = MagicMock(data=[{}])
        mock_get_client.return_value = mock_client

        ok, msg = insertar_lesion(
            atleta="Carlos",
            fecha_lesion=date(2026, 1, 15),
            zona_corporal="Hombro",
            tipo="Sobreuso",
            gravedad="Moderada",
        )
        assert ok
        assert "Carlos" in msg
        # Verificar que se insertó en la tabla correcta
        mock_client.table.assert_called_with("lesiones")

    @patch("data.db.get_client")
    def test_sin_conexion_retorna_false(self, mock_get_client):
        from data.db import insertar_lesion
        mock_get_client.return_value = None
        ok, msg = insertar_lesion(
            atleta="Ana",
            fecha_lesion=date(2026, 1, 15),
            zona_corporal="Rodilla",
            tipo="Aguda",
            gravedad="Grave",
        )
        assert not ok
        assert "conexión" in msg.lower()

    @patch("data.db.get_client")
    def test_insercion_con_fecha_alta(self, mock_get_client):
        from data.db import insertar_lesion
        mock_client = MagicMock()
        mock_client.table.return_value.insert.return_value.execute.return_value = MagicMock(data=[{}])
        mock_get_client.return_value = mock_client

        ok, msg = insertar_lesion(
            atleta="Luis",
            fecha_lesion=date(2026, 1, 1),
            zona_corporal="Tobillo",
            tipo="Aguda",
            gravedad="Leve",
            estado="Alta",
            fecha_alta=date(2026, 1, 20),
        )
        assert ok
        # Verificar que el dict insertado incluye fecha_alta
        inserted_dict = mock_client.table.return_value.insert.call_args[0][0]
        assert "fecha_alta" in inserted_dict
        assert inserted_dict["fecha_alta"] == "2026-01-20"


class TestCargarLesionesActivas:
    @patch("data.db.get_client")
    def test_retorna_dataframe_vacio_si_no_hay_datos(self, mock_get_client):
        from data.db import cargar_lesiones_activas
        mock_client = MagicMock()
        # Simular cadena: .table().select().in_().order().execute()
        mock_client.table.return_value.select.return_value \
            .in_.return_value.order.return_value \
            .execute.return_value = MagicMock(data=[])
        mock_get_client.return_value = mock_client

        df = cargar_lesiones_activas()
        assert isinstance(df, pd.DataFrame)
        assert df.empty

    @patch("data.db.get_client")
    def test_retorna_dataframe_con_datos(self, mock_get_client):
        from data.db import cargar_lesiones_activas
        mock_client = MagicMock()
        mock_client.table.return_value.select.return_value \
            .in_.return_value.order.return_value \
            .execute.return_value = MagicMock(data=[
                {
                    "id": "uuid-1", "atleta": "Ana",
                    "fecha_lesion": "2026-01-10", "zona_corporal": "Rodilla",
                    "tipo": "Aguda", "gravedad": "Grave", "estado": "Activa",
                    "notas": "", "fecha_alta": None,
                }
            ])
        mock_get_client.return_value = mock_client

        df = cargar_lesiones_activas()
        assert len(df) == 1
        assert df.iloc[0]["atleta"] == "Ana"

    @patch("data.db.get_client")
    def test_filtra_por_atleta_si_se_pasa(self, mock_get_client):
        from data.db import cargar_lesiones_activas
        mock_client = MagicMock()
        chain = (
            mock_client.table.return_value.select.return_value
            .in_.return_value.order.return_value
        )
        chain.eq.return_value.execute.return_value = MagicMock(data=[])
        mock_get_client.return_value = mock_client

        cargar_lesiones_activas(atleta="Carlos")
        chain.eq.assert_called_once_with("atleta", "Carlos")

    @patch("data.db.get_client")
    def test_sin_conexion_retorna_dataframe_vacio(self, mock_get_client):
        from data.db import cargar_lesiones_activas
        mock_get_client.return_value = None
        df = cargar_lesiones_activas()
        assert isinstance(df, pd.DataFrame)
        assert df.empty


class TestActualizarEstadoLesion:
    def test_estado_invalido_rechazado_sin_db(self):
        from data.db import actualizar_estado_lesion
        ok, msg = actualizar_estado_lesion("uuid-1", "Pendiente")
        assert not ok
        assert "inválido" in msg.lower()

    @patch("data.db.get_client")
    def test_actualizacion_exitosa(self, mock_get_client):
        from data.db import actualizar_estado_lesion
        mock_client = MagicMock()
        mock_client.table.return_value.update.return_value \
            .eq.return_value.execute.return_value = MagicMock(data=[{}])
        mock_get_client.return_value = mock_client

        ok, msg = actualizar_estado_lesion("uuid-1", "Recuperación")
        assert ok
        assert "Recuperación" in msg

    @patch("data.db.get_client")
    def test_alta_sin_fecha_usa_hoy(self, mock_get_client):
        from data.db import actualizar_estado_lesion
        from datetime import date
        mock_client = MagicMock()
        mock_client.table.return_value.update.return_value \
            .eq.return_value.execute.return_value = MagicMock(data=[{}])
        mock_get_client.return_value = mock_client

        ok, _ = actualizar_estado_lesion("uuid-1", "Alta", fecha_alta=None)
        assert ok
        updated_dict = mock_client.table.return_value.update.call_args[0][0]
        assert "fecha_alta" in updated_dict
        assert updated_dict["fecha_alta"] == str(date.today())
```

### Step 2.2 — Ejecutar para verificar que fallan

- [ ] Correr:

```bash
pytest tests/test_injury_db.py -v 2>&1 | head -25
```

Salida esperada: `ImportError` o `AttributeError` porque las funciones no existen aún.

### Step 2.3 — Implementar en `data/db.py`

- [ ] Añadir al **final** de `data/db.py` (después de `insertar_carga_sesion_batch`):

```python
# ─────────────────────────────────────────────────────────────────────────────
# LESIONES — CRUD
# ─────────────────────────────────────────────────────────────────────────────

def insertar_lesion(
    atleta: str,
    fecha_lesion: date,
    zona_corporal: str,
    tipo: str,
    gravedad: str,
    estado: str = "Activa",
    notas: str = "",
    fecha_alta: Optional[date] = None,
) -> Tuple[bool, str]:
    """
    Inserta una nueva lesión en la tabla `lesiones`.
    Valida vía InjuryInput antes de cualquier llamada a la DB.
    """
    from core.schemas import InjuryInput
    try:
        record = InjuryInput(
            atleta=atleta,
            fecha_lesion=str(fecha_lesion),
            zona_corporal=zona_corporal,
            tipo=tipo,
            gravedad=gravedad,
            estado=estado,
            notas=notas,
            fecha_alta=str(fecha_alta) if fecha_alta else None,
        )
    except ValueError as exc:
        return False, str(exc)

    client = get_client()
    if not client:
        return False, "No hay conexión a la base de datos."
    try:
        client.table("lesiones").insert(record.to_dict()).execute()
        return True, f"Lesión de {atleta} ({zona_corporal} — {gravedad}) registrada."
    except Exception as exc:
        log.error("insertar_lesion(%s): %s", atleta, exc)
        return False, str(exc)


def cargar_lesiones_activas(atleta: Optional[str] = None) -> pd.DataFrame:
    """
    Retorna lesiones en estado 'Activa' o 'Recuperación', ordenadas por fecha desc.
    Si `atleta` es None, retorna lesiones activas de todo el equipo.
    """
    client = get_client()
    if not client:
        return pd.DataFrame()
    try:
        query = (
            client.table("lesiones")
            .select("*")
            .in_("estado", ["Activa", "Recuperación"])
            .order("fecha_lesion", desc=True)
        )
        if atleta:
            query = query.eq("atleta", atleta)
        resp = query.execute()
        df = pd.DataFrame(resp.data or [])
        if not df.empty:
            df["fecha_lesion"] = pd.to_datetime(df["fecha_lesion"], errors="coerce")
        return df
    except Exception as exc:
        log.error("cargar_lesiones_activas: %s", exc)
        return pd.DataFrame()


def cargar_historial_lesiones(atleta: str) -> pd.DataFrame:
    """
    Retorna el historial completo de lesiones (todos los estados) de un atleta,
    ordenado por fecha descendente.
    """
    client = get_client()
    if not client:
        return pd.DataFrame()
    try:
        resp = (
            client.table("lesiones")
            .select("*")
            .eq("atleta", atleta)
            .order("fecha_lesion", desc=True)
            .execute()
        )
        df = pd.DataFrame(resp.data or [])
        if not df.empty:
            df["fecha_lesion"] = pd.to_datetime(df["fecha_lesion"], errors="coerce")
            if "fecha_alta" in df.columns:
                df["fecha_alta"] = pd.to_datetime(df["fecha_alta"], errors="coerce")
        return df
    except Exception as exc:
        log.error("cargar_historial_lesiones(%s): %s", atleta, exc)
        return pd.DataFrame()


def actualizar_estado_lesion(
    lesion_id: str,
    nuevo_estado: str,
    fecha_alta: Optional[date] = None,
) -> Tuple[bool, str]:
    """
    Actualiza el estado de una lesión por su UUID.
    Si `nuevo_estado` es 'Alta' y no se pasa `fecha_alta`, se usa la fecha de hoy.
    """
    from core.schemas import ESTADO_LESION_OPTIONS
    if nuevo_estado not in ESTADO_LESION_OPTIONS:
        return False, (
            f"estado '{nuevo_estado}' inválido. "
            f"Opciones: {ESTADO_LESION_OPTIONS}"
        )

    client = get_client()
    if not client:
        return False, "No hay conexión a la base de datos."

    update_data: dict = {"estado": nuevo_estado}
    if nuevo_estado == "Alta":
        update_data["fecha_alta"] = (
            str(fecha_alta) if fecha_alta else str(date.today())
        )

    try:
        client.table("lesiones").update(update_data).eq("id", lesion_id).execute()
        return True, f"Estado actualizado a '{nuevo_estado}'."
    except Exception as exc:
        log.error("actualizar_estado_lesion(%s): %s", lesion_id, exc)
        return False, str(exc)
```

### Step 2.4 — Verificar que los tests pasan

- [ ] Correr:

```bash
pytest tests/test_injury_db.py -v
```

Salida esperada: `14 passed`

### Step 2.5 — Commit

- [ ] Commit:

```bash
git add data/db.py tests/test_injury_db.py
git commit -m "feat(db): add injury CRUD — insertar, cargar_activas, historial, actualizar_estado"
```

---

## Tarea 3: Lógica de Servicio (`core/services.py`)

**Files:**
- Modify: `core/services.py` (añadir al final + modificar firma de `pipeline_diagnostico`)
- Create: `tests/test_injury_services.py`

### Step 3.1 — Escribir tests fallidos

- [ ] Crear `tests/test_injury_services.py`:

```python
# tests/test_injury_services.py
"""
Tests de servicios de lesiones. Toda la lógica es pura (DataFrames como input);
no requiere mocking de DB.
"""
import pytest
import pandas as pd
from core.services import (
    tiene_lesion_activa,
    get_advertencia_lesion,
    resumen_lesiones_equipo,
)


@pytest.fixture
def df_lesiones():
    return pd.DataFrame([
        {
            "atleta": "Ana",    "zona_corporal": "Rodilla",
            "gravedad": "Grave",    "estado": "Activa",
        },
        {
            "atleta": "Carlos", "zona_corporal": "Hombro",
            "gravedad": "Moderada", "estado": "Recuperación",
        },
        {
            "atleta": "Luis",   "zona_corporal": "Tobillo",
            "gravedad": "Leve",     "estado": "Alta",
        },
    ])


class TestTieneLesionActiva:
    def test_true_para_estado_activa(self, df_lesiones):
        assert tiene_lesion_activa("Ana", df_lesiones) is True

    def test_true_para_estado_recuperacion(self, df_lesiones):
        assert tiene_lesion_activa("Carlos", df_lesiones) is True

    def test_false_para_estado_alta(self, df_lesiones):
        assert tiene_lesion_activa("Luis", df_lesiones) is False

    def test_false_para_atleta_sin_lesiones(self, df_lesiones):
        assert tiene_lesion_activa("Pedro", df_lesiones) is False

    def test_false_con_dataframe_vacio(self):
        assert tiene_lesion_activa("Ana", pd.DataFrame()) is False


class TestGetAdvertenciaLesion:
    def test_retorna_string_con_zona_y_gravedad(self, df_lesiones):
        adv = get_advertencia_lesion("Ana", df_lesiones)
        assert adv is not None
        assert "Rodilla" in adv
        assert "Grave" in adv

    def test_retorna_none_para_estado_alta(self, df_lesiones):
        adv = get_advertencia_lesion("Luis", df_lesiones)
        assert adv is None

    def test_retorna_none_sin_lesiones(self, df_lesiones):
        adv = get_advertencia_lesion("Desconocido", df_lesiones)
        assert adv is None

    def test_retorna_none_con_dataframe_vacio(self):
        adv = get_advertencia_lesion("Ana", pd.DataFrame())
        assert adv is None

    def test_formato_incluye_emoji(self, df_lesiones):
        adv = get_advertencia_lesion("Carlos", df_lesiones)
        assert adv is not None
        assert "🩹" in adv


class TestResumenLesionesEquipo:
    def test_total_activas_excluye_alta(self, df_lesiones):
        resumen = resumen_lesiones_equipo(df_lesiones)
        assert resumen["total_activas"] == 2  # Ana (Activa) + Carlos (Recuperación)

    def test_atletas_lesionados_excluye_alta(self, df_lesiones):
        resumen = resumen_lesiones_equipo(df_lesiones)
        assert "Ana" in resumen["atletas_lesionados"]
        assert "Carlos" in resumen["atletas_lesionados"]
        assert "Luis" not in resumen["atletas_lesionados"]

    def test_por_zona_cuenta_activas(self, df_lesiones):
        resumen = resumen_lesiones_equipo(df_lesiones)
        assert "Rodilla" in resumen["por_zona"]
        assert "Hombro" in resumen["por_zona"]
        assert "Tobillo" not in resumen["por_zona"]  # Luis está en Alta

    def test_dataframe_vacio_retorna_estructura_correcta(self):
        resumen = resumen_lesiones_equipo(pd.DataFrame())
        assert resumen["total_activas"] == 0
        assert resumen["por_zona"] == {}
        assert resumen["por_gravedad"] == {}
        assert resumen["atletas_lesionados"] == []


class TestPipelineDiagnosticoConLesiones:
    """
    Verifica que pipeline_diagnostico acepta df_lesiones sin romper
    el contrato existente (parámetro opcional, default None).
    """
    def test_firma_acepta_df_lesiones_none(self):
        import inspect
        from core.services import pipeline_diagnostico
        sig = inspect.signature(pipeline_diagnostico)
        assert "df_lesiones" in sig.parameters
        assert sig.parameters["df_lesiones"].default is None

    def test_firma_backward_compatible_sin_df_lesiones(self):
        """pipeline_diagnostico debe poder llamarse sin df_lesiones."""
        import inspect
        from core.services import pipeline_diagnostico
        sig = inspect.signature(pipeline_diagnostico)
        # Todos los parámetros antes de df_lesiones tienen default o son los originales
        params = list(sig.parameters.values())
        # atleta, df_raw, simulador son posicionales; el resto tienen defaults
        params_with_defaults = [p for p in params if p.default is not inspect.Parameter.empty]
        assert any(p.name == "df_lesiones" for p in params_with_defaults)
```

### Step 3.2 — Ejecutar para verificar que fallan

- [ ] Correr:

```bash
pytest tests/test_injury_services.py -v 2>&1 | head -20
```

Salida esperada: `ImportError: cannot import name 'tiene_lesion_activa'`

### Step 3.3 — Añadir funciones de servicio al final de `core/services.py`

- [ ] Añadir al **final** de `core/services.py` (después de `calcular_historial_fatiga`):

```python
# ─────────────────────────────────────────────────────────────────────────────
# LESIONES — SERVICIOS
# ─────────────────────────────────────────────────────────────────────────────

def tiene_lesion_activa(atleta: str, df_lesiones: pd.DataFrame) -> bool:
    """
    True si el atleta tiene al menos una lesión en estado 'Activa' o 'Recuperación'.
    Diseñado para ser llamado con el resultado de db.cargar_lesiones_activas().
    """
    if df_lesiones.empty or "estado" not in df_lesiones.columns:
        return False
    mask = (
        (df_lesiones["atleta"] == atleta)
        & df_lesiones["estado"].isin(["Activa", "Recuperación"])
    )
    return bool(mask.any())


def get_advertencia_lesion(atleta: str, df_lesiones: pd.DataFrame) -> str | None:
    """
    Retorna advertencia formateada si el atleta tiene una lesión activa, o None.
    Diseñado para inyectarse en el pipeline_diagnostico como advertencia adicional.

    Formato: "🩹 Lesión ACTIVA — Rodilla (Grave)"
    """
    if df_lesiones.empty or "estado" not in df_lesiones.columns:
        return None
    activas = df_lesiones[
        (df_lesiones["atleta"] == atleta)
        & df_lesiones["estado"].isin(["Activa", "Recuperación"])
    ]
    if activas.empty:
        return None
    row = activas.iloc[0]
    return (
        f"🩹 Lesión {str(row['estado']).upper()} — "
        f"{row['zona_corporal']} ({row['gravedad']})"
    )


def resumen_lesiones_equipo(df_lesiones: pd.DataFrame) -> dict:
    """
    Estadísticas agregadas del equipo para el KPI banner de tab_lesiones.

    Retorna:
        {
            "total_activas":      int,
            "por_zona":           dict[str, int],
            "por_gravedad":       dict[str, int],
            "atletas_lesionados": list[str],
        }
    """
    if df_lesiones.empty or "estado" not in df_lesiones.columns:
        return {
            "total_activas": 0,
            "por_zona": {},
            "por_gravedad": {},
            "atletas_lesionados": [],
        }
    activas = df_lesiones[df_lesiones["estado"].isin(["Activa", "Recuperación"])]
    return {
        "total_activas":      len(activas),
        "por_zona":           activas["zona_corporal"].value_counts().to_dict(),
        "por_gravedad":       activas["gravedad"].value_counts().to_dict(),
        "atletas_lesionados": sorted(activas["atleta"].unique().tolist()),
    }
```

### Step 3.4 — Modificar la firma de `pipeline_diagnostico`

- [ ] Localizar la definición de `pipeline_diagnostico` en `core/services.py` y **reemplazar únicamente la firma y el docstring** (no el cuerpo):

```python
# ANTES:
def pipeline_diagnostico(
    atleta: str,
    df_raw: pd.DataFrame,
    simulador,
    ventana_meso: int = 28,
    perfil: dict | None = None,
    wellness_respuestas: dict | None = None,
    clavados_planificados: list[dict] | None = None,
) -> DiagnosticResult:
```

```python
# DESPUÉS:
def pipeline_diagnostico(
    atleta: str,
    df_raw: pd.DataFrame,
    simulador,
    ventana_meso: int = 28,
    perfil: dict | None = None,
    wellness_respuestas: dict | None = None,
    clavados_planificados: list[dict] | None = None,
    df_lesiones: pd.DataFrame | None = None,   # ← nuevo, backward-compatible
) -> DiagnosticResult:
```

- [ ] Localizar el bloque de advertencias clínicas en `pipeline_diagnostico` (la línea `advertencias: list[str] = []`) y añadir la inyección de advertencia de lesión **como primer check**:

```python
    # ── Advertencias clínicas ─────────────────────────────────────────────────
    advertencias: list[str] = []

    # Lesión activa — máxima prioridad clínica
    if df_lesiones is not None:
        adv_lesion = get_advertencia_lesion(atleta, df_lesiones)
        if adv_lesion:
            advertencias.append(adv_lesion)

    # Estrés pediátrico
    if metricas["edad_atleta"] < 15 and metricas["delta_pct"] > 20:
        advertencias.append("🚨 Estrés pediátrico — revisar carga semanal")
    # ... resto del bloque sin cambios ...
```

### Step 3.5 — Verificar que los tests pasan

- [ ] Correr:

```bash
pytest tests/test_injury_services.py -v
```

Salida esperada: `14 passed`

- [ ] Verificar que los tests existentes del pipeline no se rompen:

```bash
pytest tests/ -v --ignore=tests/test_injury_schemas.py --ignore=tests/test_injury_db.py --ignore=tests/test_injury_services.py 2>&1 | tail -5
```

Salida esperada: todos los tests previos siguen pasando.

### Step 3.6 — Commit

- [ ] Commit:

```bash
git add core/services.py tests/test_injury_services.py
git commit -m "feat(services): add injury service functions + optional df_lesiones in pipeline"
```

---

## Tarea 4: Componente UI (`components/tab_lesiones.py`)

**Files:**
- Create: `components/tab_lesiones.py`

> La UI de Streamlit no se testea con pytest directamente. Los tests de Tarea 3 ya validan las funciones de servicio que alimentan la UI. En esta tarea se crea el componente y se verifica manualmente.

### Step 4.1 — Crear `components/tab_lesiones.py`

- [ ] Crear el archivo con el contenido exacto:

```python
"""
components/tab_lesiones.py — Pestaña de Gestión de Lesiones v1.0
Punto de entrada: render_tab_lesiones().

Estructura interna:
  _render_kpi_banner()       — métricas rápidas del equipo
  _render_form_registro()    — formulario para registrar nueva lesión
  _render_seguimiento_activo() — lista expandible de lesiones activas con edición de estado
  _render_historial()        — historial completo por atleta + timeline Plotly
"""
from __future__ import annotations

from datetime import date
from typing import Optional

import pandas as pd
import plotly.express as px
import streamlit as st

from core.schemas import (
    ZONA_CORPORAL_OPTIONS,
    TIPO_LESION_OPTIONS,
    GRAVEDAD_OPTIONS,
    ESTADO_LESION_OPTIONS,
)
from core.services import resumen_lesiones_equipo
from data.db import (
    insertar_lesion,
    cargar_historial_lesiones,
    actualizar_estado_lesion,
)

_GRAVEDAD_COLOR: dict[str, str] = {
    "Leve":     "#16a34a",
    "Moderada": "#ca8a04",
    "Grave":    "#dc2626",
}


def render_tab_lesiones(atletas: list[str], df_lesiones: pd.DataFrame) -> None:
    """
    Punto de entrada principal para Tab 6.

    Parámetros
    ----------
    atletas     : Lista de nombres de atletas activos (de db.cargar_atletas()).
    df_lesiones : DataFrame de lesiones activas del equipo
                  (de db.cargar_lesiones_activas(), ya filtrado a estados Activa/Recuperación).
    """
    st.header("🩹 Gestión de Lesiones")

    resumen = resumen_lesiones_equipo(df_lesiones)
    _render_kpi_banner(resumen)
    st.divider()

    tab_reg, tab_seg, tab_hist = st.tabs(
        ["➕ Registrar", "📋 Seguimiento Activo", "📈 Historial"]
    )
    with tab_reg:
        _render_form_registro(atletas)
    with tab_seg:
        _render_seguimiento_activo(df_lesiones)
    with tab_hist:
        _render_historial(atletas)


def _render_kpi_banner(resumen: dict) -> None:
    col1, col2, col3 = st.columns(3)
    col1.metric("⚡ Lesiones Activas", resumen["total_activas"])
    col2.metric(
        "📍 Zona más afectada",
        max(resumen["por_zona"], key=resumen["por_zona"].get)
        if resumen["por_zona"] else "—",
    )
    col3.metric("👤 Atletas afectados", len(resumen["atletas_lesionados"]))


def _render_form_registro(atletas: list[str]) -> None:
    st.subheader("Registrar Nueva Lesión")
    with st.form("form_lesion_nueva", clear_on_submit=True):
        col1, col2 = st.columns(2)
        with col1:
            atleta_sel = st.selectbox("Atleta", atletas, key="reg_atleta")
            fecha_les  = st.date_input("Fecha de Lesión", value=date.today(), key="reg_fecha")
            zona       = st.selectbox("Zona Corporal", ZONA_CORPORAL_OPTIONS, key="reg_zona")
        with col2:
            tipo       = st.selectbox("Tipo", TIPO_LESION_OPTIONS, key="reg_tipo")
            gravedad   = st.selectbox("Gravedad", GRAVEDAD_OPTIONS, key="reg_gravedad")
            estado_ini = st.selectbox("Estado Inicial", ESTADO_LESION_OPTIONS, index=0, key="reg_estado")
        notas = st.text_area(
            "Notas clínicas",
            placeholder="Mecanismo de lesión, contexto del entrenamiento, observaciones...",
            key="reg_notas",
        )
        submitted = st.form_submit_button("💾 Registrar Lesión", type="primary")

    if submitted:
        ok, msg = insertar_lesion(
            atleta=atleta_sel,
            fecha_lesion=fecha_les,
            zona_corporal=zona,
            tipo=tipo,
            gravedad=gravedad,
            estado=estado_ini,
            notas=notas,
        )
        if ok:
            st.success(msg)
            st.rerun()
        else:
            st.error(msg)


def _render_seguimiento_activo(df_lesiones: pd.DataFrame) -> None:
    st.subheader("Lesiones Activas y en Recuperación")
    if df_lesiones.empty:
        st.info("✅ No hay lesiones activas en el equipo.")
        return

    for idx, row in df_lesiones.iterrows():
        gravedad_tag = row["gravedad"]
        color        = _GRAVEDAD_COLOR.get(gravedad_tag, "#6b7280")
        fecha_str    = str(row["fecha_lesion"])[:10]

        with st.expander(
            f"**{row['atleta']}** — {row['zona_corporal']} ({gravedad_tag}) | "
            f"Estado: {row['estado']} | Desde: {fecha_str}",
            expanded=(gravedad_tag == "Grave"),
        ):
            col_info, col_accion = st.columns([2, 1])
            with col_info:
                st.markdown(
                    f"**Tipo:** {row['tipo']}  \n"
                    f"**Fecha Lesión:** {fecha_str}"
                )
                if row.get("notas"):
                    st.markdown(f"**Notas:** {row['notas']}")
            with col_accion:
                nuevo_estado = st.selectbox(
                    "Actualizar estado",
                    ESTADO_LESION_OPTIONS,
                    index=ESTADO_LESION_OPTIONS.index(row["estado"]),
                    key=f"sel_estado_{row['id']}",
                )
                fecha_alta_input: Optional[date] = None
                if nuevo_estado == "Alta":
                    fecha_alta_input = st.date_input(
                        "Fecha de Alta",
                        value=date.today(),
                        key=f"fecha_alta_{row['id']}",
                    )
                if st.button("Guardar", key=f"btn_guardar_{row['id']}"):
                    ok, msg = actualizar_estado_lesion(
                        lesion_id=str(row["id"]),
                        nuevo_estado=nuevo_estado,
                        fecha_alta=fecha_alta_input,
                    )
                    if ok:
                        st.success(msg)
                        st.rerun()
                    else:
                        st.error(msg)


def _render_historial(atletas: list[str]) -> None:
    st.subheader("Historial por Atleta")
    atleta_sel = st.selectbox("Seleccionar atleta", atletas, key="hist_atleta_sel")
    df_hist = cargar_historial_lesiones(atleta_sel)

    if df_hist.empty:
        st.info(f"Sin lesiones registradas para **{atleta_sel}**.")
        return

    # ── Tabla resumen ──────────────────────────────────────────────────────────
    cols_display = [c for c in
                    ["fecha_lesion", "zona_corporal", "tipo", "gravedad", "estado", "fecha_alta", "notas"]
                    if c in df_hist.columns]
    st.dataframe(
        df_hist[cols_display].rename(columns={
            "fecha_lesion":  "Fecha Lesión",
            "zona_corporal": "Zona",
            "tipo":          "Tipo",
            "gravedad":      "Gravedad",
            "estado":        "Estado",
            "fecha_alta":    "Fecha Alta",
            "notas":         "Notas",
        }),
        use_container_width=True,
        hide_index=True,
    )

    # ── Línea de tiempo Plotly (sólo si hay ≥ 1 lesión con fecha_alta) ──────
    if "fecha_alta" not in df_hist.columns:
        return

    df_timeline = df_hist.copy()
    df_timeline["Inicio"] = pd.to_datetime(df_timeline["fecha_lesion"], errors="coerce")
    df_timeline["Fin"]    = pd.to_datetime(
        df_timeline["fecha_alta"].fillna(str(date.today())), errors="coerce"
    )
    df_timeline = df_timeline.dropna(subset=["Inicio"])

    if df_timeline.empty:
        return

    fig = px.timeline(
        df_timeline,
        x_start="Inicio",
        x_end="Fin",
        y="zona_corporal",
        color="gravedad",
        color_discrete_map=_GRAVEDAD_COLOR,
        hover_data=["tipo", "estado", "notas"],
        title=f"Línea de Tiempo de Lesiones — {atleta_sel}",
        labels={"zona_corporal": "Zona Corporal"},
    )
    fig.update_layout(showlegend=True, height=300)
    st.plotly_chart(fig, use_container_width=True)
```

### Step 4.2 — Verificar importaciones sin levantar Streamlit

- [ ] Correr desde el directorio raíz del proyecto:

```bash
python -c "
import sys; sys.path.insert(0, '.')
# Mock streamlit antes de importar el componente
import unittest.mock as m
sys.modules['streamlit'] = m.MagicMock()
sys.modules['plotly'] = m.MagicMock()
sys.modules['plotly.express'] = m.MagicMock()
from components.tab_lesiones import render_tab_lesiones, resumen_lesiones_equipo
print('OK — importaciones sin error')
"
```

Salida esperada: `OK — importaciones sin error`

### Step 4.3 — Commit

- [ ] Commit:

```bash
git add components/tab_lesiones.py
git commit -m "feat(ui): add tab_lesiones component with register, follow-up and history views"
```

---

## Tarea 5: Integración en `app.py` y Tests de Integración

**Files:**
- Modify: `app.py` (reemplazar Tab 6 placeholder)
- Create: `tests/test_injury_integration.py`

### Step 5.1 — Escribir tests de integración fallidos

- [ ] Crear `tests/test_injury_integration.py`:

```python
# tests/test_injury_integration.py
"""
Tests de integración end-to-end (sin red): verifica que las capas se conectan
correctamente sin llamadas reales a Supabase.
"""
import pytest
import pandas as pd
from datetime import date
from unittest.mock import MagicMock, patch


class TestPipelineDiagnosticoConLesionActiva:
    """
    Verifica que pipeline_diagnostico eleva la advertencia de lesión
    como primera advertencia cuando df_lesiones contiene una lesión activa.
    """
    @patch("core.services.calcular_metricas")
    def test_advertencia_lesion_aparece_primera(self, mock_calcular):
        from core.services import pipeline_diagnostico

        # Simular métricas válidas (sin estado INSUFICIENTE)
        mock_calcular.return_value = {
            "atleta": "Ana", "edad_atleta": 20, "n_sesiones": 10,
            "ultima_fecha": "2026-01-15", "dias_sin_datos": 0,
            "vmp_hoy": 1.2, "mma7": 1.1, "mmc28": 1.15,
            "acwr": 0.95, "delta_pct": 4.3, "z_meso": -0.2,
            "beta_aguda": 0.01, "beta_28": 0.005,
            "dqi": 0.9, "calidad_dato": "alta",
            "swc_personal": 0.05, "sd_personal": 0.05,
            "caida_absoluta": 0.05, "es_ruido_biologico": False,
            "n_sesiones_desc": 0, "historial": [], "fechas": [],
            "cv_pct": 4.0, "p_normalidad": 0.5,
            "wellness_norm": 0.7, "carga_integrada_plan": 50.0,
            "clavados_planificados": None,
        }

        # Mock del simulador fuzzy
        mock_simulador = MagicMock()
        mock_simulador.output = {"fatiga": 80.0}

        df_lesiones = pd.DataFrame([{
            "atleta": "Ana", "zona_corporal": "Rodilla",
            "gravedad": "Grave", "estado": "Activa",
        }])

        resultado = pipeline_diagnostico(
            atleta="Ana",
            df_raw=pd.DataFrame(),
            simulador=mock_simulador,
            df_lesiones=df_lesiones,
        )

        assert resultado["advertencias"][0].startswith("🩹")
        assert "Rodilla" in resultado["advertencias"][0]

    @patch("core.services.calcular_metricas")
    def test_sin_lesiones_no_hay_advertencia_lesion(self, mock_calcular):
        from core.services import pipeline_diagnostico

        mock_calcular.return_value = {
            "atleta": "Carlos", "edad_atleta": 22, "n_sesiones": 10,
            "ultima_fecha": "2026-01-15", "dias_sin_datos": 0,
            "vmp_hoy": 1.3, "mma7": 1.2, "mmc28": 1.25,
            "acwr": 0.96, "delta_pct": 3.8, "z_meso": 0.1,
            "beta_aguda": 0.005, "beta_28": 0.003,
            "dqi": 0.95, "calidad_dato": "alta",
            "swc_personal": 0.04, "sd_personal": 0.04,
            "caida_absoluta": 0.04, "es_ruido_biologico": False,
            "n_sesiones_desc": 0, "historial": [], "fechas": [],
            "cv_pct": 3.5, "p_normalidad": 0.6,
            "wellness_norm": 0.8, "carga_integrada_plan": 60.0,
            "clavados_planificados": None,
        }
        mock_simulador = MagicMock()
        mock_simulador.output = {"fatiga": 82.0}

        resultado = pipeline_diagnostico(
            atleta="Carlos",
            df_raw=pd.DataFrame(),
            simulador=mock_simulador,
            df_lesiones=pd.DataFrame(),
        )
        lesion_advs = [a for a in resultado["advertencias"] if "🩹" in a]
        assert len(lesion_advs) == 0

    @patch("core.services.calcular_metricas")
    def test_backward_compatible_sin_df_lesiones(self, mock_calcular):
        """Llamar sin df_lesiones no debe lanzar excepción."""
        from core.services import pipeline_diagnostico

        mock_calcular.return_value = {
            "atleta": "Luis", "edad_atleta": 18, "n_sesiones": 10,
            "ultima_fecha": "2026-01-15", "dias_sin_datos": 0,
            "vmp_hoy": 1.1, "mma7": 1.0, "mmc28": 1.05,
            "acwr": 0.95, "delta_pct": 4.8, "z_meso": -0.1,
            "beta_aguda": 0.0, "beta_28": 0.0,
            "dqi": 0.85, "calidad_dato": "alta",
            "swc_personal": 0.05, "sd_personal": 0.05,
            "caida_absoluta": 0.05, "es_ruido_biologico": False,
            "n_sesiones_desc": 0, "historial": [], "fechas": [],
            "cv_pct": 4.5, "p_normalidad": 0.45,
            "wellness_norm": 0.6, "carga_integrada_plan": 40.0,
            "clavados_planificados": None,
        }
        mock_simulador = MagicMock()
        mock_simulador.output = {"fatiga": 78.0}

        # Sin df_lesiones → no debe fallar
        resultado = pipeline_diagnostico(
            atleta="Luis",
            df_raw=pd.DataFrame(),
            simulador=mock_simulador,
            # df_lesiones no se pasa → usa default None
        )
        assert "indice_fatiga" in resultado
```

### Step 5.2 — Ejecutar para verificar que fallan

- [ ] Correr:

```bash
pytest tests/test_injury_integration.py -v 2>&1 | head -20
```

Salida esperada: tests fallan porque `pipeline_diagnostico` aún no tiene el parámetro `df_lesiones` (si Tarea 3 no está completa) o porque el mock de `calcular_metricas` necesita ajuste.

### Step 5.3 — Modificar `app.py` para conectar Tab 6

- [ ] Localizar el bloque `with tab6:` en `app.py` (contiene el placeholder) y reemplazarlo:

```python
with tab6:
    from components.tab_lesiones import render_tab_lesiones
    from data.db import cargar_lesiones_activas

    _df_lesiones = cargar_lesiones_activas()   # Todo el equipo, estados Activa/Recuperación
    render_tab_lesiones(atletas=atletas, df_lesiones=_df_lesiones)
```

> **Nota:** `atletas` ya debe estar en scope (cargado en el arranque de `app.py` vía `db.cargar_atletas()`). Si el pipeline de la tab de diagnóstico necesita `df_lesiones`, pasarlo en la llamada a `pipeline_diagnostico` en ese contexto.

### Step 5.4 — Ejecutar suite completa de tests

- [ ] Correr:

```bash
pytest tests/test_injury_schemas.py tests/test_injury_db.py \
       tests/test_injury_services.py tests/test_injury_integration.py \
       -v --tb=short 2>&1 | tail -15
```

Salida esperada:
```
========================= N passed in X.XXs =========================
```
con todos los tests en verde. El número total debe ser ≥ 42 (11 + 14 + 14 + 3).

### Step 5.5 — Verificar ausencia de regresiones

- [ ] Correr suite completa del proyecto:

```bash
pytest tests/ -v --tb=short 2>&1 | tail -5
```

Salida esperada: cero failures (sólo los tests nuevos y los existentes).

### Step 5.6 — Commit final

- [ ] Commit:

```bash
git add app.py tests/test_injury_integration.py
git commit -m "feat(app): wire tab_lesiones into Tab 6 + integration tests"
```

---

## Resumen de Comandos de Verificación

| Fase | Comando | Expected |
|------|---------|----------|
| SQL | `SELECT COUNT(*) FROM lesiones;` en Supabase | `0` sin error |
| Task 1 | `pytest tests/test_injury_schemas.py -v` | 11 passed |
| Task 2 | `pytest tests/test_injury_db.py -v` | 14 passed |
| Task 3 | `pytest tests/test_injury_services.py -v` | 14 passed |
| Task 4 | `python -c "...import check..."` | `OK` |
| Task 5 | `pytest tests/test_injury_integration.py -v` | 3 passed |
| Regresión | `pytest tests/ -v` | 0 failures |

---

## Self-Review

**Spec Coverage:**
- ✅ Modelo de datos `Injury` con todos los campos requeridos (atleta, fecha, zona, tipo, gravedad, estado, notas)
- ✅ Registro de nuevas lesiones por Staff → `_render_form_registro()`
- ✅ Seguimiento y actualización de estado → `_render_seguimiento_activo()` + `actualizar_estado_lesion()`
- ✅ Visualización histórica por atleta → `_render_historial()` + timeline Plotly
- ✅ Integración con análisis de fatiga → `get_advertencia_lesion()` + parámetro `df_lesiones` en `pipeline_diagnostico`
- ✅ Migración SQL con CHECK constraints, RLS y trigger `updated_at`
- ✅ TDD en todas las tareas con código de test exacto
- ✅ Firmas CRUD exactas en `data/db.py`, esquemas exactos en `core/schemas.py`
- ✅ Comandos `pytest` específicos por fase

**Placeholder Scan:** ningún "TBD", "TODO" o código incompleto.

**Type Consistency:**
- `insertar_lesion()` usa `InjuryInput.to_dict()` → las claves del dict coinciden con las columnas SQL
- `cargar_lesiones_activas()` retorna `pd.DataFrame` → consumido por `resumen_lesiones_equipo()` y `_render_seguimiento_activo()` con el mismo shape
- `actualizar_estado_lesion(lesion_id, nuevo_estado, fecha_alta)` — `lesion_id` es `str` (UUID de la DB), consistente con `str(row["id"])` en la UI
- `df_lesiones` en `pipeline_diagnostico` y en las funciones de servicio es siempre `pd.DataFrame` (vacío = sin lesiones, nunca `None` en las funciones puras)
