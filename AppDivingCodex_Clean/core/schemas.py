"""
core/schemas.py — Definiciones de tipos y contratos de datos.
Asegura la consistencia entre Core, Data y UI.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import TypedDict, List, Optional, Any
import pandas as pd

# Rango fisiológico VMP (m/s)
VMP_MIN: float = 0.100
VMP_MAX: float = 2.500

@dataclass
class SessionInput:
    """Valida una sesión antes de insertar en la base de datos."""
    nombre: str
    fecha:  str      # ISO 8601: "YYYY-MM-DD"
    vmp:    float
    notas:  str = field(default="")

    def __post_init__(self) -> None:
        errors: list[str] = []
        if not self.nombre or not self.nombre.strip():
            errors.append("nombre no puede estar vacío")
        if not (VMP_MIN <= self.vmp <= VMP_MAX):
            errors.append(
                f"VMP {self.vmp:.3f} fuera de rango fisiológico "
                f"[{VMP_MIN}, {VMP_MAX}] m/s"
            )
        try:
            pd.Timestamp(self.fecha)
        except Exception:
            errors.append(f"fecha inválida: '{self.fecha}'")
        if errors:
            raise ValueError("SessionInput inválido: " + "; ".join(errors))

    def to_dict(self) -> dict:
        return {
            "nombre": self.nombre.strip(),
            "fecha":  self.fecha,
            "vmp":    self.vmp,
            "notas":  self.notas.strip(),
        }

class AthleteMetrics(TypedDict, total=False):
    """Métricas temporales calculadas para un atleta."""
    atleta: str
    edad_atleta: int
    n_sesiones: int
    ultima_fecha: Optional[str]
    dias_sin_datos: Optional[int]
    vmp_hoy: Optional[float]  # Velocidad media propulsiva actual (m/s)
    mma7: Optional[float]
    mmc28: Optional[float]
    acwr: Optional[float]
    delta_pct: Optional[float]
    z_meso: Optional[float]
    beta_aguda: Optional[float]
    beta_28: Optional[float]
    dqi: float
    calidad_dato: str
    swc_personal: Optional[float]
    sd_personal: Optional[float]
    caida_absoluta: Optional[float]
    es_ruido_biologico: bool
    n_sesiones_desc: int
    historial: List[float]
    fechas: List[str]
    cv_pct: Optional[float]
    p_normalidad: Optional[float]
    wellness_norm: Optional[float]
    carga_integrada_plan: Optional[float]
    clavados_planificados: Optional[List[dict]]
    # Campos auxiliares para estados de error/insuficiencia
    estado: Optional[str]
    indice_fatiga: Optional[float]
    color: Optional[str]
    accion: Optional[str]
    advertencias: List[str]
    contexto_cientifico: str
    nota_swc: str

class DiagnosticResult(AthleteMetrics):
    """Resultado completo tras pasar por el motor difuso."""
    indice_fatiga: float
    estado: str
    color: str
    accion: str
    accion_primaria: str
    advertencias: List[str]
    contexto_cientifico: str
    nota_swc: str


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
