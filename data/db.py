"""
data/db.py — NMF-Optimizer v4.4
Capa CRUD pura. Sin lógica de cálculo ni transformación de DataFrames.
Clark-Wilson: CDIs modificados solo vía funciones SECURITY DEFINER en PostgreSQL.
SQL-First: lógica de negocio vive en logic/services.py, no aquí.
"""
from __future__ import annotations

import logging
import os
from datetime import date
import pandas as pd

log = logging.getLogger(__name__)

# ── Límite anti-DoS para importaciones masivas (V-DOS) ──────────────────────
MAX_IMPORT_ROWS: int = 500

# ── Conjuntos válidos para carga grupal ─────────────────────────────────────
_VALID_PLATAFORMAS: frozenset[str] = frozenset({"trampolín", "plataforma"})
_VALID_CAIDAS:      frozenset[str] = frozenset({"pie", "mano"})

# ── Columnas canónicas que devuelve sesiones_vmp ─────────────────────────────
# nombre | fecha | vmp_hoy | vmp_ref | notas | created_at | id
SESIONES_COLS = ("id", "nombre", "fecha", "vmp_hoy", "vmp_ref", "notas", "created_at")


# ─────────────────────────────────────────────────────────────────────────────
# CLIENTE SUPABASE
# ─────────────────────────────────────────────────────────────────────────────

def _get_client():
    """Retorna cliente Supabase autenticado usando secretos de Streamlit."""
    try:
        import streamlit as st
        from supabase import create_client
        
        # Intentar sacar de st.secrets primero (el archivo secrets.toml)
        try:
            url = st.secrets["SUPABASE_URL"]
            key = st.secrets["SUPABASE_KEY"]
        except:
            # Si falla, intentar de variables de entorno (por si acaso)
            import os
            url = os.environ.get("SUPABASE_URL")
            key = os.environ.get("SUPABASE_KEY")

        if not url or not key:
            st.error("🔑 No se encontraron las credenciales de Supabase en secrets.toml")
            return None
            
        return create_client(url, key)
    except Exception as e:
        log.error("Error de conexión: %s", e)
        return None


# ─────────────────────────────────────────────────────────────────────────────
# LECTURA — SELECT
# ─────────────────────────────────────────────────────────────────────────────

def cargar_atletas() -> list[str]:
    """Retorna lista de nombres de atletas activos (tabla atletas)."""
    try:
        resp = _get_client().table("atletas").select("nombre").eq("activo", True).execute()
        return [r["nombre"] for r in (resp.data or [])]
    except Exception as exc:
        log.error("cargar_atletas: %s", exc)
        return []


def cargar_sesiones_raw() -> pd.DataFrame:
    """
    Retorna DataFrame con TODAS las sesiones VMP ordenadas por fecha ASC.

    Columnas garantizadas (snake_case): nombre, fecha, vmp_hoy, vmp_ref, notas, created_at.
    Devuelve DataFrame vacío si hay error de red.
    """
    try:
        resp = _get_client().table("sesiones_vmp").select("*").order("fecha").execute()
        df = pd.DataFrame(resp.data or [])
        if df.empty:
            return df
        # Coerción de tipos mínima — sin transformaciones de negocio
        df["fecha"]   = pd.to_datetime(df["fecha"], errors="coerce").dt.date
        df["vmp_hoy"] = pd.to_numeric(df["vmp_hoy"], errors="coerce")
        df["vmp_ref"] = pd.to_numeric(df.get("vmp_ref"), errors="coerce")
        return df
    except Exception as exc:
        log.error("cargar_sesiones_raw: %s", exc)
        return pd.DataFrame()


def cargar_wellness_atleta(nombre: str) -> pd.DataFrame:
    """Retorna registros Hooper de un atleta, ordenados por fecha DESC."""
    try:
        resp = (
            _get_client()
            .table("wellness")
            .select("*")
            .eq("nombre", nombre)
            .order("fecha", desc=True)
            .execute()
        )
        df = pd.DataFrame(resp.data or [])
        if not df.empty:
            df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce").dt.date
        return df
    except Exception as exc:
        log.error("cargar_wellness_atleta(%s): %s", nombre, exc)
        return pd.DataFrame()


# ─────────────────────────────────────────────────────────────────────────────
# ESCRITURA — INSERT vía Transformation Procedures (Clark-Wilson)
# ─────────────────────────────────────────────────────────────────────────────

def insertar_sesion(
    nombre: str,
    fecha: date,
    vmp_hoy: float,
    vmp_ref: float | None = None,
    notas: str = "",
) -> tuple[bool, str]:
    """Inserta sesión VMP vía RPC insertar_sesion_vmp (SECURITY DEFINER)."""
    if not (0.1 <= vmp_hoy <= 2.5):
        return False, f"VMP fuera de rango fisiológico [0.1, 2.5]: {vmp_hoy}"
    if not nombre or not nombre.strip():
        return False, "nombre no puede estar vacío"
    try:
        _get_client().rpc("insertar_sesion_vmp", {
            "p_nombre": nombre.strip(),
            "p_fecha":  str(fecha),
            "p_vmp_hoy": vmp_hoy,
            "p_vmp_ref": vmp_ref,
            "p_notas":   notas,
        }).execute()
        return True, "Sesión registrada correctamente."
    except Exception as exc:
        log.error("insertar_sesion: %s", exc)
        return False, str(exc)


def insertar_wellness(
    nombre: str,
    fecha: date,
    sueno: int,
    fatiga_hooper: int,
    estres: int,
    dolor: int,
    humor: int,
    notas: str = "",
) -> tuple[bool, str]:
    """Inserta cuestionario Hooper vía RPC insertar_wellness_hooper (SECURITY DEFINER)."""
    for campo, val in [("sueno", sueno), ("fatiga", fatiga_hooper),
                       ("estres", estres), ("dolor", dolor), ("humor", humor)]:
        if not (1 <= val <= 7):
            return False, f"'{campo}' fuera de rango Hooper [1, 7]: {val}"
    try:
        _get_client().rpc("insertar_wellness_hooper", {
            "p_nombre":  nombre,
            "p_fecha":   str(fecha),
            "p_sueno":   sueno,
            "p_fatiga":  fatiga_hooper,
            "p_estres":  estres,
            "p_dolor":   dolor,
            "p_humor":   humor,
            "p_notas":   notas,
        }).execute()
        return True, "Wellness registrado correctamente."
    except Exception as exc:
        log.error("insertar_wellness: %s", exc)
        return False, str(exc)


def insertar_carga_sesion(
    nombre: str,
    fecha: date,
    carga_ua: float,
    notas: str = "",
) -> tuple[bool, str]:
    """Inserta carga de sesión individual (UA) en tabla cargas_sesion."""
    if carga_ua < 0:
        return False, "carga_ua no puede ser negativa."
    try:
        _get_client().table("cargas_sesion").insert({
            "nombre":   nombre,
            "fecha":    str(fecha),
            "carga_ua": carga_ua,
            "notas":    notas,
        }).execute()
        return True, "Carga registrada."
    except Exception as exc:
        log.error("insertar_carga_sesion: %s", exc)
        return False, str(exc)


def insertar_carga_grupal_batch(
    fecha: str,
    df_ejercicios: pd.DataFrame,
    atletas: list[str],
    notas: str = "",
) -> tuple[bool, list[str]]:
    """
    Inserta sesión grupal en cargas_grupales + cargas_grupales_atletas.
    Valida plataforma y caída ANTES de cualquier operación de I/O.
    No toca sesiones_vmp.
    """
    errors: list[str] = []

    # Normalizar columnas (defensivo)
    df = df_ejercicios.copy()
    df.columns = [c.lower().strip() for c in df.columns]

    if not atletas:
        return False, ["La lista de atletas no puede estar vacía."]
    if df.empty:
        return False, ["El DataFrame de ejercicios no puede estar vacío."]

    plataformas_inv = set(df["tipo_plataforma"].unique()) - _VALID_PLATAFORMAS
    if plataformas_inv:
        errors.append(f"tipo_plataforma inválido: {plataformas_inv}. Válidos: {_VALID_PLATAFORMAS}")

    caidas_inv = set(df["tipo_caida"].unique()) - _VALID_CAIDAS
    if caidas_inv:
        errors.append(f"tipo_caida inválido: {caidas_inv}. Válidos: {_VALID_CAIDAS}")

    if errors:
        return False, errors

    try:
        client = _get_client()
        for _, row in df.iterrows():
            resp = (
                client.table("cargas_grupales")
                .insert({
                    "fecha":           str(fecha),
                    "tipo_plataforma": str(row["tipo_plataforma"]),
                    "altura_salto":    float(row["altura_salto"]),
                    "n_saltos":        int(row["n_saltos"]),
                    "tipo_caida":      str(row["tipo_caida"]),
                    "notas":           notas,
                })
                .execute()
            )
            carga_id = resp.data[0]["id"]
            for atleta in atletas:
                client.table("cargas_grupales_atletas").insert({
                    "carga_grupal_id": carga_id,
                    "nombre":          atleta,
                }).execute()
        return True, []
    except Exception as exc:
        log.error("insertar_carga_grupal_batch: %s", exc)
        return False, [str(exc)]


# ─────────────────────────────────────────────────────────────────────────────
# IMPORTACIÓN MASIVA CSV
# ─────────────────────────────────────────────────────────────────────────────

def importar_dataframe(df: pd.DataFrame) -> tuple[int, int, list[str]]:
    """
    Importa sesiones VMP desde DataFrame normalizado (columnas snake_case).
    Retorna (insertados, omitidos, errores).
    """
    if len(df) > MAX_IMPORT_ROWS:
        return 0, 0, [
            f"Importación rechazada: {len(df)} filas supera límite {MAX_IMPORT_ROWS}. "
            "Divide el CSV en lotes."
        ]

    insertados = omitidos = 0
    errores: list[str] = []
    for _, row in df.iterrows():
        vmp = float(row.get("vmp_hoy", 0) or 0)
        if vmp <= 0:
            omitidos += 1
            continue
        ok, msg = insertar_sesion(
            nombre=str(row.get("nombre", "")).strip(),
            fecha=row.get("fecha", date.today()),
            vmp_hoy=vmp,
        )
        if ok:
            insertados += 1
        else:
            errores.append(f"{row.get('nombre', '?')}: {msg}")
    return insertados, omitidos, errores


def importar_wellness_dataframe(df: pd.DataFrame) -> tuple[int, int, list[str]]:
    """
    Importa registros Wellness desde DataFrame.
    Retorna (insertados, omitidos, errores).
    """
    if len(df) > MAX_IMPORT_ROWS:
        return 0, 0, [
            f"Importación rechazada: {len(df)} filas supera límite {MAX_IMPORT_ROWS}. "
            "Divide el CSV en lotes."
        ]

    insertados = omitidos = 0
    errores: list[str] = []
    for _, row in df.iterrows():
        ok, msg = insertar_wellness(
            nombre=str(row.get("Nombre", row.get("nombre", ""))),
            fecha=row.get("Fecha", row.get("fecha", date.today())),
            sueno=int(row.get("Sueno", row.get("sueno", 4))),
            fatiga_hooper=int(row.get("Fatiga", row.get("fatiga", 4))),
            estres=int(row.get("Estres", row.get("estres", 4))),
            dolor=int(row.get("Dolor", row.get("dolor", 4))),
            humor=int(row.get("Humor", row.get("humor", 4))),
            notas=str(row.get("Notas", row.get("notas", ""))),
        )
        if ok:
            insertados += 1
        else:
            errores.append(f"{row.get('Nombre', row.get('nombre', '?'))}: {msg}")
    return insertados, omitidos, errores


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS UI (solo factories de datos estáticos, sin lógica de negocio)
# ─────────────────────────────────────────────────────────────────────────────

def wellness_masivo_template(atletas: list[str]) -> pd.DataFrame:
    """
    Genera DataFrame plantilla para st.data_editor en la sub-tab Wellness Masivo.
    Valores por defecto = 4 (punto medio Likert).
    """
    return pd.DataFrame({
        "Nombre": atletas,
        "Sueño":  [4] * len(atletas),
        "Estrés": [4] * len(atletas),
        "Fatiga": [4] * len(atletas),
        "Humor":  [4] * len(atletas),
        "Dolor":  [4] * len(atletas),
    })
