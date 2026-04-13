"""
db.py — NMF-Optimizer
Capa de acceso a Supabase. SQL-First: lógica en PostgreSQL via TPs.
Clark-Wilson: CDIs modificados solo via funciones SECURITY DEFINER.
"""
from __future__ import annotations

import logging
import os
from datetime import date
from typing import Any

import pandas as pd

log = logging.getLogger(__name__)

# ── Límite de seguridad para importaciones masivas (V-DOS) ──────────────────
MAX_IMPORT_ROWS: int = 500

# ── Validaciones de carga grupal ────────────────────────────────────────────
_VALID_PLATAFORMAS = {"trampolín", "plataforma"}
_VALID_CAIDAS = {"pie", "mano"}

def normalizar_columnas_bien(df: pd.DataFrame) -> pd.DataFrame:
    """Mapeo explícito de aliases a nombres estándar."""
    rename_map = {}
    lower_cols = {col.lower().strip(): col for col in df.columns}
    
    # Mapeo de aliases en múltiples idiomas
    aliases = {
        "nombre": ["nombre", "atleta", "deportista", "jugador"],
        "Fecha": ["fecha", "date", "fecha_sesion"],
        "vmp_hoy": ["vmp_hoy", "vmp", "velocidad", "mpv"],
    }
    
    for target, sources in aliases.items():
        for src in sources:
            if src in lower_cols:
                rename_map[lower_cols[src]] = target
                break
    
    return df.rename(columns=rename_map)


def _get_client():
    """Retorna cliente Supabase autenticado con service_role (nunca exponer en frontend)."""
    try:
        from supabase import create_client
        url = os.environ.get("SUPABASE_URL", "")
        key = os.environ.get("SUPABASE_SERVICE_KEY", "")
        return create_client(url, key)
    except Exception as exc:
        log.error("Supabase client error: %s", exc)
        raise


# ─────────────────────────────────────────────────────────────────────────────
# FUNCIONES EXISTENTES (stubs representativos del código original)
# ─────────────────────────────────────────────────────────────────────────────

def insertar_sesion(
    nombre: str,
    fecha: date,
    vmp_hoy: float,
    vmp_ref: float | None = None,
    notas: str = "",
) -> tuple[bool, str]:
    """Inserta sesión VMP via TP PostgreSQL (SECURITY DEFINER)."""
    if not (0.1 <= vmp_hoy <= 2.5):
        return False, f"VMP fuera de rango fisiológico: {vmp_hoy}"
    try:
        client = _get_client()
        client.rpc("insertar_sesion_vmp", {
            "p_nombre": nombre,
            "p_fecha": str(fecha),
            "p_vmp_hoy": vmp_hoy,
            "p_vmp_ref": vmp_ref,
            "p_notas": notas,
        }).execute()
        return True, "Sesión registrada correctamente."
    except Exception as exc:
        log.error("insertar_sesion error: %s", exc)
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
    """Inserta cuestionario Hooper modificado via TP PostgreSQL."""
    for campo, val in [("sueno", sueno), ("fatiga", fatiga_hooper),
                       ("estres", estres), ("dolor", dolor), ("humor", humor)]:
        if not (1 <= val <= 7):
            return False, f"Valor fuera de rango [1-7] en campo '{campo}': {val}"
    try:
        client = _get_client()
        client.rpc("insertar_wellness_hooper", {
            "p_nombre": nombre,
            "p_fecha": str(fecha),
            "p_sueno": sueno,
            "p_fatiga": fatiga_hooper,
            "p_estres": estres,
            "p_dolor": dolor,
            "p_humor": humor,
            "p_notas": notas,
        }).execute()
        return True, "Wellness registrado correctamente."
    except Exception as exc:
        log.error("insertar_wellness error: %s", exc)
        return False, str(exc)


def importar_dataframe(df: pd.DataFrame) -> tuple[int, int, list[str]]:
    """
    Importación masiva de sesiones VMP desde DataFrame.
    Retorna (insertados, omitidos, errores).
    """
    # Guard V-DOS — rechazar DataFrames que superen el límite operativo
    if len(df) > MAX_IMPORT_ROWS:
        return (
            0,
            0,
            [
                f"Importación rechazada: el archivo contiene {len(df)} filas "
                f"(límite operativo: {MAX_IMPORT_ROWS}). "
                "Divide el archivo en lotes más pequeños."
            ],
        )

    insertados, omitidos, errores = 0, 0, []
    for _, row in df.iterrows():
        vmp = float(row.get("VMP_Hoy", 0))
        if vmp <= 0:
            omitidos += 1
            continue
        ok, msg = insertar_sesion(
            nombre=str(row["Nombre"]),
            fecha=row.get("Fecha", date.today()),
            vmp_hoy=vmp,
        )
        if ok:
            insertados += 1
        else:
            errores.append(f"{row['Nombre']}: {msg}")
    return insertados, omitidos, errores


def importar_wellness_dataframe(df: pd.DataFrame) -> tuple[int, int, list[str]]:
    """
    Importación masiva de registros Wellness desde DataFrame.
    Retorna (insertados, omitidos, errores).
    """
    # Guard V-DOS
    if len(df) > MAX_IMPORT_ROWS:
        return (
            0,
            0,
            [
                f"Importación rechazada: el archivo contiene {len(df)} filas "
                f"(límite operativo: {MAX_IMPORT_ROWS}). "
                "Divide el archivo en lotes más pequeños."
            ],
        )

    insertados, omitidos, errores = 0, 0, []
    for _, row in df.iterrows():
        ok, msg = insertar_wellness(
            nombre=str(row["Nombre"]),
            fecha=row.get("Fecha", date.today()),
            sueno=int(row.get("Sueno", 4)),
            fatiga_hooper=int(row.get("Fatiga", 4)),
            estres=int(row.get("Estres", 4)),
            dolor=int(row.get("Dolor", 4)),
            humor=int(row.get("Humor", 4)),
            notas=str(row.get("Notas", "")),
        )
        if ok:
            insertados += 1
        else:
            errores.append(f"{row['Nombre']}: {msg}")
    return insertados, omitidos, errores


def cargar_atletas() -> list[str]:
    """Retorna lista de nombres de atletas activos."""
    try:
        client = _get_client()
        resp = client.table("atletas").select("nombre").eq("activo", True).execute()
        return [r["nombre"] for r in (resp.data or [])]
    except Exception as exc:
        log.error("cargar_atletas error: %s", exc)
        return []


def cargar_sesiones_raw() -> pd.DataFrame:
    """Retorna DataFrame con todas las sesiones VMP (CDI — solo lectura directa)."""
    try:
        client = _get_client()
        resp = client.table("sesiones_vmp").select("*").order("fecha").execute()
        return pd.DataFrame(resp.data or [])
    except Exception as exc:
        log.error("cargar_sesiones_raw error: %s", exc)
        return pd.DataFrame()


def insertar_carga_sesion(
    nombre: str,
    fecha: date,
    carga_ua: float,
    notas: str = "",
) -> tuple[bool, str]:
    """Inserta carga de sesión individual (UA)."""
    if carga_ua < 0:
        return False, "La carga UA no puede ser negativa."
    try:
        client = _get_client()
        client.table("cargas_sesion").insert({
            "nombre": nombre,
            "fecha": str(fecha),
            "carga_ua": carga_ua,
            "notas": notas,
        }).execute()
        return True, "Carga registrada."
    except Exception as exc:
        log.error("insertar_carga_sesion error: %s", exc)
        return False, str(exc)


# ── CARGA GRUPAL ─────────────────────────────────────────────────────────────

def insertar_carga_grupal_batch(
    fecha: str,
    df_ejercicios: pd.DataFrame,
    atletas: list[str],
    notas: str = "",
) -> tuple[bool, list[str]]:
    """
    Inserta una sesión grupal de carga en `cargas_grupales` y vincula
    cada atleta en `cargas_grupales_atletas`.

    Retorna (True, []) en éxito, (False, [errores]) en validación fallida.
    No toca la BD si falla validación.
    """
    errors: list[str] = []

    if not atletas:
        errors.append("La lista de atletas no puede estar vacía.")
        return False, errors

    if df_ejercicios.empty:
        errors.append("El DataFrame de ejercicios no puede estar vacío.")
        return False, errors

    plataformas_inv = set(df_ejercicios["tipo_plataforma"].unique()) - _VALID_PLATAFORMAS
    if plataformas_inv:
        errors.append(
            f"tipo_plataforma inválido: {plataformas_inv}. Válidos: {_VALID_PLATAFORMAS}"
        )

    caidas_inv = set(df_ejercicios["tipo_caida"].unique()) - _VALID_CAIDAS
    if caidas_inv:
        errors.append(
            f"tipo_caida inválido: {caidas_inv}. Válidos: {_VALID_CAIDAS}"
        )

    if errors:
        return False, errors

    try:
        client = _get_client()
        for _, row in df_ejercicios.iterrows():
            resp = (
                client.table("cargas_grupales")
                .insert({
                    "fecha":           str(fecha),
                    "tipo_plataforma": row["tipo_plataforma"],
                    "altura_salto":    float(row["altura_salto"]),
                    "n_saltos":        int(row["n_saltos"]),
                    "tipo_caida":      row["tipo_caida"],
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
        log.error("insertar_carga_grupal_batch error: %s", exc)
        return False, [str(exc)]


def wellness_masivo_template(atletas: list[str]) -> pd.DataFrame:
    """
    Genera un DataFrame editable con todos los atletas y columnas de wellness.
    Diseñado para uso con st.data_editor en la sub-tab Wellness Masivo.
    """
    return pd.DataFrame({
        "Nombre":  atletas,
        "Sueño":   [4] * len(atletas),
        "Estrés":  [4] * len(atletas),
        "Fatiga":  [4] * len(atletas),
        "Humor":   [4] * len(atletas),
        "Dolor":   [4] * len(atletas),
    })
