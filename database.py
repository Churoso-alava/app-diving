# =============================================================================
#  database.py — Club Tornados · Capa de acceso a Supabase
# =============================================================================
#  Responsabilidades:
#   - Conexión singleton a Supabase (cached)
#   - CRUD completo sobre la tabla `sesiones`
#   - Conversión a DataFrame normalizado para el motor fuzzy
# =============================================================================

from __future__ import annotations
import streamlit as st
import pandas as pd
from datetime import date
from supabase import create_client, Client


# ── Singleton de conexión ─────────────────────────────────────────────────────

@st.cache_resource
def get_supabase() -> Client:
    """Retorna el cliente Supabase. Se instancia una sola vez por sesión."""
    url = st.secrets["supabase"]["url"]
    key = st.secrets["supabase"]["key"]
    return create_client(url, key)


# ── READ ──────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=30)
def cargar_sesiones() -> pd.DataFrame:
    """
    Carga todas las sesiones desde Supabase y retorna un DataFrame
    con los nombres de columna estándar que espera el motor fuzzy:
    Nombre | Fecha | VMP_Hoy | id | notas
    """
    client = get_supabase()
    resp = (
        client.table("sesiones")
        .select("id, nombre, fecha, vmp_hoy, notas")
        .order("fecha", desc=False)
        .execute()
    )

    if not resp.data:
        return pd.DataFrame(columns=["id", "Nombre", "Fecha", "VMP_Hoy", "notas"])

    df = pd.DataFrame(resp.data)
    df = df.rename(columns={
        "nombre": "Nombre",
        "fecha":  "Fecha",
        "vmp_hoy": "VMP_Hoy",
    })
    df["Fecha"]   = pd.to_datetime(df["Fecha"])
    df["VMP_Hoy"] = pd.to_numeric(df["VMP_Hoy"], errors="coerce")
    df = df.dropna(subset=["Nombre", "Fecha", "VMP_Hoy"])
    return df.sort_values(["Nombre", "Fecha"]).reset_index(drop=True)


def cargar_atletas() -> list[str]:
    """Lista de atletas registrados en la tabla `atletas`."""
    client = get_supabase()
    resp = (
        client.table("atletas")
        .select("nombre")
        .eq("activo", True)
        .order("nombre")
        .execute()
    )
    return [r["nombre"] for r in resp.data] if resp.data else []


# ── CREATE ────────────────────────────────────────────────────────────────────

def insertar_sesion(
    nombre: str,
    fecha: date,
    vmp: float,
    notas: str = "",
) -> tuple[bool, str]:
    """
    Inserta una sesión nueva.
    Retorna (éxito: bool, mensaje: str).
    La constraint UNIQUE (nombre, fecha) evita duplicados silenciosos.
    """
    client = get_supabase()
    try:
        client.table("sesiones").insert({
            "nombre":  nombre.strip(),
            "fecha":   str(fecha),
            "vmp_hoy": round(float(vmp), 4),
            "notas":   notas.strip(),
        }).execute()
        st.cache_data.clear()
        return True, f"Sesión guardada: {nombre} · {fecha} · {vmp:.3f} m/s"
    except Exception as e:
        msg = str(e)
        if "unique" in msg.lower() or "23505" in msg:
            return False, f"Ya existe un registro para {nombre} el {fecha}. Usa 'Editar' para modificarlo."
        return False, f"Error al guardar: {msg}"


# ── UPDATE ────────────────────────────────────────────────────────────────────

def actualizar_sesion(
    session_id: int,
    vmp: float,
    notas: str = "",
) -> tuple[bool, str]:
    """Actualiza VMP y notas de una sesión existente por su id."""
    client = get_supabase()
    try:
        client.table("sesiones").update({
            "vmp_hoy": round(float(vmp), 4),
            "notas":   notas.strip(),
        }).eq("id", session_id).execute()
        st.cache_data.clear()
        return True, "Sesión actualizada correctamente."
    except Exception as e:
        return False, f"Error al actualizar: {e}"


# ── DELETE ────────────────────────────────────────────────────────────────────

def eliminar_sesion(session_id: int) -> tuple[bool, str]:
    """Elimina una sesión por id."""
    client = get_supabase()
    try:
        client.table("sesiones").delete().eq("id", session_id).execute()
        st.cache_data.clear()
        return True, "Sesión eliminada."
    except Exception as e:
        return False, f"Error al eliminar: {e}"


# ── BULK INSERT (migración desde Google Sheets / CSV) ─────────────────────────

def importar_dataframe(df: pd.DataFrame) -> tuple[int, int, list[str]]:
    """
    Inserta masivamente registros desde un DataFrame con columnas:
    Nombre | Fecha | VMP_Hoy
    Retorna (insertados, omitidos, errores[]).
    """
    client    = get_supabase()
    insertados = 0
    omitidos   = 0
    errores: list[str] = []

    records = []
    for _, row in df.iterrows():
        try:
            records.append({
                "nombre":  str(row["Nombre"]).strip(),
                "fecha":   str(pd.to_datetime(row["Fecha"]).date()),
                "vmp_hoy": round(float(row["VMP_Hoy"]), 4),
                "notas":   str(row.get("notas", "")),
            })
        except Exception as e:
            errores.append(f"Fila inválida: {row.to_dict()} → {e}")

    if records:
        try:
            # upsert ignora duplicados sin romper
            resp = client.table("sesiones").upsert(
                records,
                on_conflict="nombre,fecha",
                ignore_duplicates=True,
            ).execute()
            insertados = len(resp.data) if resp.data else 0
            omitidos   = len(records) - insertados
        except Exception as e:
            errores.append(f"Error batch: {e}")

    st.cache_data.clear()
    return insertados, omitidos, errores
