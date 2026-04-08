"""
db.py — Capa de acceso a Supabase
Toda consulta usa el cliente autenticado y respeta RLS.
No contiene lógica de negocio ni referencias a Streamlit.
"""
from __future__ import annotations

import logging
from datetime import date

import pandas as pd
import streamlit as st
from supabase import create_client, Client

log = logging.getLogger(__name__)


# =============================================================================
#  CLIENTE SUPABASE (singleton por sesión)
# =============================================================================

@st.cache_resource
def get_client() -> Client:
    """
    Crea el cliente Supabase con Service Role Key.
    ⚠️  CAMBIO CRÍTICO: Usar service_role_key (NO anon key)
    
    La anon key tiene RLS activa pero sin usuario autenticado → queries vacías.
    Service Role Key bypasa RLS para queries backend.
    
    Requiere en .streamlit/secrets.toml:
        [supabase]
        url = "https://xxxx.supabase.co"
        service_role_key = "eyJ..."   # ← Copiar de Settings > API > service_role secret
    """
    url = st.secrets["supabase"]["url"]
    service_role_key = st.secrets["supabase"]["service_role_key"]
    return create_client(url, service_role_key)


# =============================================================================
#  LECTURA
# =============================================================================

def cargar_sesiones() -> pd.DataFrame:
    """Carga todas las sesiones visibles para el usuario autenticado (RLS)."""
    try:
        client = get_client()
        resp = client.table("sesiones").select("*").execute()
        
        if not resp.data:
            log.warning("cargar_sesiones: tabla vacía.")
            return pd.DataFrame(columns=["id", "Nombre", "Fecha", "VMP_Hoy", "notas"])
        
        df = pd.DataFrame(resp.data)
        df = df.rename(columns={
            "nombre": "Nombre",
            "fecha": "Fecha",
            "vmp_hoy": "VMP_Hoy",
        })
        df["Fecha"] = pd.to_datetime(df["Fecha"])
        df["VMP_Hoy"] = pd.to_numeric(df["VMP_Hoy"], errors="coerce")
        
        log.info("cargar_sesiones: %d registros.", len(df))
        return df
    except Exception as exc:
        log.error("cargar_sesiones falló: %s", exc)
        raise


def cargar_atletas() -> list[str]:
    """Retorna lista de nombres únicos de atletas."""
    try:
        client = get_client()
        resp = client.table("atletas").select("nombre").order("nombre").execute()
        nombres = [r["nombre"] for r in resp.data]
        log.info("cargar_atletas: %d atletas.", len(nombres))
        return nombres
    except Exception as exc:
        log.error("cargar_atletas falló: %s", exc)
        raise


# =============================================================================
#  ESCRITURA
# =============================================================================

def insertar_sesion(
    nombre: str,
    fecha:  date,
    vmp:    float,
    notas:  str = "",
) -> tuple[bool, str]:
    """Inserta una sesión validada. Retorna (ok, mensaje)."""
    try:
        from services import SessionInput
        SessionInput(nombre=nombre, fecha=str(fecha), vmp=vmp, notas=notas)  # valida
        client = get_client()
        client.table("sesiones").insert({
            "nombre":   nombre,
            "fecha":    str(fecha),
            "vmp_hoy":  vmp,
            "notas":    notas,
        }).execute()
        log.info("insertar_sesion: %s %s %.3f m/s", nombre, fecha, vmp)
        return True, f"✅ Sesión guardada: {nombre} — {fecha} — {vmp:.3f} m/s"
    except ValueError as exc:
        log.warning("insertar_sesion rechazada por validación: %s", exc)
        return False, f"❌ Datos inválidos: {exc}"
    except Exception as exc:
        log.error("insertar_sesion error inesperado: %s", exc)
        return False, f"❌ Error al guardar: {exc}"


def actualizar_sesion(
    sesion_id: int | str,
    nuevo_vmp: float,
    notas:     str = "",
) -> tuple[bool, str]:
    """Actualiza VMP y notas de una sesión por ID."""
    try:
        if not (0.100 <= nuevo_vmp <= 2.500):
            return False, f"❌ VMP {nuevo_vmp:.3f} fuera del rango fisiológico [0.100, 2.500]"
        client = get_client()
        client.table("sesiones").update({
            "vmp_hoy": nuevo_vmp,
            "notas":   notas,
        }).eq("id", sesion_id).execute()
        log.info("actualizar_sesion id=%s → %.3f m/s", sesion_id, nuevo_vmp)
        return True, f"✅ Sesión actualizada: {nuevo_vmp:.3f} m/s"
    except Exception as exc:
        log.error("actualizar_sesion error: %s", exc)
        return False, f"❌ Error al actualizar: {exc}"


def eliminar_sesion(sesion_id: int | str) -> tuple[bool, str]:
    """Elimina una sesión por ID."""
    try:
        client = get_client()
        client.table("sesiones").delete().eq("id", sesion_id).execute()
        log.info("eliminar_sesion id=%s", sesion_id)
        return True, "✅ Sesión eliminada."
    except Exception as exc:
        log.error("eliminar_sesion error: %s", exc)
        return False, f"❌ Error al eliminar: {exc}"


def importar_dataframe(df: pd.DataFrame) -> tuple[int, int, list[str]]:
    """
    Importa masivamente desde un DataFrame validado.
    Retorna (insertados, omitidos, errores).
    """
    insertados, omitidos = 0, 0
    errores: list[str] = []

    for _, row in df.iterrows():
        try:
            ok, msg = insertar_sesion(
                nombre=str(row["Nombre"]),
                fecha=row["Fecha"],
                vmp=float(row["VMP_Hoy"]),
                notas=str(row.get("notas", "") or ""),
            )
            if ok:
                insertados += 1
            else:
                omitidos += 1
                errores.append(f"{row['Nombre']} {row['Fecha']}: {msg}")
        except Exception as exc:
            omitidos += 1
            errores.append(f"{row.get('Nombre','?')} {row.get('Fecha','?')}: {exc}")

    log.info("importar_dataframe: %d insertados, %d omitidos.", insertados, omitidos)
    return insertados, omitidos, errores
