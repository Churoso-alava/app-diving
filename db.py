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


# =============================================================================
#  WELLNESS — CRUD
# =============================================================================

_HOOPER_ITEMS = ("sueno", "fatiga_hooper", "estres", "dolor", "humor")


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
    """
    Inserta un registro de wellness (Cuestionario Hooper Modificado).
    Valida rango [1, 7] antes de escribir.
    Requiere tabla `wellness` en Supabase:
        id, nombre, fecha, sueno, fatiga_hooper, estres, dolor, humor,
        w_norm (float), notas, created_at.
    """
    items = {
        "sueno": sueno, "fatiga_hooper": fatiga_hooper,
        "estres": estres, "dolor": dolor, "humor": humor,
    }
    errores = [
        f"{k}={v} fuera de [1,7]"
        for k, v in items.items()
        if not (1 <= v <= 7)
    ]
    if not nombre or not nombre.strip():
        errores.insert(0, "nombre vacío")
    if errores:
        return False, "❌ Datos inválidos: " + "; ".join(errores)

    # w_norm calculado aquí para evitar duplicar lógica en la capa de UI
    w_sueno_n  = (7 - sueno)         / 6.0
    w_fat_n    = (7 - fatiga_hooper) / 6.0
    w_est_n    = (7 - estres)        / 6.0
    w_dol_n    = (7 - dolor)         / 6.0
    w_hum_n    = (humor - 1)         / 6.0
    w_norm     = round((w_sueno_n + w_fat_n + w_est_n + w_dol_n + w_hum_n) / 5.0, 4)

    try:
        client = get_client()
        client.table("wellness").insert({
            "nombre":        nombre.strip(),
            "fecha":         str(fecha),
            "sueno":         sueno,
            "fatiga_hooper": fatiga_hooper,
            "estres":        estres,
            "dolor":         dolor,
            "humor":         humor,
            # w_norm is GENERATED ALWAYS AS STORED in PostgreSQL — do NOT insert
            "notas":         notas.strip(),
        }).execute()
        log.info("insertar_wellness: %s %s W=%.2f", nombre, fecha, w_norm)
        return True, f"✅ Wellness guardado: {nombre} — {fecha} — W_norm={w_norm:.2f}"
    except Exception as exc:
        log.error("insertar_wellness error: %s", exc)
        return False, f"❌ Error al guardar wellness: {exc}"


def cargar_wellness(nombre: str | None = None) -> pd.DataFrame:
    """
    Carga registros de wellness. Si nombre se especifica, filtra por atleta.
    Retorna DataFrame con columnas del esquema wellness.
    """
    try:
        client = get_client()
        q = client.table("wellness").select("*").order("fecha", desc=True)
        if nombre:
            q = q.eq("nombre", nombre)
        resp = q.execute()
        if not resp.data:
            return pd.DataFrame()
        df = pd.DataFrame(resp.data)
        df["fecha"] = pd.to_datetime(df["fecha"])
        return df
    except Exception as exc:
        log.error("cargar_wellness falló: %s", exc)
        raise


def importar_wellness_dataframe(df: pd.DataFrame) -> tuple[int, int, list[str]]:
    """
    Importa masivamente registros de wellness desde un DataFrame.

    Columnas esperadas (nombres exactos tras normalización):
        Nombre, Fecha, Sueno, Fatiga, Estres, Dolor, Humor, Notas (opcional)

    Retorna (insertados, omitidos, errores).
    """
    # Normalización de columnas
    rename = {}
    for c in df.columns:
        cl = c.lower().strip()
        if cl in ("nombre", "atleta", "deportista"):
            rename[c] = "Nombre"
        elif cl in ("fecha", "date"):
            rename[c] = "Fecha"
        elif cl in ("sueno", "sueño", "sleep"):
            rename[c] = "Sueno"
        elif cl in ("fatiga", "fatiga_hooper", "tiredness"):
            rename[c] = "Fatiga"
        elif cl in ("estres", "estrés", "stress"):
            rename[c] = "Estres"
        elif cl in ("dolor", "pain", "muscle_soreness"):
            rename[c] = "Dolor"
        elif cl in ("humor", "mood"):
            rename[c] = "Humor"
        elif cl in ("notas", "notes", "comentarios"):
            rename[c] = "Notas"
    df = df.rename(columns=rename)

    requeridas = ["Nombre", "Fecha", "Sueno", "Fatiga", "Estres", "Dolor", "Humor"]
    faltantes  = [c for c in requeridas if c not in df.columns]
    if faltantes:
        return 0, len(df), [f"Columnas faltantes: {faltantes}"]

    insertados, omitidos = 0, 0
    errores: list[str] = []

    for _, row in df.iterrows():
        try:
            ok, msg = insertar_wellness(
                nombre=str(row["Nombre"]),
                fecha=pd.Timestamp(row["Fecha"]).date(),
                sueno=int(row["Sueno"]),
                fatiga_hooper=int(row["Fatiga"]),
                estres=int(row["Estres"]),
                dolor=int(row["Dolor"]),
                humor=int(row["Humor"]),
                notas=str(row.get("Notas", "") or ""),
            )
            if ok:
                insertados += 1
            else:
                omitidos += 1
                errores.append(f"{row['Nombre']} {row['Fecha']}: {msg}")
        except Exception as exc:
            omitidos += 1
            errores.append(f"{row.get('Nombre','?')} {row.get('Fecha','?')}: {exc}")

    log.info("importar_wellness_dataframe: %d insertados, %d omitidos.", insertados, omitidos)
    return insertados, omitidos, errores


# =============================================================================
#  CARGA SESIÓN CLAVADOS — [A7]
# =============================================================================

def insertar_carga_sesion(
    nombre: str,
    fecha: date,
    n_clavados: int,
    l_bruta: float,
    l_norm: float,
    w_norm: float,
    ci: float,
    zona_dominante: str,
    notas: str = "",
) -> tuple[bool, str]:
    """
    Persiste el resultado de una sesión de clavados (Carga Integrada).
    Valida rangos antes de escribir en la tabla cargas_sesion.
    Requiere migración: 20250411_create_cargas_sesion.sql
    Retorna (ok, mensaje).
    """
    errores: list[str] = []
    if not nombre or not nombre.strip():
        errores.append("nombre vacío")
    if n_clavados <= 0:
        errores.append(f"n_clavados={n_clavados} debe ser > 0")
    if not (0.0 <= l_norm <= 100.0):
        errores.append(f"l_norm={l_norm:.2f} fuera de [0, 100]")
    if not (0.0 <= w_norm <= 1.0):
        errores.append(f"w_norm={w_norm:.4f} fuera de [0, 1]")
    if not (0.0 <= ci <= 200.0):
        errores.append(f"ci={ci:.2f} fuera de [0, 200]")
    if errores:
        return False, "❌ Datos inválidos: " + "; ".join(errores)

    try:
        client = get_client()
        client.table("cargas_sesion").insert({
            "nombre":         nombre.strip(),
            "fecha":          str(fecha),
            "n_clavados":     n_clavados,
            "l_bruta":        round(l_bruta, 4),
            "l_norm":         round(l_norm, 4),
            "w_norm":         round(w_norm, 4),
            "ci":             round(ci, 4),
            "zona_dominante": zona_dominante,
            "notas":          notas.strip(),
        }).execute()
        log.info("insertar_carga_sesion: %s %s CI=%.1f zona=%s",
                 nombre, fecha, ci, zona_dominante)
        return True, (
            f"✅ Carga guardada: {nombre} — {fecha} — "
            f"CI={ci:.1f} ({zona_dominante})"
        )
    except Exception as exc:
        log.error("insertar_carga_sesion error: %s", exc)
        return False, f"❌ Error al guardar carga: {exc}"
