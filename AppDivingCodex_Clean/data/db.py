"""
data/db.py — Capa de persistencia (Supabase).
Capa CRUD pura. Sin dependencias de Streamlit ni lógica de negocio.
Las credenciales deben proveerse vía variables de entorno:
  - SUPABASE_URL
  - SUPABASE_KEY
"""
from __future__ import annotations
import logging
import os
import base64
import hashlib
import hmac
import json
from datetime import date
from typing import Any, Optional, List, Tuple
from core.schemas import InjuryInput, ESTADO_LESION_OPTIONS
import pandas as pd
import streamlit as st
from supabase import create_client, Client

log = logging.getLogger(__name__)
_LAST_DB_ERROR: Optional[str] = None

# ── Límite anti-DoS para importaciones masivas (V-DOS) ──────────────────────
MAX_IMPORT_ROWS: int = 500

# ── Conjuntos válidos para carga grupal ─────────────────────────────────────
_VALID_PLATAFORMAS: frozenset[str] = frozenset({"trampolín", "plataforma"})
_VALID_CAIDAS:      frozenset[str] = frozenset({"pie", "mano"})

# ── Seguridad: hashing de PIN (PBKDF2) ──────────────────────────────────────
# Formato almacenado en DB (columna `pin_hashed`):
#   pbkdf2_sha256$<iterations>$<salt_b64>$<dk_b64>
# No requiere dependencias externas (bcrypt/argon2), pero es suficientemente
# robusto para un piloto si el número de iteraciones es alto.
_PIN_HASH_ITERS: int = 200_000


def _hash_pin_pbkdf2_sha256(pin: str, *, salt: bytes, iterations: int) -> bytes:
    return hashlib.pbkdf2_hmac("sha256", pin.encode("utf-8"), salt, iterations, dklen=32)


def crear_pin_hash(pin: str, *, iterations: int = _PIN_HASH_ITERS, salt_b64: str | None = None) -> str:
    """
    Crea un hash para almacenar en `perfiles.pin_hashed`.
    Si `salt_b64` no se pasa, se genera uno nuevo.
    """
    if not pin or not pin.strip():
        raise ValueError("PIN no puede estar vacío")
    if salt_b64 is None:
        salt = os.urandom(16)
        salt_b64 = base64.b64encode(salt).decode("ascii")
    else:
        salt = base64.b64decode(salt_b64.encode("ascii"))
    dk = _hash_pin_pbkdf2_sha256(pin.strip(), salt=salt, iterations=iterations)
    dk_b64 = base64.b64encode(dk).decode("ascii")
    return f"pbkdf2_sha256${iterations}${salt_b64}${dk_b64}"


def verificar_pin(pin: str, pin_hashed: str) -> bool:
    """Verifica PIN contra el formato `pbkdf2_sha256$...`."""
    try:
        scheme, iters_s, salt_b64, dk_b64 = pin_hashed.split("$", 3)
        if scheme != "pbkdf2_sha256":
            return False
        iterations = int(iters_s)
        salt = base64.b64decode(salt_b64.encode("ascii"))
        expected = base64.b64decode(dk_b64.encode("ascii"))
        actual = _hash_pin_pbkdf2_sha256(pin.strip(), salt=salt, iterations=iterations)
        return hmac.compare_digest(actual, expected)
    except Exception:
        return False


# ─────────────────────────────────────────────────────────────────────────────
# CLIENTE SUPABASE
# ─────────────────────────────────────────────────────────────────────────────

def _read_config_entry(name: str) -> tuple[Optional[str], Optional[str]]:
    value = os.environ.get(name)
    if value:
        return value, "environment"
    try:
        value = st.secrets.get(name)
    except Exception:
        value = None
    return (str(value), "streamlit secrets") if value else (None, None)


def _read_config_value(name: str) -> Optional[str]:
    value, _source = _read_config_entry(name)
    return value


def _resolve_supabase_credentials_with_source() -> tuple[Optional[str], Optional[str], Optional[str]]:
    url, _url_source = _read_config_entry("SUPABASE_URL")
    for key_name in ("SUPABASE_SERVICE_ROLE_KEY", "SUPABASE_SECRET_KEY", "SUPABASE_KEY"):
        key, _key_source = _read_config_entry(key_name)
        if key:
            return url, key, key_name
    return url, None, None


def _resolve_supabase_credentials() -> tuple[Optional[str], Optional[str]]:
    url, key, _key_name = _resolve_supabase_credentials_with_source()
    return url, key


def _decode_jwt_role(key: str) -> Optional[str]:
    parts = key.split(".")
    if len(parts) < 2:
        return None
    try:
        payload = parts[1] + "=" * (-len(parts[1]) % 4)
        decoded = json.loads(base64.urlsafe_b64decode(payload.encode("ascii")).decode("utf-8"))
        role = decoded.get("role")
        return str(role) if role else None
    except Exception:
        return None


def _service_key_config_error(key_name: Optional[str], key: Optional[str]) -> Optional[str]:
    if key_name not in {"SUPABASE_SERVICE_ROLE_KEY", "SUPABASE_SECRET_KEY"} or not key:
        return None

    role = _decode_jwt_role(key)
    if role == "anon":
        return (
            f"{key_name} esta configurada con una key de role anon. "
            "Para esta app Streamlit server-side usa una Supabase Secret key "
            "(sb_secret_...) o la legacy service_role key real. "
            "La key anon/publica no puede leer tablas protegidas por RLS ni "
            "objetos como public.users."
        )
    return None


def _format_db_error(error: Any) -> str:
    raw = str(error)
    if "permission denied for function is_staff_or_admin" in raw.lower():
        return (
            "Supabase bloqueo la lectura porque el rol de la API no puede ejecutar "
            "public.is_staff_or_admin(). Para uso local server-side, configura "
            "SUPABASE_SERVICE_ROLE_KEY en Streamlit secrets o en variables de entorno. "
            "Alternativa SQL: grant execute on function public.is_staff_or_admin() "
            "to anon, authenticated;"
        )
    if "permission denied for table users" in raw.lower():
        return (
            "Supabase bloqueo cargar atletas porque la consulta termina accediendo "
            "a public.users y el rol actual no tiene permisos. Verifica que "
            "SUPABASE_SERVICE_ROLE_KEY sea una Secret key/service_role real; la "
            "key anon/publica no es suficiente. Si decides no usar service_role, "
            "debes revisar GRANT/RLS para public.users, public.atletas y las vistas "
            "o politicas que las conectan."
        )
    return raw


def _set_last_db_error(error: Any) -> None:
    global _LAST_DB_ERROR
    _LAST_DB_ERROR = _format_db_error(error)


def clear_last_db_error() -> None:
    global _LAST_DB_ERROR
    _LAST_DB_ERROR = None


def get_last_db_error() -> Optional[str]:
    return _LAST_DB_ERROR


def get_client() -> Optional[Client]:
    """Retorna cliente Supabase autenticado usando secrets o variables de entorno."""
    url, key, key_name = _resolve_supabase_credentials_with_source()

    if not url or not key:
        message = "🔑 No se encontraron las credenciales de Supabase (SUPABASE_URL/SUPABASE_KEY)"
        _set_last_db_error(message)
        log.error(message)
        return None

    config_error = _service_key_config_error(key_name, key)
    if config_error:
        _set_last_db_error(config_error)
        log.error(config_error)
        return None

    try:
        return create_client(url, key)
    except Exception as e:
        _set_last_db_error(e)
        log.error("Error al crear cliente Supabase: %s", e)
        return None


# ─────────────────────────────────────────────────────────────────────────────
# LECTURA — SELECT
# ─────────────────────────────────────────────────────────────────────────────

def validar_credenciales_deportista(usuario: str, pin: str) -> Optional[dict]:
    """Valida ID + PIN y retorna el perfil si es correcto."""
    client = get_client()
    if not client: return None
    try:
        resp = (
            client.table("perfiles")
            .select("*")
            .eq("usuario_acceso", usuario)
            .eq("rol", "deportista")
            .execute()
        )
        perfil = resp.data[0] if resp.data else None
        if not perfil:
            return None

        pin_hashed = perfil.get("pin_hashed")
        if not pin_hashed:
            # No permitimos PIN en texto plano. Si aún existe una columna `pin`,
            # se debe migrar a `pin_hashed`.
            log.error("Perfil '%s' no tiene pin_hashed; migración requerida.", usuario)
            return None

        if not verificar_pin(pin, str(pin_hashed)):
            return None

        return perfil
    except Exception as exc:
        log.error("validar_credenciales_deportista: %s", exc)
        return None


def get_perfil_staff(email: str) -> Optional[dict]:
    """Retorna el perfil del staff por email."""
    client = get_client()
    if not client: return None
    try:
        resp = (
            client.table("perfiles")
            .select("*")
            .eq("email", email)
            .eq("rol", "staff")
            .execute()
        )
        return resp.data[0] if resp.data else None
    except Exception as exc:
        log.error("get_perfil_staff: %s", exc)
        return None


def borrar_registros_atleta(nombre: str) -> Tuple[bool, str]:
    """Borra todos los registros de un atleta de sesiones_vmp y wellness_hooper."""
    client = get_client()
    if not client: return False, "No hay conexión a la base de datos."
    try:
        client.table("sesiones_vmp").delete().eq("nombre", nombre).execute()
        client.table("wellness_hooper").delete().eq("nombre", nombre).execute()
        return True, f"Registros de {nombre} borrados correctamente."
    except Exception as exc:
        log.error("borrar_registros_atleta: %s", exc)
        return False, str(exc)


def cargar_atletas() -> List[str]:
    """Retorna lista de nombres de atletas activos."""
    client = get_client()
    if not client: return []
    try:
        resp = client.table("atletas").select("nombre").eq("activo", True).execute()
        clear_last_db_error()
        return [r["nombre"] for r in (resp.data or [])]
    except Exception as exc:
        _set_last_db_error(exc)
        log.error("cargar_atletas: %s", exc)
        return []


def cargar_sesiones_raw() -> pd.DataFrame:
    """Carga todas las sesiones de la tabla sesiones_vmp."""
    client = get_client()
    if not client: return pd.DataFrame()
    try:
        resp = client.table("sesiones_vmp").select("*").order("fecha").execute()
        if not resp.data:
            log.warning("La tabla 'sesiones_vmp' está vacía.")
            clear_last_db_error()
            return pd.DataFrame()
        df = pd.DataFrame(resp.data)
        df.columns = [c.lower() for c in df.columns]

        df["fecha"]   = pd.to_datetime(df["fecha"], errors="coerce").dt.date
        df["vmp_hoy"] = pd.to_numeric(df["vmp_hoy"], errors="coerce")
        df["vmp_ref"] = pd.to_numeric(df.get("vmp_ref"), errors="coerce")

        clear_last_db_error()
        return df
    except Exception as exc:
        _set_last_db_error(exc)
        log.error("cargar_sesiones_raw falló: %s", exc, exc_info=True)
        return pd.DataFrame()


def cargar_wellness_atleta(nombre: str) -> pd.DataFrame:
    """Retorna registros Wellness de un atleta."""
    client = get_client()
    if not client: return pd.DataFrame()
    try:
        resp = (
            client.table("wellness_hooper")
            .select("*")
            .eq("nombre", nombre)
            .order("fecha", desc=True)
            .execute()
        )
        df = pd.DataFrame(resp.data or [])
        if not df.empty:
            df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
        return df
    except Exception as exc:
        log.error("cargar_wellness_atleta(%s): %s", nombre, exc)
        return pd.DataFrame()


# ─────────────────────────────────────────────────────────────────────────────
# ESCRITURA — INSERT
# ─────────────────────────────────────────────────────────────────────────────

def insertar_sesion(
    nombre: str,
    fecha: date,
    vmp_hoy: float,
    vmp_ref: Optional[float] = None,
    notas: str = "",
) -> Tuple[bool, str]:
    """Inserta sesión VMP vía RPC."""
    client = get_client()
    if not client: return False, "No hay conexión a la base de datos."

    if not (0.1 <= vmp_hoy <= 2.5):
        return False, f"VMP fuera de rango fisiológico [0.1, 2.5]: {vmp_hoy}"
    if not nombre or not nombre.strip():
        return False, "nombre no puede estar vacío"
    try:
        client.rpc("insertar_sesion_vmp", {
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
) -> Tuple[bool, str]:
    """Inserta cuestionario Wellness vía RPC."""
    client = get_client()
    if not client: return False, "No hay conexión a la base de datos."

    for campo, val in [("sueno", sueno), ("fatiga", fatiga_hooper),
                       ("estres", estres), ("dolor", dolor), ("humor", humor)]:
        if not (1 <= val <= 7):
            return False, f"'{campo}' fuera de rango Hooper [1, 7]: {val}"
    try:
        client.rpc("insertar_wellness_hooper", {
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
        error_msg = str(exc)
        log.error("insertar_wellness: %s", exc)

        # --- Inicio de mejora para mensajes de error ---
        # Supabase/PostgreSQL usualmente devuelve errores de duplicidad en el mensaje de excepción.
        # Buscamos una frase común para identificar estos casos.
        if "duplicate key value violates unique constraint" in error_msg.lower():
            # Mensaje amigable para duplicados
            return False, "Error: Ya existe un registro de wellness para este atleta en esta fecha."
        else:
            # Mensaje genérico para otros tipos de error
            return False, "Ocurrió un error al registrar el wellness. Por favor, inténtalo de nuevo más tarde o contacta al soporte."
        # --- Fin de mejora para mensajes de error ---


def insertar_wellness_batch(sesiones: List[dict]) -> Tuple[bool, str]:
    """
    Inserta múltiples registros Wellness en batch.
    NOTA: El llamador es responsable de validar los datos (RPE, fechas, etc.) antes de llamar a esta función.
    """
    client = get_client()
    if not client: return False, "No hay conexión a la base de datos."
    
    try:
        # Supabase Python client handles list of dicts for bulk inserts
        client.table("wellness_hooper").insert(sesiones).execute()
        return True, "Registros Wellness guardados correctamente."
    except Exception as exc:
        log.error("insertar_wellness_batch: %s", exc)
        return False, str(exc)


def insertar_sesiones_batch(sesiones: List[dict]) -> Tuple[bool, str]:
    """
    Inserta múltiples sesiones VMP en batch.
    NOTA: El llamador es responsable de validar los datos (VMP, fechas, etc.) antes de llamar a esta función.
    """
    client = get_client()
    if not client: return False, "No hay conexión a la base de datos."
    
    try:
        # Supabase Python client handles list of dicts for bulk inserts
        client.table("sesiones_vmp").insert(sesiones).execute()
        return True, "Sesiones registradas correctamente."
    except Exception as exc:
        log.error("insertar_sesiones_batch: %s", exc)
        return False, str(exc)


# ─────────────────────────────────────────────────────────────────────────────
# CARGA DE ENTRENAMIENTO SUBJETIVA (RPE × Duración)
# ─────────────────────────────────────────────────────────────────────────────

def validate_carga_entrenamiento(rpe: int, duracion_min: int) -> tuple[bool, str]:
    """Valida RPE (1-10) y duración (>0 minutos)."""
    if not (isinstance(rpe, int) and 1 <= rpe <= 10):
        return False, f"RPE fuera de rango [1, 10]: {rpe}"
    if not (isinstance(duracion_min, int) and duracion_min > 0):
        return False, f"Duración debe ser un entero positivo (minutos): {duracion_min}"
    return True, ""


def insertar_carga_sesion(
    nombre: str,
    fecha: date,
    rpe: int,
    duracion_min: int,
) -> Tuple[bool, str]:
    """
    Guarda la percepción subjetiva (RPE 1-10) y la duración del entreno
    en la tabla sesiones_vmp mediante upsert por (nombre, fecha).
    Carga total = rpe * duracion_min (Foster Method).
    """
    ok, msg = validate_carga_entrenamiento(rpe, duracion_min)
    if not ok:
        return False, msg
    if not nombre or not nombre.strip():
        return False, "nombre no puede estar vacío"

    client = get_client()
    if not client:
        return False, "No hay conexión a la base de datos."
    try:
        client.table("sesiones_vmp").upsert(
            {
                "nombre": nombre.strip(),
                "fecha": str(fecha),
                "carga_subjetiva": rpe,
                "duracion_min": duracion_min,
            },
            on_conflict="nombre,fecha",
        ).execute()
        carga_total = rpe * duracion_min
        return True, f"Carga registrada: RPE {rpe} × {duracion_min} min = {carga_total} UA."
    except Exception as exc:
        log.error("insertar_carga_sesion(%s, %s): %s", nombre, fecha, exc)
        return False, str(exc)


def insertar_carga_sesion_batch(cargas: List[dict]) -> Tuple[bool, str]:
    """
    Inserta múltiples registros de carga en batch.
    Cada dict debe tener: nombre, fecha, carga_subjetiva, duracion_min.
    """
    client = get_client()
    if not client:
        return False, "No hay conexión a la base de datos."
    try:
        client.table("sesiones_vmp").upsert(
            cargas, on_conflict="nombre,fecha"
        ).execute()
        return True, f"{len(cargas)} registros de carga guardados correctamente."
    except Exception as exc:
        log.error("insertar_carga_sesion_batch: %s", exc)
        return False, str(exc)

def insertar_lesion(atleta: str, fecha_lesion: date, zona_corporal: str, tipo: str, gravedad: str, estado: str = "Activa", notas: str = "", fecha_alta: Optional[date] = None) -> Tuple[bool, str]:
    # Validate with InjuryInput
    try:
        injury = InjuryInput(
            atleta=atleta,
            fecha_lesion=str(fecha_lesion),
            zona_corporal=zona_corporal,
            tipo=tipo,
            gravedad=gravedad,
            estado=estado,
            notas=notas,
            fecha_alta=str(fecha_alta) if fecha_alta else None
        )
    except ValueError as e:
        return False, str(e)
    
    client = get_client()
    if not client: return False, "No hay conexión a la base de datos."
    
    try:
        client.table("lesiones").insert(injury.to_dict()).execute()
        return True, "Lesión registrada correctamente."
    except Exception as exc:
        log.error("insertar_lesion: %s", exc)
        return False, str(exc)

def cargar_lesiones_activas(atleta: Optional[str] = None) -> pd.DataFrame:
    client = get_client()
    if not client: return pd.DataFrame()
    try:
        query = client.table("lesiones").select("*").in_("estado", ["Activa", "Recuperación"])
        if atleta:
            query = query.eq("atleta", atleta)
        resp = query.order("fecha_lesion", desc=True).execute()
        return pd.DataFrame(resp.data or [])
    except Exception as exc:
        log.error("cargar_lesiones_activas: %s", exc)
        return pd.DataFrame()

def cargar_historial_lesiones(atleta: str) -> pd.DataFrame:
    client = get_client()
    if not client: return pd.DataFrame()
    try:
        resp = client.table("lesiones").select("*").eq("atleta", atleta).order("fecha_lesion", desc=True).execute()
        return pd.DataFrame(resp.data or [])
    except Exception as exc:
        log.error("cargar_historial_lesiones(%s): %s", atleta, exc)
        return pd.DataFrame()

def actualizar_estado_lesion(lesion_id: str, nuevo_estado: str, fecha_alta: Optional[date] = None) -> Tuple[bool, str]:
    if nuevo_estado not in ESTADO_LESION_OPTIONS:
        return False, f"Estado '{nuevo_estado}' inválido. Opciones: {ESTADO_LESION_OPTIONS}"
        
    client = get_client()
    if not client: return False, "No hay conexión a la base de datos."
    
    data = {"estado": nuevo_estado}
    if fecha_alta:
        data["fecha_alta"] = str(fecha_alta)
    
    try:
        client.table("lesiones").update(data).eq("id", lesion_id).execute()
        return True, f"Estado de lesión {lesion_id} actualizado a {nuevo_estado}."
    except Exception as exc:
        log.error("actualizar_estado_lesion(%s): %s", lesion_id, exc)
        return False, str(exc)
