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
from datetime import date
from typing import Optional, List, Tuple
import pandas as pd
from supabase import create_client, Client

log = logging.getLogger(__name__)

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

def get_client() -> Optional[Client]:
    """Retorna cliente Supabase autenticado usando variables de entorno."""
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_KEY")

    if not url or not key:
        log.error("🔑 No se encontraron las credenciales de Supabase (SUPABASE_URL/SUPABASE_KEY)")
        return None
    try:
        return create_client(url, key)
    except Exception as e:
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


def cargar_atletas() -> List[str]:
    """Retorna lista de nombres de atletas activos."""
    client = get_client()
    if not client: return []
    try:
        resp = client.table("atletas").select("nombre").eq("activo", True).execute()
        return [r["nombre"] for r in (resp.data or [])]
    except Exception as exc:
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
            return pd.DataFrame()
        df = pd.DataFrame(resp.data)
        df.columns = [c.lower() for c in df.columns]

        df["fecha"]   = pd.to_datetime(df["fecha"], errors="coerce").dt.date
        df["vmp_hoy"] = pd.to_numeric(df["vmp_hoy"], errors="coerce")
        df["vmp_ref"] = pd.to_numeric(df.get("vmp_ref"), errors="coerce")

        return df
    except Exception as exc:
        log.error("cargar_sesiones_raw falló: %s", exc, exc_info=True)
        return pd.DataFrame()


def cargar_wellness_atleta(nombre: str) -> pd.DataFrame:
    """Retorna registros Wellness de un atleta."""
    client = get_client()
    if not client: return pd.DataFrame()
    try:
        resp = (
            client.table("wellness")
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


def insertar_carga_sesion(
    nombre: str,
    fecha: date,
    carga_ua: float,
    notas: str = "",
) -> Tuple[bool, str]:
    """Inserta carga de sesión individual."""
    client = get_client()
    if not client: return False, "No hay conexión a la base de datos."

    if carga_ua < 0:
        return False, "carga_ua no puede ser negativa."
    try:
        client.table("cargas_sesion").insert({
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
    atletas: List[str],
    notas: str = "",
) -> Tuple[bool, List[str]]:
    """Inserta sesión grupal en batch."""
    client = get_client()
    if not client: return False, ["No hay conexión a la base de datos."]

    errors: list[str] = []
    df = df_ejercicios.copy()
    df.columns = [c.lower().strip() for c in df.columns]

    if not atletas:
        return False, ["La lista de atletas no puede estar vacía."]
    if df.empty:
        return False, ["El DataFrame de ejercicios no puede estar vacío."]

    # Validaciones
    plataformas_inv = set(df["tipo_plataforma"].unique()) - _VALID_PLATAFORMAS
    if plataformas_inv:
        errors.append(f"tipo_plataforma inválido: {plataformas_inv}. Válidos: {_VALID_PLATAFORMAS}")

    caidas_inv = set(df["tipo_caida"].unique()) - _VALID_CAIDAS
    if caidas_inv:
        errors.append(f"tipo_caida inválido: {caidas_inv}. Válidos: {_VALID_CAIDAS}")

    if errors:
        return False, errors

    try:
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
