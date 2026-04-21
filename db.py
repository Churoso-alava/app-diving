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
import streamlit as st
from supabase import create_client

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

@st.cache_data
def get_divers():
    client = _get_client()
    # Ejecutar la consulta y extraer 'data'
    res = client.table("divers").select("*").execute()
    return pd.DataFrame(res.data)
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
    try:
        # 1. Asegúrate de que el nombre de la tabla sea exacto en Supabase
        resp = _get_client().table("sesiones_vmp").select("*").order("fecha").execute()
        df = pd.DataFrame(resp.data or [])
        
        if df.empty:
            log.warning("La tabla 'sesiones_vmp' respondió con datos vacíos.")
            return df

        # 2. Normalizar nombres de columnas a minúsculas para evitar el KeyError
        df.columns = [c.lower() for c in df.columns]

        # Log NaT values after date coercion
        df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce").dt.date
        initial_rows = len(df)
        if df["fecha"].isnull().any():
            log.warning("cargar_sesiones_raw: NaT values found in 'fecha' column after coercion.")
        
        df_filtered_fecha = df.dropna(subset=["fecha"])
        rows_discarded_fecha = initial_rows - len(df_filtered_fecha)
        if rows_discarded_fecha > 0:
            log.info(f"cargar_sesiones_raw: Discarded {rows_discarded_fecha} rows due to NaT in 'fecha' column.")
        
        df_intermediate = df_filtered_fecha

        # Coerce numeric columns and validate using helper
        numeric_cols_to_validate = []
        if "vmp_hoy" in df_intermediate.columns:
            df_intermediate["vmp_hoy"] = pd.to_numeric(df_intermediate["vmp_hoy"], errors="coerce")
            if df_intermediate["vmp_hoy"].isnull().any():
                log.warning("cargar_sesiones_raw: NaN values found in 'vmp_hoy' column after coercion.")
            numeric_cols_to_validate.append("vmp_hoy")
            
        if "vmp_ref" in df_intermediate.columns: # Check if vmp_ref exists before processing
            df_intermediate["vmp_ref"] = pd.to_numeric(df_intermediate.get("vmp_ref"), errors="coerce")
            if df_intermediate["vmp_ref"].isnull().any():
                log.warning("cargar_sesiones_raw: NaN values found in 'vmp_ref' column after coercion.")
            numeric_cols_to_validate.append("vmp_ref")

        df_final = df_intermediate
        if numeric_cols_to_validate:
             df_final = _validar_numericos(df_intermediate, numeric_cols_to_validate)

        # Log total discarded rows across both filters if any rows were discarded
        total_rows_discarded = initial_rows - len(df_final)
        if total_rows_discarded > 0:
            log.info(f"cargar_sesiones_raw: Total rows discarded due to NaT/NaN: {total_rows_discarded}.")

        return df_final
    except Exception as exc:
        # Si ves 0 registros, revisa los logs de tu terminal, aquí saldrá el error real
        log.error("cargar_sesiones_raw falló: %s", exc)
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


def cargar_lesiones() -> pd.DataFrame:
    """Retorna todos los registros de la tabla lesiones ordenados por fecha_inicio DESC."""
    try:
        resp = (
            _get_client()
            .table("lesiones")
            .select("*")
            .order("fecha_inicio", desc=True)
            .execute()
        )
        df = pd.DataFrame(resp.data or [])
        if df.empty:
            return df

        # Normalizar nombres de columnas a minúsculas
        df.columns = [c.lower() for c in df.columns]

        if "fecha_inicio" in df.columns:
            df["fecha_inicio"] = pd.to_datetime(df["fecha_inicio"], errors="coerce").dt.date
            if df["fecha_inicio"].isnull().any():
                log.warning("cargar_lesiones: NaT values found in 'fecha_inicio' column after coercion.")
        if "fecha_alta" in df.columns:
            df["fecha_alta"] = pd.to_datetime(df["fecha_alta"], errors="coerce").dt.date
            if df["fecha_alta"].isnull().any():
                log.warning("cargar_lesiones: NaT values found in 'fecha_alta' column after coercion.")
            
        # Filter out rows with NaT in critical date columns
        initial_rows_lesiones = len(df)
        df_filtered = df.copy() # Create a copy to avoid SettingWithCopyWarning
        if "fecha_inicio" in df_filtered.columns:
            df_filtered = df_filtered.dropna(subset=["fecha_inicio"])
        if "fecha_alta" in df_filtered.columns:
            df_filtered = df_filtered.dropna(subset=["fecha_alta"])

        rows_discarded = initial_rows_lesiones - len(df_filtered)
        if rows_discarded > 0:
            log.info(f"cargar_lesiones: Discarded {rows_discarded} rows due to NaT in critical date columns.")
            
        return df_filtered
    except Exception as exc:
        log.error("cargar_lesiones: %s", exc)
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


def insertar_lesion(
    atleta: str,
    fecha_inicio: date,
    zona_cuerpo: str,
    sistema: str,
    estado: str,
    notas: str = "",
    fecha_alta: date | None = None,
) -> tuple[bool, str]:
    """Inserta una nueva lesión en la tabla lesiones."""
    # Validaciones básicas de dominio
    valid_sistemas = {'Musculoesquelético', 'Cardiopulmonar', 'Neuromuscular', 'Tegumentario'}
    valid_estados = {'Lesionado', 'Sigue lesionado', 'Alta'}
    
    if sistema not in valid_sistemas:
        return False, f"Sistema inválido: {sistema}. Válidos: {valid_sistemas}"
    if estado not in valid_estados:
        return False, f"Estado inválido: {estado}. Válidos: {valid_estados}"

    try:
        data = {
            "atleta":       atleta,
            "fecha_inicio": str(fecha_inicio),
            "zona_cuerpo":  zona_cuerpo,
            "sistema":      sistema,
            "estado":       estado,
            "notas":        notas,
        }
        if fecha_alta:
            data["fecha_alta"] = str(fecha_alta)
            
        _get_client().table("lesiones").insert(data).execute()
        return True, "Lesión registrada correctamente."
    except Exception as exc:
        log.error("insertar_lesion: %s", exc)
        return False, str(exc)


def actualizar_estado_lesion(
    lesion_id: str,
    nuevo_estado: str,
    fecha_alta: date | None = None,
) -> tuple[bool, str]:
    """Actualiza el estado y opcionalmente la fecha de alta de una lesión."""
    valid_estados = {'Lesionado', 'Sigue lesionado', 'Alta'}
    if nuevo_estado not in valid_estados:
        return False, f"Estado inválido: {nuevo_estado}. Válidos: {valid_estados}"

    try:
        update_data = {"estado": nuevo_estado}
        if fecha_alta:
            update_data["fecha_alta"] = str(fecha_alta)
            
        _get_client().table("lesiones").update(update_data).eq("id", lesion_id).execute()
        return True, "Estado de lesión actualizado."
    except Exception as exc:
        log.error("actualizar_estado_lesion: %s", exc)
        return False, str(exc)


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


# The existing importar_wellness_dataframe function content
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

# New placeholder function
def importar_mesociclo_batch(df: pd.DataFrame) -> tuple[int, int, list[str]]:
    """
    Importa entradas de mesociclo desde DataFrame.
    Itera sobre fechas y realiza operaciones de carga según la estructura del mesociclo.
    Retorna (insertados, omitidos, errores).
    """
    # TODO: Implementar lógica específica para importar mesociclos.
    # Esta función es un placeholder.
    log.warning("importar_mesociclo_batch: No implementado.")
    return 0, 0, ["Función 'importar_mesociclo_batch' aún no implementada."]

# The existing wellness_masivo_template function content
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



# ─────────────────────────────────────────────────────────────────────────────
# HELPERS UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def _validar_numericos(df: pd.DataFrame, columnas: list[str]) -> pd.DataFrame:
    """
    Filtra filas de un DataFrame donde las columnas especificadas contengan NaN.
    Retorna el DataFrame filtrado y registra cuántas filas se eliminaron.
    Solo considera columnas que existen en el DataFrame.
    """
    initial_rows = len(df)
    existing_columns = [col for col in columnas if col in df.columns]

    if not existing_columns:
        if columnas:
            log.warning(f"Filtro _validar_numericos: No specified columns found in DataFrame: {columnas}.")
        return df # Return original df if no valid columns to filter by

    # Create a boolean mask for rows where all specified existing columns are NOT NaN
    # .notna() returns True for non-NaN values. .all(axis=1) checks if all are True.
    mask = df[existing_columns].notna().all(axis=1)

    df_filtered = df.loc[mask] # Apply the mask to the original DataFrame

    rows_discarded = initial_rows - len(df_filtered)
    if rows_discarded > 0:
        log.warning(f"Filtro _validar_numericos: Descartadas {rows_discarded} filas con NaN en columnas existentes: {existing_columns}.")
    return df_filtered
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
