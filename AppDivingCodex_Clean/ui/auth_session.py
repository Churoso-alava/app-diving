"""
ui/auth_session.py — Gestión de sesión de usuario con Streamlit.
Cumple con la Regla #1: Sin dependencias en core/.
"""
import streamlit as st
from typing import Optional, Any
from data.db import validar_credenciales_deportista, get_perfil_staff

# Claves en st.session_state
KEY_AUTHENTICATED = "auth_is_authenticated"
KEY_USER_PROFILE  = "auth_user_profile"

def iniciar_sesion(perfil: dict[str, Any]) -> None:
    """
    Registra el perfil del usuario en la sesión y marca como autenticado.
    """
    st.session_state[KEY_AUTHENTICATED] = True
    st.session_state[KEY_USER_PROFILE] = perfil

def cerrar_sesion() -> None:
    """
    Limpia los datos de la sesión y desautentica.
    """
    st.session_state[KEY_AUTHENTICATED] = False
    st.session_state[KEY_USER_PROFILE] = None
    if hasattr(st, "rerun"):
        st.rerun()

def is_authenticated() -> bool:
    """
    Verifica si hay un usuario autenticado en la sesión actual.
    """
    return st.session_state.get(KEY_AUTHENTICATED, False)

def get_role() -> Optional[str]:
    """
    Retorna el rol del usuario actual (ej: 'deportista', 'staff').
    """
    perfil = st.session_state.get(KEY_USER_PROFILE)
    if perfil:
        return perfil.get("rol")
    return None
def get_user_id() -> Optional[Any]:
    """
    Retorna el ID deportivo único del usuario autenticado.
    """
    perfil = st.session_state.get(KEY_USER_PROFILE)
    if perfil:
        return perfil.get("id_deportivo")
    return None

def get_user_name() -> Optional[str]:
    """
    Retorna el nombre de usuario o email para mostrar.
    """
    perfil = st.session_state.get(KEY_USER_PROFILE)
    if perfil:
        return perfil.get("usuario_acceso") or perfil.get("email")
    return None

# ─────────────────────────────────────────────────────────────────────────────
# LOGICA DE LOGIN (Regla #2 y #3)
# ─────────────────────────────────────────────────────────────────────────────

def login_deportista(usuario: str, pin: str) -> bool:
    """
    Valida credenciales de deportista y arranca sesión.
    Usa PBKDF2 (Regla #3).
    """
    perfil = validar_credenciales_deportista(usuario, pin)
    if perfil:
        iniciar_sesion(perfil)
        return True
    return False

def login_staff(email: str, password: str) -> bool:
    """
    Valida credenciales de staff.
    Usa st.secrets (Regla #2).
    """
    staff_pass = st.secrets.get("STAFF_PASSWORD")
    if not staff_pass:
        st.warning("⚠️ Login de staff deshabilitado temporalmente (falta configuración).")
        return False
    
    if password == staff_pass:
        perfil = get_perfil_staff(email)
        if perfil:
            iniciar_sesion(perfil)
            return True
        else:
            st.error("El email no corresponde a un perfil de staff.")
    else:
        st.error("Contraseña de staff incorrecta.")
    
    return False
