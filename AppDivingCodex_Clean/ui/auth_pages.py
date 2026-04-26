"""
ui/auth_pages.py — Interfaz de usuario para autenticación.
"""
import streamlit as st
from ui.auth_session import login_deportista, login_staff, is_authenticated, get_role

def render_login():
    """
    Renderiza la pantalla de login con dos pestañas: Deportista y Staff.
    """
    st.title("🌊 AppDivingCodex")
    st.subheader("Control de Fatiga y Monitoreo")

    tab_deportista, tab_staff = st.tabs(["👤 Deportista", "🛠️ Staff"])

    with tab_deportista:
        with st.form("login_deportista_form"):
            usuario = st.text_input("Usuario (ID Deportivo)")
            pin = st.text_input("PIN", type="password")
            submitted = st.form_submit_button("Entrar")
            if submitted:
                if login_deportista(usuario, pin):
                    st.success("¡Bienvenido!")
                    st.rerun()
                else:
                    st.error("Usuario o PIN incorrectos.")

    with tab_staff:
        with st.form("login_staff_form"):
            email = st.text_input("Email de Staff")
            password = st.text_input("Contraseña de Staff", type="password")
            submitted = st.form_submit_button("Acceder")
            if submitted:
                if login_staff(email, password):
                    st.success("Sesión de Staff iniciada.")
                    st.rerun()
                else:
                    # El error específico lo muestra login_staff
                    pass

def render_user_info():
    """
    Muestra información del usuario actual en el sidebar y botón de cerrar sesión.
    """
    from ui.auth_session import get_user_name, cerrar_sesion
    with st.sidebar:
        st.markdown("---")
        st.markdown(f"**Usuario:** {get_user_name()}")
        st.markdown(f"**Rol:** {get_role()}")
        if st.button("🚪 Cerrar Sesión"):
            cerrar_sesion()
