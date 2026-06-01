# Auth and User Management Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement a secure, role-based access control system (Staff vs Athlete) with simplified login for athletes and enhanced data entry/feedback features for staff.

**Architecture:** A modular integration within the existing Streamlit app, using Supabase for authentication and data isolation (RLS). UI logic is kept in `ui/` to avoid Streamlit dependencies in `core/`.

**Tech Stack:** Streamlit, Supabase, Python (hashlib/hmac for PBKDF2 SHA256).

**Git Root:** `C:/Users/PC/OneDrive/Desktop/AppDivingCodex`

### Task 1: Data Layer - Profile Management (Secure) [COMPLETED]

**Files:**
- Modify: `AppDivingCodex_Clean/data/db.py`

- [x] **Step 1: Add profile fetching and secure PBKDF2 validation**
- [x] **Step 2: Commit from repo root**

---

### Task 2: Auth Logic (UI Layer)

**Files:**
- Create: `AppDivingCodex_Clean/ui/auth_logic.py`

- [ ] **Step 1: Implement session management and login logic**

```python
import streamlit as st
from data.db import validar_credenciales_deportista, get_perfil_staff

def iniciar_sesion(perfil: dict):
    st.session_state.authenticated = True
    st.session_state.user_role = perfil["rol"]
    st.session_state.id_deportivo = perfil["id_deportivo"]
    st.session_state.nombre_usuario = perfil.get("usuario_acceso", perfil.get("email"))

def cerrar_sesion():
    for key in ["authenticated", "user_role", "id_deportivo", "nombre_usuario"]:
        if key in st.session_state:
            del st.session_state[key]
    st.rerun()

def is_authenticated():
    return st.session_state.get("authenticated", False)
```

- [ ] **Step 2: Commit auth logic from repo root**

```bash
cd ..
git add AppDivingCodex_Clean/ui/auth_logic.py
git commit -m "feat(ui): implement auth session management in UI layer"
```

---

### Task 3: Authentication UI

**Files:**
- Create: `AppDivingCodex_Clean/ui/auth_pages.py`

- [ ] **Step 1: Build the login interface with role selection**

```python
import streamlit as st
from ui.auth_logic import iniciar_sesion, is_authenticated
from data.db import validar_credenciales_deportista

def render_login():
    st.title("🌊 AppDivingCodex - Acceso")
    
    rol_sel = st.radio("Entrar como:", ["Deportista/Padre", "Staff Técnico"])
    
    if rol_sel == "Staff Técnico":
        with st.form("login_staff"):
            email = st.text_input("Correo")
            password = st.text_input("Contraseña", type="password")
            if st.form_submit_button("Entrar"):
                # Simulación para el piloto - En producción usar Supabase Auth real
                if email == "staff@diving.com" and password == "pilot123":
                    iniciar_sesion({"rol": "staff", "id_deportivo": "STAFF-01", "email": email})
                    st.rerun()
                else:
                    st.error("Credenciales inválidas")
    else:
        with st.form("login_deportista"):
            usuario = st.text_input("ID de Acceso")
            pin = st.text_input("PIN", type="password")
            if st.form_submit_button("Entrar"):
                perfil = validar_credenciales_deportista(usuario, pin)
                if perfil:
                    iniciar_sesion(perfil)
                    st.rerun()
                else:
                    st.error("ID o PIN incorrectos")
```

- [ ] **Step 2: Commit auth UI from repo root**

```bash
cd ..
git add AppDivingCodex_Clean/ui/auth_pages.py
git commit -m "feat(ui): add login interface for staff and athletes"
```

---

### Task 4: Staff Features - Data Entry Enhancements

**Files:**
- Create: `AppDivingCodex_Clean/ui/input_forms.py`

- [ ] **Step 1: Implement Wellness entry with emojis**

```python
import streamlit as st
from data.db import insertar_wellness, cargar_atletas

def form_wellness_emoji():
    st.subheader("📝 Registro Wellness (Escala Hooper)")
    atletas = cargar_atletas()
    atleta = st.selectbox("Seleccione Deportista", atletas)
    
    emojis = {1: "😫", 2: "🙁", 3: "😐", 4: "🙂", 5: "😊", 6: "😃", 7: "🤩"}
    
    col1, col2, col3 = st.columns(3)
    with col1:
        sueno = st.select_slider("Calidad de Sueño", options=list(emojis.keys()), format_func=lambda x: emojis[x])
        fatiga = st.select_slider("Nivel de Fatiga", options=list(emojis.keys()), format_func=lambda x: emojis[x])
    with col2:
        estres = st.select_slider("Nivel de Estrés", options=list(emojis.keys()), format_func=lambda x: emojis[x])
        dolor = st.select_slider("Nivel de Dolor", options=list(emojis.keys()), format_func=lambda x: emojis[x])
    with col3:
        humor = st.select_slider("Estado de Humor", options=list(emojis.keys()), format_func=lambda x: emojis[x])
    
    if st.button("Guardar Wellness"):
        # Lógica de inserción...
        pass
```

- [ ] **Step 2: Commit input forms from repo root**

```bash
cd ..
git add AppDivingCodex_Clean/ui/input_forms.py
git commit -m "feat(ui): implement wellness emoji form"
```

---

### Task 5: App Integration and Routing

**Files:**
- Modify: `AppDivingCodex_Clean/app.py`

- [ ] **Step 1: Update app.py to handle authentication routing**

```python
from ui.auth_logic import is_authenticated, cerrar_sesion
from ui.auth_pages import render_login

if not is_authenticated():
    render_login()
    st.stop()

# Si está autenticado, mostrar Sidebar con Logout
with st.sidebar:
    st.write(f"Conectado como: **{st.session_state.nombre_usuario}**")
    if st.button("Cerrar Sesión"):
        cerrar_sesion()

# Routing basado en rol
if st.session_state.user_role == "staff":
    # Renderizar UI de Staff (Tabs existentes + nuevos forms)
    pass
else:
    # Renderizar UI de Deportista (Vista simplificada)
    pass
```

- [ ] **Step 2: Commit app integration from repo root**

```bash
cd ..
git add AppDivingCodex_Clean/app.py
git commit -m "feat: integrate authentication routing in main app"
```

---

## Checkpoint 2026-04-27 (Post-RLS Investigation)
- **Completado:** Initial UI implementation for Staff (Wellness Hooper, Data Entry).
- **Supabase:** Data models (`perfiles`, `wellness_hooper`, etc.) exist. Initial RLS policy for staff was recursive.
- **App:** Authentication flow implemented, with distinct UI for staff and athletes.
- **Investigation Findings:**
    - Identified `infinite recursion detected in policy for relation "perfiles"` error due to RLS policy structure.
    - Revised RLS policy to avoid recursion.
    - Subsequent tests revealed a persistent `500 Internal Server Error` when querying the `perfiles` table from the application, indicating a server-side issue on Supabase.
- **Root Cause Hypothesis:** The `500` error likely stems from either:
    - A remaining misconfiguration in RLS policies for `perfiles`.
    - An issue with the application's Supabase client interaction in the Streamlit environment.
- **Pendientes:**
  - [ ] **Resolve `500 Internal Server Error`:** Deeply review all RLS policies on the `perfiles` table for any remaining conflicts or errors. Consider temporarily disabling RLS on `perfiles` to confirm it's the source.
  - [ ] **Verify Supabase Client Initialization:** Ensure `data/db.py`'s `get_client()` is correctly configured and initialized within the Streamlit context.
  - [ ] Implement historical data views for athletes (Wellness).
  - [ ] General refactor of `app.py` to separate `tab_ingreso` logic.
- **How to proceed:** Focus on debugging the `perfiles` table's RLS policies and Supabase client interaction within the Streamlit app.
