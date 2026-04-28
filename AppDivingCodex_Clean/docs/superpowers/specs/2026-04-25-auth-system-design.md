# Spec: Sistema de Gestión de Usuarios y Perfiles (AppDivingCodex)

**Fecha:** 2026-04-25
**Estado:** Draft
**Autor:** Gemini CLI Agent

## 1. Objetivo
Implementar un sistema de autenticación y gestión de perfiles para la aplicación de monitoreo de fatiga en deportistas de alto rendimiento (clavados). El sistema debe diferenciar claramente entre el personal técnico (**Staff**) y los **Deportistas** (menores de edad/padres), asegurando la privacidad de los datos y proporcionando feedback diferenciado según el rol.

## 2. Arquitectura del Sistema
Se adopta una arquitectura de **Single-App Modular** sobre Streamlit, utilizando Supabase como backend para la persistencia y seguridad.

### 2.1 Roles y Permisos
| Característica | Staff (Analista/Fisio/Entrenador) | Deportista (Atleta/Padres) |
| :--- | :--- | :--- |
| **Acceso** | Correo + Contraseña (Supabase Auth) | ID de Acceso + PIN (Simplificado) |
| **Permisos Escritura** | VMP, Wellness (Hooper), Carga (Indiv/Batch) | Ninguno (Lectura de feedback) |
| **Permisos Lectura** | Dashboard Global, Todos los Atletas | Perfil Propio EXCLUSIVO |
| **Feedback** | Técnico (Certeza, Métricas, Ruido Bio) | Motivador (Semáforo, Consejos) |

## 3. Componentes de Datos (Supabase)

### 3.1 Tabla `perfiles`
Extensión de la gestión de atletas para manejar el acceso.
- `id_deportivo` (UUID/Fijo): Clave primaria vinculada a los datos históricos.
- `rol` (text): 'staff' | 'deportista'.
- `usuario_acceso` (text/Unique): Para el login de deportistas.
- `pin_hashed` (text): PIN cifrado.
- `email` (text/Unique): Solo para Staff.

### 3.2 Seguridad RLS (Row Level Security)
- **Staff:** Política para leer y escribir en todas las tablas.
- **Deportista:** Política restrictiva que permite `SELECT` solo en filas donde `id_deportivo` coincida con el ID de su sesión activa.

## 4. Componentes de Interfaz (UI)

### 4.1 Módulo de Autenticación (`ui/auth.py`)
- Pantalla de inicio con selección de rol.
- Login Staff: Formulario estándar de email/pass.
- Login Deportista: Campo de ID y teclado numérico para PIN.
- Manejo de sesión mediante `st.session_state`.

### 4.2 Módulo de Registro (Staff Only)
- **Wellness Hooper:** Interfaz de selección mediante **emojis** (Escala 1-7) para registrar sueño, fatiga, estrés, dolor y humor.
- **Carga Grupal:** Formulario para aplicar la misma carga de trabajo a múltiples atletas simultáneamente.

### 4.3 Módulo de Feedback Diferenciado
- Lógica en `core/services.py` para generar dos versiones del `DiagnosticResult`.
- Visualización amigable para deportistas en la pestaña "Mi Estado".

## 5. Flujo de Datos
1. **Login:** El usuario se autentica y se carga su perfil en la sesión.
2. **Staff Input:** El Staff registra datos de VMP/Wellness/Carga.
3. **Análisis:** El `core/fuzzy_engine.py` procesa los datos.
4. **Visualización:**
   - Si es **Staff**, se muestra el Dashboard Global con detalles técnicos.
   - Si es **Deportista**, se muestra su estado individual con lenguaje sencillo.

## 6. Testing y Validación
- Pruebas de integración para asegurar que un deportista no pueda acceder a datos de otro (simulación de brecha RLS).
- Validación de formularios de entrada para evitar datos fuera de rango fisiológico.
- Verificación del hashing de PIN para seguridad de acceso.

---
**Auto-revisión del Spec:**
- **Placeholders:** No hay TBDs. Los rangos (1-7 para Wellness, 0-200 para Carga) están alineados con el código core existente.
- **Consistencia:** El rol de Staff como único escritor está explícito.
- **Alcance:** Enfocado en la gestión de usuarios y entrada de datos, sin tocar la lógica matemática del motor fuzzy que ya funciona.
