# Plan de Integración: Módulo Unificado de Eventos Médicos

**Objetivo:** Integrar el registro de Enfermedades (gripa, vómito, etc.) en el módulo existente de Lesiones, creando un sistema unificado de "Eventos Médicos".

**Estado Actual:**
- Backend (`data/db.py`) sincronizado con la tabla `lesiones` (mapeo `zona_cuerpo`, `sistema`).
- UI (`tab_lesiones.py`) simplificada, pero sin soporte para enfermedades.

---

### Tareas:

- [ ] **Task 1: Migración de Base de Datos**
    - Ejecutar SQL para añadir `tipo_evento` (ENUM: 'Lesión', 'Enfermedad') y `tipo_enfermedad` (TEXT) a la tabla `lesiones`.
    ```sql
    ALTER TABLE lesiones ADD COLUMN IF NOT EXISTS tipo_evento TEXT DEFAULT 'Lesión';
    ALTER TABLE lesiones ADD COLUMN IF NOT EXISTS tipo_enfermedad TEXT;
    ```

- [ ] **Task 2: Actualización de Backend (`data/db.py`)**
    - Modificar `insertar_lesion` para aceptar `tipo_evento` y `tipo_enfermedad`.
    - Ajustar la lógica de inserción para que, si es 'Enfermedad', no valide campos específicos de lesiones (como `zona_cuerpo` o `sistema`).

- [ ] **Task 3: Refactorización de Frontend (`tab_lesiones.py`)**
    - Actualizar `_render_form_registro`: Añadir selector `tipo_evento`.
    - Implementar lógica condicional:
        - Si es 'Lesión': mostrar campos de biomecánica (zona, sistema, etc.).
        - Si es 'Enfermedad': mostrar campos de enfermedad (tipo_enfermedad).
    - Actualizar la visualización en `_render_seguimiento_activo` y `_render_historial`.
---
