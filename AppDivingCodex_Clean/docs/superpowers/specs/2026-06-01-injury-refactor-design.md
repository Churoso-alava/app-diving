# Diseño de Refactorización: Módulo de Gestión de Lesiones (Avanzado)

**Fecha:** 2026-06-01
**Goal:** Evolucionar el módulo de lesiones para registrar hitos biológicos del retorno a la competición (RTP) y una clasificación biomecánica más precisa.

## 1. Arquitectura y Enfoque
La refactorización se basa en extender el modelo actual de datos para soportar un seguimiento clínico más detallado sin romper la persistencia existente. Se utilizarán `Enum`s de Python para tipar fuertemente la clasificación y se actualizará el esquema de la base de datos mediante comandos `ALTER TABLE`.

## 2. Esquema de Datos (Base de Datos)
La tabla `lesiones` se ampliará con:
- `tipo_tejido` (TEXT, CHECK constraint)
- `mecanismo` (TEXT, CHECK constraint)
- `recurrencia` (TEXT, CHECK constraint)
- `mecanismo_contacto` (BOOLEAN, default FALSE)
- `fecha_evento` (DATE)
- `fecha_alta_medica` (DATE)
- `fecha_rtt` (DATE)
- `fecha_rtp` (DATE)

## 3. Modelo de Datos (Python - `core/schemas.py`)
Se introducirán las siguientes estructuras:
- `TipoTejido` (Enum: musculo, tendon, ligamento, otro)
- `MecanismoInicio` (Enum: aguda, sobreuso)
- `HistorialRecurrencia` (Enum: nueva, recurrencia)
- Actualización de `InjuryInput` para incorporar estos nuevos campos y validaciones en `__post_init__`.

## 4. UI Layout (Formulario Único)
La UI en `components/tab_lesiones.py` se reorganizará en un layout de 3 columnas dentro de un formulario único para mejorar la UX:

*   **Columna 1 (General):** Atleta, Fecha del Evento, Zona Corporal.
*   **Columna 2 (Clasificación):** Tipo de Tejido, Mecanismo de Inicio, Recurrencia, Checkbox de contacto.
*   **Columna 3 (Hitos RTP):** Fecha Alta Médica, Fecha RTT, Fecha RTP.

## 5. Manejo de Errores y Testing
- Se mantendrá la validación centralizada en `InjuryInput.__post_init__`.
- Se añadirán tests unitarios para verificar la validación de los nuevos campos y la serialización a diccionario.

---
## 6. Self-Review
- [x] Placeholder scan: No hay TBDs.
- [x] Consistencia interna: Los nombres de los Enums coinciden con el SQL propuesto.
- [x] Scope: El refactor es acotado al módulo de lesiones.
- [x] Ambiguity: Los hitos temporales (RTT/RTP) están definidos explícitamente en el plan.

*Documento creado y listo para revisión.*
