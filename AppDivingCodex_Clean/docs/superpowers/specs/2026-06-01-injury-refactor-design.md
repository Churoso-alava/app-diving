# Diseño: Módulo Unificado de Eventos Médicos (Lesiones y Enfermedades)

**Fecha:** 2026-06-01
**Goal:** Optimizar el registro de eventos médicos unificando lesiones y enfermedades, eliminando redundancias en fechas y mejorando la claridad de los campos.

## 1. Modelo de Datos Unificado (`eventos_medicos`)

### Campos Comunes
- `atleta` (str)
- `tipo_evento` (Enum: 'Lesión', 'Enfermedad')
- `fecha_inicio` (date) — *Fecha del suceso o inicio de síntomas.*
- `gravedad` (Enum: 'Leve', 'Moderada', 'Grave')
- `estado` (Enum: 'Activa', 'Recuperación', 'Alta')
- `notas` (str)

**Fechas de Proceso (Optimizado):**
- `fecha_alta_medica` (date, opcional)
- `fecha_vuelta_entrenamiento` (date, opcional)
- `fecha_vuelta_juego` (date, opcional)

### Campos Específicos

#### Si es 'Lesión'
- `zona_corporal` (Enum/str): [Hombro, Codo, Muñeca, Mano, Cuello, Columna Dorsal, Columna Lumbar, Cadera, Pelvis, Muslo, Rodilla, Pierna, Tobillo, Pie, Otro]
- `tipo_tejido` (Enum: Músculo, Tendón, Ligamento, Otro)
- `mecanismo` (Enum: Trauma o Golpe, Sobreuso)
- `recurrencia` (Enum: Nueva, Recurrencia)

#### Si es 'Enfermedad'
- `tipo_enfermedad` (Enum: Respiratoria, Digestiva, Infecciosa, Otra)
- `es_contagiosa` (bool)

---

## 2. Plan de Implementación (Resumen)

1. **DB:** Ajustar tabla `lesiones` (o crear `eventos_medicos`) con los nuevos campos.
2. **Core:** Actualizar `schemas.py` y `services.py` con el nuevo modelo unificado.
3. **UI:** Refactorizar `components/tab_lesiones.py` para usar `tipo_evento` y mostrar campos condicionales.
4. **Testing:** Actualizar tests de validación e integración.

---
*Diseño final aprobado por el usuario.*
