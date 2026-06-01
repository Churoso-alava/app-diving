-- Migration: add_carga_entrenamiento_sesiones
-- Fecha: 2026-06-01
-- Descripción: Agrega columnas de carga de entrenamiento subjetiva (RPE 1-10)
--              y duración (minutos) a la tabla sesiones_vmp.
-- Ejecutar en Supabase Dashboard > SQL Editor

ALTER TABLE sesiones_vmp
  ADD COLUMN IF NOT EXISTS carga_subjetiva INTEGER
    CHECK (carga_subjetiva BETWEEN 1 AND 10),
  ADD COLUMN IF NOT EXISTS duracion_min    INTEGER
    CHECK (duracion_min > 0);

-- Índice útil para consultas por atleta+fecha
CREATE INDEX IF NOT EXISTS idx_sesiones_vmp_nombre_fecha
  ON sesiones_vmp (nombre, fecha);

-- Migration: add_columnas_lesiones
-- Fecha: 2026-06-01
-- Descripción: Agrega campos de RTP y biomecánica a la tabla lesiones.

ALTER TABLE lesiones
  ADD COLUMN IF NOT EXISTS tipo_tejido TEXT,
  ADD COLUMN IF NOT EXISTS mecanismo TEXT,
  ADD COLUMN IF NOT EXISTS recurrencia TEXT,
  ADD COLUMN IF NOT EXISTS mecanismo_contacto BOOLEAN DEFAULT FALSE,
  ADD COLUMN IF NOT EXISTS fecha_evento DATE,
  ADD COLUMN IF NOT EXISTS fecha_alta_medica DATE,
  ADD COLUMN IF NOT EXISTS fecha_rtt DATE,
  ADD COLUMN IF NOT EXISTS fecha_rtp DATE;
