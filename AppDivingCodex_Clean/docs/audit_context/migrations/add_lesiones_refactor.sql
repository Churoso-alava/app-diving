-- Agregar campos de clasificación biomecánica y hitos RTP
ALTER TABLE lesiones 
ADD COLUMN IF NOT EXISTS tipo_tejido TEXT,
ADD COLUMN IF NOT EXISTS mecanismo TEXT,
ADD COLUMN IF NOT EXISTS recurrencia TEXT,
ADD COLUMN IF NOT EXISTS mecanismo_contacto BOOLEAN DEFAULT FALSE,
ADD COLUMN IF NOT EXISTS fecha_evento DATE,
ADD COLUMN IF NOT EXISTS fecha_alta_medica DATE,
ADD COLUMN IF NOT EXISTS fecha_rtt DATE,
ADD COLUMN IF NOT EXISTS fecha_rtp DATE;
