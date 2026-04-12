-- Migration: create_wellness_table
-- Run in Supabase SQL Editor or via: supabase db push
-- Branch: feat/wellness-tracking

CREATE TABLE IF NOT EXISTS public.wellness (
    id              bigserial PRIMARY KEY,
    nombre          text NOT NULL CHECK (char_length(nombre) > 0),
    fecha           date NOT NULL,
    sueno           smallint NOT NULL CHECK (sueno BETWEEN 1 AND 7),
    fatiga_hooper   smallint NOT NULL CHECK (fatiga_hooper BETWEEN 1 AND 7),
    estres          smallint NOT NULL CHECK (estres BETWEEN 1 AND 7),
    dolor           smallint NOT NULL CHECK (dolor BETWEEN 1 AND 7),
    humor           smallint NOT NULL CHECK (humor BETWEEN 1 AND 7),
    -- w_norm calculado en app (evita trigger) y almacenado para queries analíticas
    w_norm          numeric(5,4) NOT NULL
                    GENERATED ALWAYS AS (
                        ROUND(
                            ((7.0 - sueno)       / 6.0 +
                             (7.0 - fatiga_hooper)/ 6.0 +
                             (7.0 - estres)       / 6.0 +
                             (7.0 - dolor)        / 6.0 +
                             (humor - 1.0)        / 6.0) / 5.0
                        , 4)
                    ) STORED,
    notas           text DEFAULT '',
    created_at      timestamptz DEFAULT now() NOT NULL
);

-- Índices para RLS y queries frecuentes
CREATE INDEX IF NOT EXISTS idx_wellness_nombre ON public.wellness (nombre);
CREATE INDEX IF NOT EXISTS idx_wellness_fecha  ON public.wellness (fecha DESC);
CREATE INDEX IF NOT EXISTS idx_wellness_nombre_fecha ON public.wellness (nombre, fecha DESC);

-- RLS obligatorio (Clark-Wilson: CDI protegido)
ALTER TABLE public.wellness ENABLE ROW LEVEL SECURITY;

-- Política para service_role (backend — bypasa RLS por diseño Supabase)
-- La anon key queda bloqueada por defecto al no tener política SELECT abierta.
-- Si se necesita acceso autenticado por usuario, agregar política JWT aquí.

-- Comentarios de columna (documentación inline)
COMMENT ON TABLE  public.wellness IS 'Cuestionario de Wellness Hooper Modificado (5 ítems Likert 1-7)';
COMMENT ON COLUMN public.wellness.sueno   IS '1=óptimo, 7=pésimo (escala inversa)';
COMMENT ON COLUMN public.wellness.humor   IS '7=óptimo, 1=pésimo (escala directa)';
COMMENT ON COLUMN public.wellness.w_norm  IS 'Índice normalizado [0,1] calculado. 1=óptimo.';
