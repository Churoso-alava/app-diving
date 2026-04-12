-- Migration: create_cargas_sesion
-- Ejecutar en Supabase SQL Editor antes del deploy de app.py v4.2

CREATE TABLE IF NOT EXISTS public.cargas_sesion (
    id              bigserial PRIMARY KEY,
    nombre          text        NOT NULL CHECK (char_length(nombre) > 0),
    fecha           date        NOT NULL,
    n_clavados      smallint    NOT NULL CHECK (n_clavados > 0),
    l_bruta         numeric(10,4) NOT NULL CHECK (l_bruta >= 0),
    l_norm          numeric(7,4)  NOT NULL CHECK (l_norm  BETWEEN 0 AND 100),
    w_norm          numeric(5,4)  NOT NULL CHECK (w_norm  BETWEEN 0 AND 1),
    ci              numeric(8,4)  NOT NULL CHECK (ci      BETWEEN 0 AND 200),
    zona_dominante  text        NOT NULL,
    notas           text        DEFAULT '',
    created_at      timestamptz DEFAULT now() NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_cargas_nombre       ON public.cargas_sesion (nombre);
CREATE INDEX IF NOT EXISTS idx_cargas_fecha        ON public.cargas_sesion (fecha DESC);
CREATE INDEX IF NOT EXISTS idx_cargas_nombre_fecha ON public.cargas_sesion (nombre, fecha DESC);

ALTER TABLE public.cargas_sesion ENABLE ROW LEVEL SECURITY;

COMMENT ON TABLE  public.cargas_sesion IS 'Carga de sesión de clavados — CI calculada (Pandey 2022)';
COMMENT ON COLUMN public.cargas_sesion.l_norm IS '[0,100] normalizada sobre L_MAX_REFERENCIA=500';
COMMENT ON COLUMN public.cargas_sesion.w_norm IS '[0,1] Wellness Hooper normalizado';
COMMENT ON COLUMN public.cargas_sesion.ci     IS '[0,200] CI = L_norm × (2 − W_norm)';
