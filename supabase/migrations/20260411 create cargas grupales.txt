-- Migration: create_cargas_grupales
-- Ejecutar en Supabase SQL Editor o via: supabase db push
-- Fecha: 2026-04-11
-- 
-- Tablas para sesiones de carga grupal (clavados).
-- Una sesión grupal = ejercicios realizados por múltiples atletas en la misma fecha.

-- Tabla principal: sesión grupal
CREATE TABLE IF NOT EXISTS public.cargas_grupales (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    fecha           DATE        NOT NULL,
    tipo_plataforma TEXT        NOT NULL CHECK (tipo_plataforma IN ('trampolín', 'plataforma')),
    altura_salto    NUMERIC(4,1) NOT NULL CHECK (altura_salto > 0),
    n_saltos        INTEGER     NOT NULL CHECK (n_saltos >= 0),
    tipo_caida      TEXT        NOT NULL CHECK (tipo_caida IN ('pie', 'mano')),
    notas           TEXT        DEFAULT '',
    created_at      TIMESTAMPTZ DEFAULT NOW() NOT NULL
);

-- Tabla de enlace: qué atletas participaron en qué sesión grupal
CREATE TABLE IF NOT EXISTS public.cargas_grupales_atletas (
    carga_grupal_id UUID NOT NULL REFERENCES public.cargas_grupales(id) ON DELETE CASCADE,
    nombre          TEXT NOT NULL,
    PRIMARY KEY (carga_grupal_id, nombre)
);

-- Índices para queries frecuentes
CREATE INDEX IF NOT EXISTS idx_cargas_grupales_fecha 
    ON public.cargas_grupales (fecha DESC);

CREATE INDEX IF NOT EXISTS idx_cargas_grupales_atletas_nombre 
    ON public.cargas_grupales_atletas (nombre);

-- Row-Level Security (RLS)
ALTER TABLE public.cargas_grupales ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.cargas_grupales_atletas ENABLE ROW LEVEL SECURITY;

-- Política para service_role (backend — bypasa RLS por diseño Supabase)
CREATE POLICY "service_role_full_cargas_grupales" ON public.cargas_grupales
    FOR ALL USING (auth.role() = 'service_role');

CREATE POLICY "service_role_full_cargas_grupales_atletas" ON public.cargas_grupales_atletas
    FOR ALL USING (auth.role() = 'service_role');

-- Comentarios de documentación
COMMENT ON TABLE public.cargas_grupales 
    IS 'Sesiones de carga grupal (ejercicios de clavados ejecutados por múltiples atletas)';

COMMENT ON TABLE public.cargas_grupales_atletas 
    IS 'Enlace: atletas que participaron en cada sesión grupal';

COMMENT ON COLUMN public.cargas_grupales.tipo_plataforma 
    IS 'Tipo de plataforma: trampolín o plataforma rígida';

COMMENT ON COLUMN public.cargas_grupales.altura_salto 
    IS 'Altura de la plataforma o trampolín en metros';

COMMENT ON COLUMN public.cargas_grupales.n_saltos 
    IS 'Número total de saltos/clavados ejecutados en la sesión';

COMMENT ON COLUMN public.cargas_grupales.tipo_caida 
    IS 'Tipo de caída al agua: pie o mano';
