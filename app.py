def cargar_sesiones() -> pd.DataFrame:
    """Carga todas las sesiones visibles para el usuario autenticado (RLS)."""
    try:
        client = get_client()
        resp = client.table("sesiones").select("*").execute()
        
        if not resp.data:
            log.warning("cargar_sesiones: tabla vacía.")
            return pd.DataFrame(columns=["id", "Nombre", "Fecha", "VMP_Hoy", "notas"])
        
        df = pd.DataFrame(resp.data)
        df = df.rename(columns={
            "nombre": "Nombre",
            "fecha": "Fecha",
            "vmp_hoy": "VMP_Hoy",
        })
        df["Fecha"] = pd.to_datetime(df["Fecha"])
        df["VMP_Hoy"] = pd.to_numeric(df["VMP_Hoy"], errors="coerce")
        
        log.info("cargar_sesiones: %d registros.", len(df))
        return df
    except Exception as exc:
        log.error("cargar_sesiones falló: %s", exc)
        raise


def cargar_atletas() -> list[str]:
    """Retorna lista de nombres únicos de atletas."""
    try:
        client = get_client()
        resp = client.table("atletas").select("nombre").order("nombre").execute()
        nombres = [r["nombre"] for r in resp.data]
        log.info("cargar_atletas: %d atletas.", len(nombres))
        return nombres
    except Exception as exc:
        log.error("cargar_atletas falló: %s", exc)
        raise
