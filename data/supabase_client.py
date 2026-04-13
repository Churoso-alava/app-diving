from supabase import create_client, Client

# --- CONFIGURACIÓN ---
# Asegúrate de que estas sean tus credenciales correctas
URL_PROYECTO = "https://hbmoztlcbrfcayoiqmpy.supabase.co"
KEY_PROYECTO = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImhibW96dGxjYnJmY2F5b2lxbXB5Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NzUzMzI4ODEsImV4cCI6MjA5MDkwODg4MX0.FOlsrQXdBahHg1DsgvVJz43pDWXzRmXAaetPW_kVlH4"

def conectar_db():
    """Establece la conexión con Supabase."""
    try:
        supabase: Client = create_client(URL_PROYECTO, KEY_PROYECTO)
        return supabase
    except Exception as e:
        print(f"❌ Error de conexión: {e}")
        return None

# --- FUNCIONES DE CLAVADISTAS ---

def obtener_clavadistas():
    """Trae la lista de clavadistas registrados desde la tabla 'clavadistas'."""
    supabase = conectar_db() # Aquí ya no dará error porque está definida arriba
    if supabase:
        try:
            # Seleccionamos todos los campos de la tabla clavadistas
            respuesta = supabase.table("clavadistas").select("*").execute()
            return respuesta.data
        except Exception as e:
            print(f"❌ Error al obtener clavadistas: {e}")
            return []
    return []

def guardar_fatiga(clavadista_id, wellness, vmp, fatiga):
    """Guarda un registro de fatiga en la tabla 'registros_fatiga'."""
    supabase = conectar_db()
    if supabase:
        try:
            data = {
                "clavadista_id": clavadista_id,
                "wellness_puntuacion": wellness,
                "vmp_real": vmp,
                "fatiga_calculada": fatiga
            }
            resultado = supabase.table("registros_fatiga").insert(data).execute()
            return resultado
        except Exception as e:
            print(f"❌ Error al guardar registro: {e}")
            return None
    return None

# --- PRUEBA DE CONEXIÓN ---
if __name__ == "__main__":
    cliente = conectar_db()
    if cliente:
        print("✅ Conexión establecida.")
        datos = obtener_clavadistas()
        print(f"Atletas encontrados: {len(datos)}")
    else:
        print("❌ No se pudo conectar.")