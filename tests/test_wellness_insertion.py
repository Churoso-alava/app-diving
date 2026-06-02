"""
Este script prueba la función insertar_wellness de data/db.py.
Simula inserciones exitosas, duplicadas y con datos fuera de rango.
Requiere que las variables de entorno SUPABASE_URL y SUPABASE_KEY estén configuradas.
"""
import os
from datetime import date, timedelta
import sys

# Asegurarse de que el directorio actual esté en el path para importar data.db
# Esto es necesario si el script se ejecuta desde una ubicación diferente a la del proyecto
# En un entorno real, esto podría requerir un setup más robusto o ejecutar desde el directorio raíz.
# Para este caso, asumimos que el script se ejecutará desde la raíz del proyecto.
try:
    # Intentar importar directamente, asumiendo que data/db.py está accesible
    from data.db import insertar_wellness
    from core.services import cargar_atletas # Necesario para obtener un atleta válido
except ImportError as e:
    print(f"Error al importar módulos: {e}")
    print("Asegúrate de que 'data/db.py' y 'core/services.py' sean accesibles")
    print("y que las variables de entorno SUPABASE_URL y SUPABASE_KEY estén configuradas.")
    sys.exit(1)

# --- Configuración de prueba ---
# Obtener un atleta válido para las pruebas. Si no hay, usar un nombre genérico.
# En un escenario real, se obtendría dinámicamente o se usaría un atleta de test conocido.
try:
    atletas_activos = cargar_atletas()
    if atletas_activos:
        atleta_prueba = atletas_activos[0] # Usar el primer atleta activo encontrado
        print(f"Usando atleta '{atleta_prueba}' para las pruebas.")
    else:
        atleta_prueba = "Test Athlete"
        print(f"Advertencia: No se encontraron atletas activos. Usando nombre genérico '{atleta_prueba}'.")
        print("La inserción podría fallar si 'Test Athlete' no existe o no tiene permisos.")
except Exception as e:
    print(f"Error al cargar atletas: {e}. Usando nombre genérico '{atleta_prueba}'.")
    atleta_prueba = "Test Athlete"


fecha_hoy = date.today()
fecha_ayer = fecha_hoy - timedelta(days=1)
fecha_duplicada = fecha_hoy # Para probar la restricción de duplicado

# --- Datos de prueba ---
# Valores válidos para los sliders (1-7)
VAL_SUENO_OK = 5
VAL_FATIGA_OK = 4
VAL_ESTRES_OK = 3
VAL_DOLOR_OK = 2
VAL_HUMOR_OK = 6
NOTAS_OK = "Prueba de inserción exitosa."

# Valores fuera de rango para sliders
VAL_SUENO_INVALIDO = 8
VAL_FATIGA_INVALIDO = 0
VAL_ESTRES_INVALIDO = 10 # Cualquier valor fuera de 1-7
VAL_DOLOR_INVALIDO = -1
VAL_HUMOR_INVALIDO = 1

# --- Funciones de prueba ---
def ejecutar_prueba(descripcion: str, func, *args, **kwargs):
    print()
    print(f"--- Prueba: {descripcion} ---")
    try:
        success, message = func(*args, **kwargs)
        if success:
            print(f"✅ Éxito: {message}")
        else:
            print(f"❌ Fallo: {message}")
    except Exception as e:
        print(f"💥 Excepción inesperada: {e}")
        import traceback
        traceback.print_exc()

# --- Ejecución de pruebas ---
print(f"Iniciando pruebas para el atleta: '{atleta_prueba}' en la fecha: {fecha_hoy}")

# 1. Prueba de inserción exitosa
ejecutar_prueba(
    "Inserción válida exitosa",
    insertar_wellness,
    nombre=atleta_prueba,
    fecha=fecha_hoy,
    sueno=VAL_SUENO_OK,
    fatiga_hooper=VAL_FATIGA_OK,
    estres=VAL_ESTRES_OK,
    dolor=VAL_DOLOR_OK,
    humor=VAL_HUMOR_OK,
    notas=NOTAS_OK
)

# Esperar un momento para que las bases de datos se sincronicen si es necesario,
# aunque Supabase RPCs suelen ser síncronas.
# import time
# time.sleep(2)

# 2. Prueba de inserción duplicada (mismo atleta, misma fecha)
# NOTA: Esta prueba asume que la inserción anterior tuvo éxito.
#       Si la inserción anterior falló, esta podría ser la primera de su tipo.
ejecutar_prueba(
    "Intento de inserción duplicada (mismo atleta, misma fecha)",
    insertar_wellness,
    nombre=atleta_prueba,
    fecha=fecha_duplicada, # Mismo día que la prueba exitosa
    sueno=4, fatiga_hooper=4, estres=4, dolor=4, humor=4,
    notas="Esta inserción debería fallar por ser duplicada."
)

# 3. Prueba de inserción con valores fuera de rango (sueno)
ejecutar_prueba(
    "Intento de inserción con valor de sueño inválido",
    insertar_wellness,
    nombre=atleta_prueba,
    fecha=fecha_hoy, # Fecha válida
    sueno=VAL_SUENO_INVALIDO, # Valor fuera de rango (8)
    fatiga_hooper=VAL_FATIGA_OK,
    estres=VAL_ESTRES_OK,
    dolor=VAL_DOLOR_OK,
    humor=VAL_HUMOR_OK,
    notas="Esta inserción debería fallar por valor de sueño inválido."
)

# 4. Prueba de inserción con valores fuera de rango (fatiga)
ejecutar_prueba(
    "Intento de inserción con valor de fatiga inválido",
    insertar_wellness,
    nombre=atleta_prueba,
    fecha=fecha_hoy,
    sueno=VAL_SUENO_OK,
    fatiga_hooper=VAL_FATIGA_INVALIDO, # Valor fuera de rango (0)
    estres=VAL_ESTRES_OK,
    dolor=VAL_DOLOR_OK,
    humor=VAL_HUMOR_OK,
    notas="Esta inserción debería fallar por valor de fatiga inválido."
)

# Opcional: Prueba con otro atleta si se conoce uno y se quiere variar
# if len(atletas_activos) > 1:
#     otro_atleta = atletas_activos[1]
#     ejecutar_prueba(
#         f"Inserción válida para otro atleta: {otro_atleta}",
#         insertar_wellness,
#         nombre=otro_atleta,
#         fecha=fecha_hoy,
#         sueno=5, fatiga_hooper=4, estres=3, dolor=2, humor=6,
#         notas="Prueba para otro atleta."
#     )


print()
print()
print()
print()
print()
print("--- Pruebas completadas ---")
print("Revisa los resultados para verificar el comportamiento esperado.")
print("Un fallo en 'Inserción válida exitosa' indica un problema general.")
print("Un fallo en 'Inserción duplicada' indica que la restricción única funciona.")
print("Un fallo en 'valores inválidos' indica que la validación de rango funciona.")
print("NOTA: Para que las pruebas de duplicados funcionen, la primera inserción")
print("debe haber sido exitosa y el RPC 'insertar_wellness_hooper' debe tener")
print("una restricción UNIQUE en (nombre, fecha).")

# Añadir un pequeño delay para que las operaciones de DB puedan propagarse si es necesario.
# Esto es más una precaución que algo que suele ser necesario para RPCs síncronos.
import time
time.sleep(1)
