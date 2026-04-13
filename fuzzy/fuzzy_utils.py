def normalizar_wellness(valor_1_5):
    """Convierte escala 1-5 a escala 0-1."""
    return (valor_1_5 - 1) / 4

def calcular_carga_integrada(vmp, wellness_norm):
    """Calcula la CI según el plan v4.2."""
    return vmp * (1 + (1 - wellness_norm))

if __name__ == "__main__":
    print("🛠️ Utilidades listas para el motor.")
    