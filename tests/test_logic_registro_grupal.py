import pandas as pd
import pytest

def test_filtering_logic_vmp():
    # Simulamos el DataFrame del editor de VMP
    data = {
        "Atleta": ["A1", "A2", "A3", "A4"],
        "VMP Hoy": [0.0, 1.5, 0.0, 2.0]
    }
    edited_df = pd.DataFrame(data)
    
    # Lógica de app.py
    sesiones = []
    for _, row in edited_df.iterrows():
        if row["VMP Hoy"] > 0:
            sesiones.append({
                "nombre": row["Atleta"],
                "vmp_hoy": float(row["VMP Hoy"])
            })
    
    # Verificaciones
    assert len(sesiones) == 2
    assert sesiones[0]["nombre"] == "A2"
    assert sesiones[0]["vmp_hoy"] == 1.5
    assert sesiones[1]["nombre"] == "A4"
    assert sesiones[1]["vmp_hoy"] == 2.0
    print("Test VMP pasado correctamente")

def test_filtering_logic_carga():
    # Simulamos el DataFrame del editor de Carga
    data = {
        "Atleta": ["A1", "A2", "A3", "A4"],
        "RPE (1-10)": [0, 5, 0, 8],
        "Duración (min)": [240, 60, 240, 45]
    }
    edited_carga_df = pd.DataFrame(data)
    
    # Lógica de app.py
    cargas = []
    for _, row in edited_carga_df.iterrows():
        if row["RPE (1-10)"] > 0:
            cargas.append({
                "nombre": row["Atleta"],
                "carga_subjetiva": int(row["RPE (1-10)"]),
                "duracion_min": int(row["Duración (min)"])
            })
            
    # Verificaciones
    assert len(cargas) == 2
    assert cargas[0]["nombre"] == "A2"
    assert cargas[0]["carga_subjetiva"] == 5
    assert cargas[0]["duracion_min"] == 60
    assert cargas[1]["nombre"] == "A4"
    assert cargas[1]["carga_subjetiva"] == 8
    assert cargas[1]["duracion_min"] == 45
    print("Test Carga pasado correctamente")

if __name__ == "__main__":
    test_filtering_logic_vmp()
    test_filtering_logic_carga()
