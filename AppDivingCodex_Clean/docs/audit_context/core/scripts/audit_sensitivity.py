import sys
import os
sys.path.append(os.getcwd())
import numpy as np
from fuzzy_engine import construir_motor_fuzzy

def run_simulation():
    vars_tuple, simulador = construir_motor_fuzzy()
    
    # Casos de prueba: (vmp, acwr, delta, zmeso, ba, b28, well, ci)
    test_cases = {
        "Crítico Extremo": {
            "vmp_hoy": 0.2, "acwr": 0.6, "delta_pct": 30.0, "z_meso": -3.0,
            "beta_aguda": -0.2, "beta_28": -0.2, "wellness_norm": 0.1, "carga_integrada_plan": 180.0
        },
        "Adaptación Perfecta": {
            "vmp_hoy": 2.2, "acwr": 1.0, "delta_pct": -10.0, "z_meso": 2.0,
            "beta_aguda": 0.1, "beta_28": 0.1, "wellness_norm": 0.9, "carga_integrada_plan": 60.0
        },
        "Conflicto (VMP Alta/ACWR Excesivo)": {
            "vmp_hoy": 2.0, "acwr": 1.7, "delta_pct": 5.0, "z_meso": -2.0,
            "beta_aguda": -0.05, "beta_28": -0.05, "wellness_norm": 0.5, "carga_integrada_plan": 120.0
        }
    }

    print(f"{'Caso':<35} | {'Índice Fatiga':<15}")
    print("-" * 55)

    for name, inputs in test_cases.items():
        for key, val in inputs.items():
            simulador.input[key] = val
        simulador.compute()
        print(f"{name:<35} | {simulador.output['fatiga']:<15.2f}")

if __name__ == "__main__":
    run_simulation()
