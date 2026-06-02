import pytest
import pandas as pd
from core.wellness import calcular_wellness, cronbach_alpha_wellness

def test_sueno_deficiente_pesa_mas_que_dolor_leve():
    """
    Sueño muy malo + dolor leve debe dar peor wellness que sueño perfecto + dolor severo.
    El sueño tiene mayor peso predictivo de rendimiento (Saw et al. 2016).
    """
    w_sueno_malo = calcular_wellness(sueno=7, fatiga=2, estres=2, dolor=2, humor=5)
    w_dolor_malo  = calcular_wellness(sueno=1, fatiga=2, estres=2, dolor=7, humor=5)
    assert w_sueno_malo < w_dolor_malo, (
        f"Sueño malo (w={w_sueno_malo:.3f}) debe ser < dolor malo (w={w_dolor_malo:.3f})"
    )

def test_extremos_en_rango_cero_uno():
    assert calcular_wellness(1, 1, 1, 1, 7) == pytest.approx(1.0, abs=0.001)
    assert calcular_wellness(7, 7, 7, 7, 1) == pytest.approx(0.0, abs=0.001)

def test_valores_invalidos_lanzan_error():
    with pytest.raises(ValueError):
        calcular_wellness(sueno=0, fatiga=4, estres=4, dolor=4, humor=4)
    with pytest.raises(ValueError):
        calcular_wellness(sueno=4, fatiga=8, estres=4, dolor=4, humor=4)

def test_cronbach_alpha_muestra_coherente():
    """
    Con 5 ítems que varían en la misma dirección (coherentes),
    alpha debe ser > 0.70.
    """
    import pandas as pd
    # Items: sueño, fatiga, estrés, dolor (1=bueno, 7=malo) + humor (7=bueno, 1=malo)
    # Para ser coherentes, humor debe bajar cuando los otros suben.
    # Convertimos humor a 1=malo, 7=bueno (inverso del original) para el cálculo
    datos = pd.DataFrame({
        "sueno":  [1, 2, 6, 7, 3, 5, 2, 6],
        "fatiga": [1, 2, 5, 7, 3, 5, 2, 6],
        "estres": [2, 2, 5, 6, 4, 4, 3, 5],
        "dolor":  [1, 3, 4, 7, 3, 4, 2, 5],
        "humor":  [7, 6, 3, 1, 5, 3, 6, 2], # 7=bueno, 1=malo
    })
    # Revertir humor para que sea coherente con los otros (7=malo, 1=bueno)
    datos_coherentes = datos.copy()
    datos_coherentes["humor"] = 8 - datos["humor"]
    
    alpha = cronbach_alpha_wellness(datos_coherentes)
    assert alpha >= 0.70, f"Alpha={alpha:.3f} debe ser >=0.70 para items coherentes"

def test_cronbach_alpha_muestra_incoherente_es_bajo():
    """Ítems aleatorios sin relación → alpha bajo."""
    import numpy as np
    np.random.seed(7)
    datos = pd.DataFrame({
        col: np.random.randint(1, 8, 20)
        for col in ["sueno", "fatiga", "estres", "dolor", "humor"]
    })
    alpha = cronbach_alpha_wellness(datos)
    assert alpha < 0.70, f"Alpha={alpha:.3f} debe ser <0.70 para items incoherentes"
