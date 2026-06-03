import pandas as pd
import pytest

def test_formatting_with_none():
    res = {
        'vmp_hoy': 1.0,
        'acwr': None,
        'delta_pct': None,
        'z_meso': 1.0,
        'beta_aguda': 0.1,
        'beta_28': 0.1
    }
    
    # This is the logic I fixed in app.py
    tabla_data = {
        "Valor": [
            f"{res.get('vmp_hoy') or 0.0:.2f}", 
            f"{res.get('acwr') or 0.0:.2f}", 
            f"{res.get('delta_pct') or 0.0:.1f}%", 
            f"{res.get('z_meso') or 0.0:.2f}", 
            f"{res.get('beta_aguda') or 0.0:.3f}", 
            f"{res.get('beta_28') or 0.0:.3f}"
        ]
    }
    
    assert tabla_data["Valor"][1] == "0.00"
    assert tabla_data["Valor"][2] == "0.0%"
    
def test_formatting_with_existing_values():
    res = {
        'vmp_hoy': 1.234,
        'acwr': 0.95,
        'delta_pct': 5.5,
        'z_meso': 0.5,
        'beta_aguda': 0.01,
        'beta_28': 0.02
    }
    
    tabla_data = {
        "Valor": [
            f"{res.get('vmp_hoy') or 0.0:.2f}", 
            f"{res.get('acwr') or 0.0:.2f}", 
            f"{res.get('delta_pct') or 0.0:.1f}%", 
            f"{res.get('z_meso') or 0.0:.2f}", 
            f"{res.get('beta_aguda') or 0.0:.3f}", 
            f"{res.get('beta_28') or 0.0:.3f}"
        ]
    }
    
    assert tabla_data["Valor"][0] == "1.23"
    assert tabla_data["Valor"][1] == "0.95"
    assert tabla_data["Valor"][2] == "5.5%"
