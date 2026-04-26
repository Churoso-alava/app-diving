
import pytest
import numpy as np
from core.fuzzy_variables import get_variables

def test_get_variables():
    wellness_var, carga_integrada_var, fatiga_var = get_variables()
    
    # Check if they are the correct types (skfuzzy.control.Antecedent/Consequent)
    from skfuzzy import control as ctrl
    assert isinstance(wellness_var, ctrl.Antecedent)
    assert isinstance(carga_integrada_var, ctrl.Antecedent)
    assert isinstance(fatiga_var, ctrl.Consequent)
    
    # Check names
    assert wellness_var.label == 'wellness_score'
    assert carga_integrada_var.label == 'integrated_load'
    assert fatiga_var.label == 'fatiga'
    
    # Check universes
    assert np.array_equal(wellness_var.universe, np.arange(1, 7.1, 1))
    assert np.array_equal(carga_integrada_var.universe, np.arange(0, 201, 1))
    assert np.array_equal(fatiga_var.universe, np.arange(0, 101, 1))
