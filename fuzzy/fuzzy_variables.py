import skfuzzy as fuzz
import numpy as np
from skfuzzy import control as ctrl

def get_variables():
    wellness = ctrl.Antecedent(np.arange(0, 1.1, 0.1), 'wellness')
    carga_integrada = ctrl.Antecedent(np.arange(0, 201, 1), 'carga_integrada')
    fatiga = ctrl.Consequent(np.arange(0, 101, 1), 'fatiga')

    wellness['bajo'] = fuzz.trimf(wellness.universe, [0, 0, 0.5])
    wellness['medio'] = fuzz.trimf(wellness.universe, [0, 0.5, 1])
    wellness['alto'] = fuzz.trimf(wellness.universe, [0.5, 1, 1])

    carga_integrada['baja'] = fuzz.trimf(carga_integrada.universe, [0, 0, 100])
    carga_integrada['media'] = fuzz.trimf(carga_integrada.universe, [50, 100, 150])
    carga_integrada['alta'] = fuzz.trimf(carga_integrada.universe, [100, 200, 200])

    fatiga['baja'] = fuzz.trimf(fatiga.universe, [0, 0, 50])
    fatiga['media'] = fuzz.trimf(fatiga.universe, [25, 50, 75])
    fatiga['alta'] = fuzz.trimf(fatiga.universe, [50, 100, 100])

    return wellness, carga_integrada, fatiga