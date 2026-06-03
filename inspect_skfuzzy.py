import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

u = np.arange(0, 11, 1)
a = ctrl.Antecedent(u, 'a')
c = ctrl.Consequent(u, 'c')
a['low'] = fuzz.trimf(u, [0, 0, 5])
c['low'] = fuzz.trimf(u, [0, 0, 5])
r = ctrl.Rule(a['low'], c['low'])
s = ctrl.ControlSystem([r])
sim = ctrl.ControlSystemSimulation(s)
sim.input['a'] = 5

print(f"sim.inputs(): {sim.inputs()}")
print(f"sim.input._get_inputs(): {sim.input._get_inputs()}")
