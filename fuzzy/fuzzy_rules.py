import skfuzzy.control as ctrl

def definir_reglas(wellness, ci, fatiga):
    reglas = [
        ctrl.Rule(wellness['bajo'] & ci['alta'], fatiga['alta']),
        ctrl.Rule(wellness['medio'] & ci['media'], fatiga['media']),
        ctrl.Rule(wellness['alto'] & ci['baja'], fatiga['baja']),
    ]
    return reglas