from fuzzy.fuzzy_variables import get_variables

def test_configuracion_correcta():
    wellness, ci, fatiga = get_variables()
    
    # Verificamos que los nombres sean los correctos
    assert wellness.label == 'wellness'
    assert ci.label == 'carga_integrada'
    
    # Verificamos que tengan las etiquetas que definimos
    assert 'medio' in wellness.terms
    assert 'alta' in ci.terms
    print("\n✅ ¡Test aprobado! Las variables están perfectas.")