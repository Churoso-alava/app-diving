import pytest
import pandas as pd
from datetime import date
from core.services import (
    registrar_lesion_servicio,
    obtener_lesiones_activas_servicio,
    obtener_historial_lesiones_servicio,
    actualizar_estado_lesion_servicio
)
# We will mock the database functions in future tasks.

from unittest.mock import patch, MagicMock
from core.schemas import InjuryInput

@patch('core.services.insertar_lesion')
def test_registrar_lesion_servicio_success(mock_insertar):
    mock_insertar.return_value = (True, "OK")
    data = InjuryInput(
        atleta="Test Athlete", 
        fecha_lesion="2026-06-01", 
        zona_corporal="Hombro", 
        tipo="Aguda", 
        gravedad="Leve"
    )
    success, msg = registrar_lesion_servicio(data)
    assert success is True
    assert msg == "OK"
    mock_insertar.assert_called_once()
