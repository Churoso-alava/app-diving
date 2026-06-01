import pytest
import pandas as pd
from datetime import date
from unittest.mock import MagicMock, patch
from data.db import insertar_lesion, cargar_lesiones_activas, actualizar_estado_lesion

@patch("data.db.get_client")
def test_injury_flow_integration(mock_get_client):
    # Setup mock
    mock_client = MagicMock()
    mock_get_client.return_value = mock_client
    
    # 1. Register a lesion
    atleta = "Test Athlete"
    fecha = date(2026, 6, 1)
    
    # Mock insert response
    mock_client.table.return_value.insert.return_value.execute.return_value = MagicMock(data=[{}])
    
    ok, msg = insertar_lesion(atleta=atleta, fecha_lesion=fecha, zona_corporal="Rodilla", tipo="Aguda", gravedad="Leve")
    assert ok, msg
    
    # 2. Query active lesions
    # Mock return value for cargar_lesiones_activas
    mock_client.table.return_value.select.return_value.in_.return_value.order.return_value.execute.return_value = MagicMock(
        data=[{"id": "lesion_1", "atleta": atleta, "estado": "Activa"}]
    )
    
    df = cargar_lesiones_activas()
    assert len(df) == 1
    assert df.iloc[0]["atleta"] == atleta
    assert df.iloc[0]["estado"] == "Activa"
    
    # 3. Update lesion status
    # Mock update response
    mock_client.table.return_value.update.return_value.eq.return_value.execute.return_value = MagicMock(data=[{}])
    
    ok, msg = actualizar_estado_lesion("lesion_1", "Recuperación")
    assert ok, msg
    
    # Verify update called
    mock_client.table.assert_called_with("lesiones")
