# tests/test_injury_db.py
import pytest
from datetime import date
from unittest.mock import MagicMock, patch
import pandas as pd

class TestInsertarLesion:
    def test_atleta_vacio_rechazado_antes_de_db(self):
        from data.db import insertar_lesion
        ok, msg = insertar_lesion(atleta="", fecha_lesion=date(2026, 1, 15), zona_corporal="Rodilla", tipo="Aguda", gravedad="Leve")
        assert not ok
        assert "atleta" in msg.lower()

    @patch("data.db.get_client")
    def test_insercion_exitosa(self, mock_get_client):
        from data.db import insertar_lesion
        mock_client = MagicMock()
        mock_client.table.return_value.insert.return_value.execute.return_value = MagicMock(data=[{}])
        mock_get_client.return_value = mock_client
        ok, msg = insertar_lesion(atleta="Carlos", fecha_lesion=date(2026, 1, 15), zona_corporal="Hombro", tipo="Sobreuso", gravedad="Moderada")
        assert ok
        mock_client.table.assert_called_with("lesiones")

class TestCargarLesionesActivas:
    @patch("data.db.get_client")
    def test_retorna_dataframe_con_datos(self, mock_get_client):
        from data.db import cargar_lesiones_activas
        mock_client = MagicMock()
        query = mock_client.table.return_value.select.return_value
        query = query.in_.return_value
        query = query.order.return_value
        query.execute.return_value = MagicMock(data=[{"id": "u1", "atleta": "Ana", "fecha_lesion": "2026-01-10", "zona_corporal": "Rodilla", "tipo": "Aguda", "gravedad": "Grave", "estado": "Activa", "notas": ""}])
        mock_get_client.return_value = mock_client
        df = cargar_lesiones_activas()
        assert len(df) == 1
        assert df.iloc[0]["atleta"] == "Ana"

class TestActualizarEstadoLesion:
    @patch("data.db.get_client")
    def test_actualizacion_exitosa(self, mock_get_client):
        from data.db import actualizar_estado_lesion
        mock_client = MagicMock()
        mock_client.table.return_value.update.return_value.eq.return_value.execute.return_value = MagicMock(data=[{}])
        mock_get_client.return_value = mock_client
        ok, msg = actualizar_estado_lesion("u1", "Recuperación")
        assert ok
        assert "Recuperación" in msg
