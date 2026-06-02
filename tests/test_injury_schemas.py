# tests/test_injury_schemas.py
"""Tests unitarios para InjuryInput. Sin dependencias de DB ni Streamlit."""
import pytest
from core.schemas import (
    InjuryInput,
    ZONA_CORPORAL_OPTIONS,
    TIPO_LESION_OPTIONS,
    GRAVEDAD_OPTIONS,
    ESTADO_LESION_OPTIONS,
)


class TestInjuryInputValidacion:
    def test_instancia_valida_defaults(self):
        record = InjuryInput(
            atleta="Carlos",
            fecha_lesion="2026-01-15",
            zona_corporal="Rodilla",
            tipo="Aguda",
            gravedad="Moderada",
        )
        assert record.estado == "Activa"
        assert record.notas == ""
        assert record.fecha_evento is None
        assert record.mecanismo_contacto is False

    def test_instancia_valida_nuevos_campos(self):
        record = InjuryInput(
            atleta="Carlos",
            fecha_lesion="2026-01-15",
            zona_corporal="Rodilla",
            tipo="Aguda",
            gravedad="Moderada",
            tipo_tejido="musculo",
            mecanismo="aguda",
            recurrencia="nueva",
            mecanismo_contacto=True,
            fecha_evento="2026-01-14",
            fecha_alta_medica="2026-01-20",
            fecha_rtt="2026-01-25",
            fecha_rtp="2026-02-01"
        )
        d = record.to_dict()
        assert d["tipo_tejido"] == "musculo"
        assert d["mecanismo"] == "aguda"
        assert d["mecanismo_contacto"] is True
        assert d["fecha_evento"] == "2026-01-14"

    def test_fecha_evento_invalida_lanza_error(self):
        with pytest.raises(ValueError, match="fecha_evento"):
            InjuryInput(
                atleta="Carlos",
                fecha_lesion="2026-01-15",
                zona_corporal="Rodilla",
                tipo="Aguda",
                gravedad="Leve",
                fecha_evento="not-a-date"
            )

    def test_to_dict_no_incluye_opcionales_none(self):
        record = InjuryInput(
            atleta="Ana",
            fecha_lesion="2026-01-15",
            zona_corporal="Hombro",
            tipo="Sobreuso",
            gravedad="Leve",
        )
        d = record.to_dict()
        assert "fecha_evento" not in d
        assert "fecha_alta_medica" not in d
        assert "fecha_rtt" not in d
        assert "fecha_rtp" not in d
        assert d["atleta"] == "Ana"
        assert d["estado"] == "Activa"

    def test_atleta_vacio_lanza_error(self):
        with pytest.raises(ValueError, match="atleta"):
            InjuryInput(
                atleta="",
                fecha_lesion="2026-01-15",
                zona_corporal="Hombro",
                tipo="Sobreuso",
                gravedad="Leve",
            )

    def test_zona_corporal_invalida_lanza_error(self):
        with pytest.raises(ValueError, match="zona_corporal"):
            InjuryInput(
                atleta="Carlos",
                fecha_lesion="2026-01-15",
                zona_corporal="Codo",  # no está en la lista
                tipo="Aguda",
                gravedad="Moderada",
            )

    def test_tipo_invalido_lanza_error(self):
        with pytest.raises(ValueError, match="tipo"):
            InjuryInput(
                atleta="Ana",
                fecha_lesion="2026-01-15",
                zona_corporal="Rodilla",
                tipo="Crónica",  # no está en la lista
                gravedad="Leve",
            )

    def test_gravedad_invalida_lanza_error(self):
        with pytest.raises(ValueError, match="gravedad"):
            InjuryInput(
                atleta="Luis",
                fecha_lesion="2026-01-15",
                zona_corporal="Espalda",
                tipo="Sobreuso",
                gravedad="Extrema",  # no está en la lista
            )

    def test_estado_invalido_lanza_error(self):
        with pytest.raises(ValueError, match="estado"):
            InjuryInput(
                atleta="Ana",
                fecha_lesion="2026-01-15",
                zona_corporal="Hombro",
                tipo="Aguda",
                gravedad="Grave",
                estado="Pendiente",  # no está en la lista
            )

    def test_fecha_lesion_invalida_lanza_error(self):
        with pytest.raises(ValueError, match="fecha_lesion"):
            InjuryInput(
                atleta="Carlos",
                fecha_lesion="not-a-date",
                zona_corporal="Rodilla",
                tipo="Aguda",
                gravedad="Leve",
            )

    def test_fecha_alta_invalida_lanza_error(self):
        with pytest.raises(ValueError, match="fecha_alta_medica"):
            InjuryInput(
                atleta="Luis",
                fecha_lesion="2026-01-01",
                zona_corporal="Tobillo",
                tipo="Aguda",
                gravedad="Leve",
                fecha_alta_medica="mañana",  # inválida
            )

    def test_multiples_errores_en_un_mensaje(self):
        with pytest.raises(ValueError) as exc_info:
            InjuryInput(
                atleta="",
                fecha_lesion="bad",
                zona_corporal="Codo",
                tipo="Aguda",
                gravedad="Leve",
            )
        msg = str(exc_info.value)
        # Todos los errores están en el mismo mensaje
        assert "atleta" in msg
        assert "zona_corporal" in msg

    def test_constantes_completas(self):
        assert "Hombro" in ZONA_CORPORAL_OPTIONS
        assert "Otro" in ZONA_CORPORAL_OPTIONS
        assert "Aguda" in TIPO_LESION_OPTIONS
        assert "Sobreuso" in TIPO_LESION_OPTIONS
        assert "Grave" in GRAVEDAD_OPTIONS
        assert "Alta" in ESTADO_LESION_OPTIONS
