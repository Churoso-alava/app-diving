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
        assert record.estado == "Activa"    # default
        assert record.notas == ""           # default
        assert record.fecha_alta is None    # default

    def test_to_dict_no_incluye_fecha_alta_none(self):
        record = InjuryInput(
            atleta="Ana",
            fecha_lesion="2026-01-15",
            zona_corporal="Hombro",
            tipo="Sobreuso",
            gravedad="Leve",
        )
        d = record.to_dict()
        assert "fecha_alta" not in d
        assert d["atleta"] == "Ana"
        assert d["estado"] == "Activa"

    def test_to_dict_incluye_fecha_alta_si_presente(self):
        record = InjuryInput(
            atleta="Luis",
            fecha_lesion="2026-01-01",
            zona_corporal="Tobillo",
            tipo="Aguda",
            gravedad="Leve",
            estado="Alta",
            fecha_alta="2026-01-20",
        )
        d = record.to_dict()
        assert d["fecha_alta"] == "2026-01-20"
        assert d["estado"] == "Alta"

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
        with pytest.raises(ValueError, match="fecha_alta"):
            InjuryInput(
                atleta="Luis",
                fecha_lesion="2026-01-01",
                zona_corporal="Tobillo",
                tipo="Aguda",
                gravedad="Leve",
                fecha_alta="mañana",  # inválida
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
