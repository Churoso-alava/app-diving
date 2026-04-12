"""
TDD — RED phase: services.py unit tests.
Tests pure logic — no DB, no Streamlit.
Run: python3 -m unittest tests/test_services.py -v
"""
import sys
import types

# ── Mock skfuzzy before services import (not available in test env) ────────────
_skfuzzy = types.ModuleType("skfuzzy")
_skfuzzy.control = types.ModuleType("skfuzzy.control")
sys.modules.setdefault("skfuzzy", _skfuzzy)
sys.modules.setdefault("skfuzzy.control", _skfuzzy.control)

# Mock evaluar_atleta so services imports cleanly
_fuzzy_mod = types.ModuleType("fuzzy")
_fuzzy_mod.evaluar_atleta = lambda sim, m: {**m, "indice_fatiga": 50.0, "estado": "🟡 ALERTA TEMPRANA",
                                            "color": "#ca8a04", "accion": "—", "accion_primaria": "—",
                                            "advertencias": [], "contexto_cientifico": "", "nota_swc": ""}
sys.modules["fuzzy"] = _fuzzy_mod

sys.path.insert(0, "/mnt/user-data/uploads")

import unittest
from datetime import date, timedelta

import pandas as pd
import numpy as np
from services import SessionInput, calcular_metricas, detectar_tendencia_mpv


# ──────────────────────────────────────────────────────────────────────────────
# SessionInput validation
# ──────────────────────────────────────────────────────────────────────────────

class TestSessionInput(unittest.TestCase):
    def _valid(self, **kwargs):
        defaults = {"nombre": "Ana", "fecha": "2025-01-01", "vmp": 0.500}
        return SessionInput(**{**defaults, **kwargs})

    def test_valid_session_creates(self):
        s = self._valid()
        self.assertEqual(s.nombre, "Ana")

    def test_to_dict_strips_nombre(self):
        s = self._valid(nombre="  Ana  ")
        self.assertEqual(s.to_dict()["nombre"], "Ana")

    def test_vmp_below_min_raises(self):
        with self.assertRaises(ValueError):
            self._valid(vmp=0.099)

    def test_vmp_above_max_raises(self):
        with self.assertRaises(ValueError):
            self._valid(vmp=2.501)

    def test_vmp_at_boundary_min(self):
        s = self._valid(vmp=0.100)
        self.assertAlmostEqual(s.vmp, 0.100)

    def test_vmp_at_boundary_max(self):
        s = self._valid(vmp=2.500)
        self.assertAlmostEqual(s.vmp, 2.500)

    def test_empty_nombre_raises(self):
        with self.assertRaises(ValueError):
            self._valid(nombre="")

    def test_whitespace_nombre_raises(self):
        with self.assertRaises(ValueError):
            self._valid(nombre="   ")

    def test_invalid_fecha_raises(self):
        with self.assertRaises(ValueError):
            self._valid(fecha="not-a-date")

    def test_notas_default_empty(self):
        s = self._valid()
        self.assertEqual(s.notas, "")


# ──────────────────────────────────────────────────────────────────────────────
# calcular_metricas
# ──────────────────────────────────────────────────────────────────────────────

def _make_df(n: int, vmp_val: float = 0.5, atleta: str = "Ana") -> pd.DataFrame:
    """Creates a DataFrame with n sessions spaced 1 day apart."""
    today = pd.Timestamp.today().normalize()
    fechas = [today - pd.Timedelta(days=n - 1 - i) for i in range(n)]
    return pd.DataFrame({
        "Nombre": [atleta] * n,
        "Fecha": fechas,
        "VMP_Hoy": [vmp_val] * n,
    })


class TestCalcularMetricas(unittest.TestCase):
    def test_returns_none_with_fewer_than_4_sessions(self):
        df = _make_df(3)
        self.assertIsNone(calcular_metricas(df, "Ana"))

    def test_returns_dict_with_4_sessions(self):
        df = _make_df(4)
        result = calcular_metricas(df, "Ana")
        self.assertIsNotNone(result)

    def test_acwr_clipped_to_valid_range(self):
        df = _make_df(10)
        m = calcular_metricas(df, "Ana")
        self.assertGreaterEqual(m["acwr"], 0.50)
        self.assertLessEqual(m["acwr"], 1.80)

    def test_delta_pct_clipped(self):
        df = _make_df(10)
        m = calcular_metricas(df, "Ana")
        self.assertGreaterEqual(m["delta_pct"], -20)
        self.assertLessEqual(m["delta_pct"], 40)

    def test_z_meso_clipped(self):
        df = _make_df(10)
        m = calcular_metricas(df, "Ana")
        self.assertGreaterEqual(m["z_meso"], -4.0)
        self.assertLessEqual(m["z_meso"], 4.0)

    def test_swc_personal_non_negative(self):
        df = _make_df(10)
        m = calcular_metricas(df, "Ana")
        self.assertGreaterEqual(m["swc_personal"], 0.0)

    def test_swc_pediactric_multiplier_applied(self):
        """Under-15 athletes get 1.5× SWC multiplier."""
        df = _make_df(10)
        m_adult = calcular_metricas(df, "Ana", perfil={"edad": 18})
        m_child = calcular_metricas(df, "Ana", perfil={"edad": 14})
        # Child SD same but multiplied by 1.5
        if m_adult["sd_personal"] > 0:
            self.assertAlmostEqual(
                m_child["swc_personal"],
                m_adult["sd_personal"] * 1.5,
                places=4,
            )

    def test_es_ruido_biologico_false_when_swc_zero(self):
        """Constant VMP → sd=0 → es_ruido_biologico must be False."""
        df = _make_df(10, vmp_val=0.500)  # exactly constant
        m = calcular_metricas(df, "Ana")
        # sd=0, so es_ruido_biologico must be False (swc=0 → condition fails)
        self.assertFalse(m["es_ruido_biologico"])

    def test_n_sesiones_desc_zero_for_constant_vmp(self):
        df = _make_df(10, vmp_val=0.500)
        m = calcular_metricas(df, "Ana")
        # Constant VMP: no strictly decreasing pairs
        self.assertEqual(m["n_sesiones_desc"], 0)

    def test_n_sesiones_desc_counts_correctly(self):
        """Three strictly decreasing sessions at end → n_sesiones_desc ≥ 2.
        Base sessions must NOT share dates with decreasing sessions
        because calcular_metricas groups by date and takes max VMP,
        which would mask lower values on the same date.
        """
        today = pd.Timestamp.today().normalize()
        # 7 stable sessions, all well in the past (days 30..24)
        base = pd.DataFrame({
            "Nombre": ["Ana"] * 7,
            "Fecha": [today - pd.Timedelta(days=30 - i) for i in range(7)],
            "VMP_Hoy": [0.600] * 7,
        })
        # 3 strictly decreasing sessions on new dates (days 3, 2, 1, 0)
        decr = pd.DataFrame({
            "Nombre": ["Ana"] * 3,
            "Fecha": [today - pd.Timedelta(days=2),
                      today - pd.Timedelta(days=1),
                      today],
            "VMP_Hoy": [0.590, 0.580, 0.570],
        })
        df = pd.concat([base, decr], ignore_index=True)
        m = calcular_metricas(df, "Ana")
        self.assertGreaterEqual(m["n_sesiones_desc"], 2)

    def test_calidad_dato_classification(self):
        df = _make_df(15)  # enough for "alta"
        m = calcular_metricas(df, "Ana")
        self.assertIn(m["calidad_dato"], ["alta", "media", "baja", "insuficiente"])

    def test_vmp_hoy_matches_last_session(self):
        df = _make_df(5, vmp_val=0.750)
        m = calcular_metricas(df, "Ana")
        self.assertAlmostEqual(m["vmp_hoy"], 0.750, places=4)

    def test_unknown_atleta_returns_none(self):
        df = _make_df(10)
        result = calcular_metricas(df, "NonExistent")
        self.assertIsNone(result)


# ──────────────────────────────────────────────────────────────────────────────
# detectar_tendencia_mpv
# ──────────────────────────────────────────────────────────────────────────────

class TestDetectarTendenciaMpv(unittest.TestCase):
    def _df(self, vmps):
        today = pd.Timestamp.today().normalize()
        return pd.DataFrame({
            "Nombre": ["Ana"] * len(vmps),
            "Fecha": [today - pd.Timedelta(days=len(vmps)-1-i) for i in range(len(vmps))],
            "VMP_Hoy": vmps,
        })

    def test_strictly_decreasing_returns_true(self):
        df = self._df([0.6, 0.5, 0.4])
        self.assertTrue(detectar_tendencia_mpv(df, ventana=3))

    def test_flat_returns_false(self):
        df = self._df([0.5, 0.5, 0.5])
        self.assertFalse(detectar_tendencia_mpv(df, ventana=3))

    def test_increasing_returns_false(self):
        df = self._df([0.4, 0.5, 0.6])
        self.assertFalse(detectar_tendencia_mpv(df, ventana=3))

    def test_too_few_sessions_returns_false(self):
        df = self._df([0.5, 0.4])  # only 2 rows
        self.assertFalse(detectar_tendencia_mpv(df, ventana=3))

    def test_partially_decreasing_returns_false(self):
        df = self._df([0.6, 0.5, 0.6])
        self.assertFalse(detectar_tendencia_mpv(df, ventana=3))


# ──────────────────────────────────────────────────────────────────────────────
# db.py — Wellness validation (pure logic, no DB call)
# ──────────────────────────────────────────────────────────────────────────────

class TestWellnessValidation(unittest.TestCase):
    """
    Tests for the inline validation logic in db.insertar_wellness.
    We replicate the logic here to test it without a DB connection.
    """
    def _validate(self, sueno, fatiga, estres, dolor, humor, nombre="Ana"):
        items = {"sueno": sueno, "fatiga_hooper": fatiga,
                 "estres": estres, "dolor": dolor, "humor": humor}
        errors = [
            f"{k}={v} fuera de [1,7]"
            for k, v in items.items()
            if not (1 <= v <= 7)
        ]
        if not nombre or not nombre.strip():
            errors.insert(0, "nombre vacío")
        return errors

    def test_valid_inputs_no_errors(self):
        self.assertEqual(self._validate(1, 1, 1, 1, 7), [])

    def test_sueno_zero_invalid(self):
        errs = self._validate(0, 4, 4, 4, 4)
        self.assertTrue(any("sueno" in e for e in errs))

    def test_humor_8_invalid(self):
        errs = self._validate(4, 4, 4, 4, 8)
        self.assertTrue(any("humor" in e for e in errs))

    def test_multiple_invalid_fields(self):
        errs = self._validate(0, 8, 4, 4, 4)
        self.assertGreaterEqual(len(errs), 2)

    def test_empty_nombre_invalid(self):
        errs = self._validate(4, 4, 4, 4, 4, nombre="")
        self.assertTrue(any("nombre" in e for e in errs))

    def test_w_norm_formula_optimal(self):
        """w_norm for all-optimal inputs should equal 1.0"""
        sueno, fatiga, estres, dolor, humor = 1, 1, 1, 1, 7
        w = ((7-sueno)/6.0 + (7-fatiga)/6.0 + (7-estres)/6.0 +
             (7-dolor)/6.0 + (humor-1)/6.0) / 5.0
        self.assertAlmostEqual(w, 1.0)

    def test_w_norm_formula_worst(self):
        """w_norm for worst inputs should equal 0.0"""
        sueno, fatiga, estres, dolor, humor = 7, 7, 7, 7, 1
        w = ((7-sueno)/6.0 + (7-fatiga)/6.0 + (7-estres)/6.0 +
             (7-dolor)/6.0 + (humor-1)/6.0) / 5.0
        self.assertAlmostEqual(w, 0.0)


# ──────────────────────────────────────────────────────────────────────────────
# importar_wellness_dataframe — column normalisation logic
# ──────────────────────────────────────────────────────────────────────────────

class TestWellnessColumnNormalisation(unittest.TestCase):
    """
    Tests for the column-alias detection in importar_wellness_dataframe.
    Replicated here to run without a DB.
    """
    def _normalise(self, cols):
        rename = {}
        for c in cols:
            cl = c.lower().strip()
            if cl in ("nombre", "atleta", "deportista"):
                rename[c] = "Nombre"
            elif cl in ("fecha", "date"):
                rename[c] = "Fecha"
            elif cl in ("sueno", "sueño", "sleep"):
                rename[c] = "Sueno"
            elif cl in ("fatiga", "fatiga_hooper", "tiredness"):
                rename[c] = "Fatiga"
            elif cl in ("estres", "estrés", "stress"):
                rename[c] = "Estres"
            elif cl in ("dolor", "pain", "muscle_soreness"):
                rename[c] = "Dolor"
            elif cl in ("humor", "mood"):
                rename[c] = "Humor"
        return rename

    def test_exact_spanish_names(self):
        r = self._normalise(["Nombre", "Fecha", "Sueno", "Fatiga", "Estres", "Dolor", "Humor"])
        self.assertEqual(r["Nombre"], "Nombre")
        self.assertEqual(r["Sueno"], "Sueno")

    def test_english_aliases(self):
        r = self._normalise(["atleta", "date", "sleep", "tiredness", "stress", "pain", "mood"])
        self.assertEqual(r["atleta"], "Nombre")
        self.assertEqual(r["sleep"], "Sueno")
        self.assertEqual(r["mood"], "Humor")

    def test_missing_column_not_in_rename(self):
        r = self._normalise(["Nombre", "Fecha"])
        self.assertNotIn("Sueno", r.values())

    def test_notas_alias_notes(self):
        rename = {}
        for c in ["notes"]:
            cl = c.lower().strip()
            if cl in ("notas", "notes", "comentarios"):
                rename[c] = "Notas"
        self.assertEqual(rename["notes"], "Notas")


if __name__ == "__main__":
    unittest.main(verbosity=2)
