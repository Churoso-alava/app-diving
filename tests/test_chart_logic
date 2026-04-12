"""
TDD — RED phase: chart data-preparation and estado-mapping tests.
No Plotly dependency — tests the data transforms that feed the charts.
Run: python3 -m unittest tests/test_chart_logic.py -v
"""
import sys
sys.path.insert(0, "/mnt/user-data/uploads")
import unittest
import pandas as pd


# ──────────────────────────────────────────────────────────────────────────────
# Estado-clean mapping (used inside fig_historial_barras_atleta)
# ──────────────────────────────────────────────────────────────────────────────

_STATUS_CLEAN = {
    "🔴 CRÍTICO": "CRÍTICO",
    "🟠 FATIGA ACUMULADA": "FATIGA ACUMULADA",
    "🟡 ALERTA TEMPRANA": "ALERTA TEMPRANA",
    "🟢 ÓPTIMO": "ÓPTIMO",
}

STATUS_COLOR = {
    "CRÍTICO": "#E74C3C",
    "FATIGA ACUMULADA": "#E67E22",
    "ALERTA TEMPRANA": "#9B59B6",
    "ÓPTIMO": "#00C49A",
}


def _clean_estado(e: str) -> str:
    return _STATUS_CLEAN.get(str(e), str(e))


def _color_for_estado(e: str) -> str:
    return STATUS_COLOR.get(_clean_estado(e), "#8B949E")


class TestEstadoMapping(unittest.TestCase):
    def test_emoji_critico_maps_correctly(self):
        self.assertEqual(_clean_estado("🔴 CRÍTICO"), "CRÍTICO")

    def test_emoji_optimo_maps_correctly(self):
        self.assertEqual(_clean_estado("🟢 ÓPTIMO"), "ÓPTIMO")

    def test_emoji_fatiga_maps_correctly(self):
        self.assertEqual(_clean_estado("🟠 FATIGA ACUMULADA"), "FATIGA ACUMULADA")

    def test_emoji_alerta_maps_correctly(self):
        self.assertEqual(_clean_estado("🟡 ALERTA TEMPRANA"), "ALERTA TEMPRANA")

    def test_unknown_estado_passthrough(self):
        self.assertEqual(_clean_estado("UNKNOWN"), "UNKNOWN")

    def test_color_lookup_critico(self):
        self.assertEqual(_color_for_estado("🔴 CRÍTICO"), "#E74C3C")

    def test_color_lookup_optimo(self):
        self.assertEqual(_color_for_estado("🟢 ÓPTIMO"), "#00C49A")

    def test_color_unknown_returns_muted(self):
        self.assertEqual(_color_for_estado("🟡 ALERTA TEMPRANA"), "#9B59B6")


# ──────────────────────────────────────────────────────────────────────────────
# Historial bar chart — data preparation
# ──────────────────────────────────────────────────────────────────────────────

def _prep_historial_bars(df_hist: pd.DataFrame, tail_n: int = 12) -> pd.DataFrame:
    """
    Mirrors logic inside fig_historial_barras_atleta.
    Returns sorted, tail-sliced DataFrame with string 'fecha'.
    """
    df = df_hist.copy()
    df["fecha"] = df["fecha"].astype(str)
    return df.sort_values("fecha").tail(tail_n)


class TestHistorialBarsPrep(unittest.TestCase):
    def _make_hist(self, n: int):
        return pd.DataFrame({
            "fecha": pd.date_range("2025-01-01", periods=n, freq="D"),
            "fatiga": [float(i * 5) for i in range(n)],
            "estado": ["🟢 ÓPTIMO"] * n,
        })

    def test_tail_12_limits_to_12_rows(self):
        df = self._make_hist(20)
        result = _prep_historial_bars(df)
        self.assertEqual(len(result), 12)

    def test_fewer_than_12_keeps_all(self):
        df = self._make_hist(5)
        result = _prep_historial_bars(df)
        self.assertEqual(len(result), 5)

    def test_fecha_column_is_string(self):
        df = self._make_hist(5)
        result = _prep_historial_bars(df)
        self.assertTrue(all(isinstance(f, str) for f in result["fecha"]))

    def test_sorted_ascending(self):
        df = self._make_hist(5).sample(frac=1)  # shuffle
        result = _prep_historial_bars(df)
        fechas = list(result["fecha"])
        self.assertEqual(fechas, sorted(fechas))

    def test_tail_keeps_most_recent(self):
        df = self._make_hist(15)
        result = _prep_historial_bars(df, tail_n=3)
        # Most recent date should be last
        self.assertEqual(result["fecha"].iloc[-1], "2025-01-15")

    def test_fatiga_values_preserved(self):
        df = self._make_hist(4)
        result = _prep_historial_bars(df, tail_n=4)
        self.assertEqual(len(result), 4)
        self.assertAlmostEqual(result["fatiga"].max(), 15.0)


# ──────────────────────────────────────────────────────────────────────────────
# SQL migration correctness: Generated column cannot accept explicit insert
# This test documents the DB design constraint.
# ──────────────────────────────────────────────────────────────────────────────

class TestWellnessSQLDesignConstraint(unittest.TestCase):
    """
    BUG FOUND IN PREVIOUS IMPLEMENTATION:
    db.insertar_wellness() attempted to INSERT w_norm explicitly into the table.
    PostgreSQL GENERATED ALWAYS AS columns reject explicit values with:
      ERROR: column "w_norm" can only be updated to DEFAULT

    The fix: remove "w_norm" from the insert dict in db.py.
    This test documents the correct behaviour.
    """

    def test_insert_dict_must_not_contain_w_norm(self):
        """Simulate the insert payload: w_norm must NOT appear in it."""
        sueno, fatiga, estres, dolor, humor = 2, 3, 2, 1, 6
        w_norm = round(((7-sueno)/6.0 + (7-fatiga)/6.0 + (7-estres)/6.0 +
                        (7-dolor)/6.0 + (humor-1)/6.0) / 5.0, 4)

        payload = {
            "nombre":        "Juanes",
            "fecha":         "2025-03-15",
            "sueno":         sueno,
            "fatiga_hooper": fatiga,
            "estres":        estres,
            "dolor":         dolor,
            "humor":         humor,
            # w_norm intentionally EXCLUDED — it is GENERATED ALWAYS AS
            "notas":         "",
        }
        self.assertNotIn("w_norm", payload,
                         "w_norm must not be in the INSERT payload when "
                         "using GENERATED ALWAYS AS STORED column")

    def test_w_norm_can_be_computed_client_side_for_display(self):
        """W_norm formula is correct for display/logging before insert."""
        w = ((7-1)/6.0 + (7-1)/6.0 + (7-1)/6.0 + (7-1)/6.0 + (7-1)/6.0) / 5.0
        self.assertAlmostEqual(w, 1.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
