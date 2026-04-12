"""
TDD — RED phase: diving_load.py unit tests.
Run: python3 -m unittest tests/test_diving_load.py -v
"""
import sys, os
sys.path.insert(0, "/mnt/user-data/uploads")
import unittest
from diving_load import (
    k_alt, k_dd, k_tipo, k_angulo,
    carga_bruta_sesion, normalizar_carga,
    calcular_wellness, carga_integrada,
    K_TIPO, L_MAX_REFERENCIA,
)


class TestKAlt(unittest.TestCase):
    def test_reference_height_returns_one(self):
        self.assertAlmostEqual(k_alt(1.0), 1.0)

    def test_linear_scaling_10m(self):
        self.assertAlmostEqual(k_alt(10.0), 10.0)

    def test_zero_height_raises(self):
        with self.assertRaises(ValueError):
            k_alt(0)

    def test_negative_height_raises(self):
        with self.assertRaises(ValueError):
            k_alt(-1)

    def test_beta_half_sqrt(self):
        self.assertAlmostEqual(k_alt(4.0, beta=0.5), 2.0, places=5)


class TestKDd(unittest.TestCase):
    def test_reference_dd_returns_one(self):
        self.assertAlmostEqual(k_dd(2.0), 1.0)

    def test_below_min_raises(self):
        with self.assertRaises(ValueError):
            k_dd(1.1)

    def test_above_max_raises(self):
        with self.assertRaises(ValueError):
            k_dd(4.5)

    def test_max_dd_value(self):
        self.assertAlmostEqual(k_dd(4.4), 2.2, places=5)

    def test_min_dd_value(self):
        self.assertAlmostEqual(k_dd(1.2), 0.6, places=5)


class TestKTipo(unittest.TestCase):
    def test_head_returns_one(self):
        self.assertAlmostEqual(k_tipo("HEAD"), 1.0)

    def test_twist_higher_than_head(self):
        self.assertGreater(k_tipo("TWIST"), k_tipo("HEAD"))

    def test_feet_lower_than_head(self):
        self.assertLess(k_tipo("FEET"), k_tipo("HEAD"))

    def test_invalid_tipo_raises(self):
        with self.assertRaises(ValueError):
            k_tipo("FLIP")

    def test_all_valid_tipos_exist(self):
        for t in ["HEAD", "FEET", "TWIST", "PIKE", "SYNC"]:
            self.assertIn(t, K_TIPO)


class TestKAngulo(unittest.TestCase):
    def test_none_returns_one(self):
        self.assertAlmostEqual(k_angulo(None), 1.0)

    def test_zero_degrees_returns_one(self):
        self.assertAlmostEqual(k_angulo(0.0), 1.0)

    def test_45_degrees_returns_1_3(self):
        self.assertAlmostEqual(k_angulo(45.0), 1.3, places=5)

    def test_negative_angle_raises(self):
        with self.assertRaises(ValueError):
            k_angulo(-1.0)

    def test_above_45_raises(self):
        with self.assertRaises(ValueError):
            k_angulo(46.0)

    def test_midpoint_22_5_degrees(self):
        result = k_angulo(22.5)
        self.assertAlmostEqual(result, 1.15, places=5)


class TestCargaBrutaSesion(unittest.TestCase):
    def setUp(self):
        self.single = [{"altura": 1.0, "dd": 2.0, "tipo": "HEAD"}]

    def test_single_clavado_head_1m_dd2(self):
        # k_alt=1.0, k_dd=1.0, k_tipo=1.0 → 1.0
        self.assertAlmostEqual(carga_bruta_sesion(self.single), 1.0)

    def test_missing_key_raises(self):
        with self.assertRaises(ValueError):
            carga_bruta_sesion([{"altura": 1.0, "dd": 2.0}])  # sin "tipo"

    def test_invalid_tipo_raises(self):
        with self.assertRaises(ValueError):
            carga_bruta_sesion([{"altura": 1.0, "dd": 2.0, "tipo": "FLIP"}])

    def test_multiple_clavados_sum(self):
        clavados = [
            {"altura": 1.0, "dd": 2.0, "tipo": "HEAD"},
            {"altura": 1.0, "dd": 2.0, "tipo": "HEAD"},
        ]
        self.assertAlmostEqual(carga_bruta_sesion(clavados), 2.0)

    def test_angulo_optional_defaults_to_one(self):
        without = carga_bruta_sesion([{"altura": 10.0, "dd": 2.0, "tipo": "HEAD"}])
        with_zero = carga_bruta_sesion([{"altura": 10.0, "dd": 2.0, "tipo": "HEAD", "angulo_grados": 0.0}])
        self.assertAlmostEqual(without, with_zero)

    def test_empty_list_returns_zero(self):
        self.assertAlmostEqual(carga_bruta_sesion([]), 0.0)


class TestNormalizarCarga(unittest.TestCase):
    def test_zero_load_is_zero(self):
        self.assertAlmostEqual(normalizar_carga(0.0), 0.0)

    def test_l_max_returns_100(self):
        self.assertAlmostEqual(normalizar_carga(L_MAX_REFERENCIA), 100.0)

    def test_above_max_capped_at_100(self):
        self.assertAlmostEqual(normalizar_carga(L_MAX_REFERENCIA * 2), 100.0)

    def test_half_max_returns_50(self):
        self.assertAlmostEqual(normalizar_carga(L_MAX_REFERENCIA * 0.5), 50.0)


class TestCalcularWellness(unittest.TestCase):
    def test_all_optimal_returns_one(self):
        # sueno=1 (óptimo), fatiga=1, estres=1, dolor=1, humor=7 (óptimo)
        self.assertAlmostEqual(calcular_wellness(1, 1, 1, 1, 7), 1.0)

    def test_all_worst_returns_zero(self):
        self.assertAlmostEqual(calcular_wellness(7, 7, 7, 7, 1), 0.0)

    def test_neutral_midpoint(self):
        w = calcular_wellness(4, 4, 4, 4, 4)
        self.assertAlmostEqual(w, 0.5, places=5)

    def test_out_of_range_raises(self):
        with self.assertRaises(ValueError):
            calcular_wellness(0, 4, 4, 4, 4)  # sueno=0 inválido

    def test_out_of_range_high_raises(self):
        with self.assertRaises(ValueError):
            calcular_wellness(4, 4, 4, 4, 8)  # humor=8 inválido

    def test_result_in_0_1_range(self):
        w = calcular_wellness(3, 3, 3, 3, 5)
        self.assertGreaterEqual(w, 0.0)
        self.assertLessEqual(w, 1.0)


class TestCargaIntegrada(unittest.TestCase):
    def test_optimal_wellness_no_amplification(self):
        # CI = L_norm * (2 - 1.0) = L_norm
        self.assertAlmostEqual(carga_integrada(50.0, 1.0), 50.0)

    def test_worst_wellness_doubles_load(self):
        # CI = L_norm * (2 - 0.0) = 2 * L_norm
        self.assertAlmostEqual(carga_integrada(50.0, 0.0), 100.0)

    def test_neutral_wellness_factor_1_5(self):
        self.assertAlmostEqual(carga_integrada(100.0, 0.5), 150.0)

    def test_zero_load_always_zero(self):
        self.assertAlmostEqual(carga_integrada(0.0, 0.5), 0.0)

    def test_max_possible_output(self):
        # L_norm=100, W_norm=0 → CI=200
        self.assertAlmostEqual(carga_integrada(100.0, 0.0), 200.0)


if __name__ == "__main__":
    unittest.main()
