"""
tests/test_app_startup.py — Verifica que app.py sea importable y no tenga errores de sintaxis o de importación.
"""
import unittest
import sys
import os

# Asegurar que el directorio raíz esté en el path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class TestAppStartup(unittest.TestCase):

    def test_app_importable(self):
        """Verifica que app.py se pueda importar sin errores."""
        try:
            import app
            self.assertTrue(True)
        except Exception as e:
            self.fail(f"Error al importar app.py: {e}")

    def test_db_importable(self):
        """Verifica que db.py se pueda importar sin errores."""
        try:
            import db
            self.assertTrue(True)
        except Exception as e:
            self.fail(f"Error al importar db.py: {e}")

if __name__ == "__main__":
    unittest.main()
