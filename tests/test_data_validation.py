"""
Tests para validar que db.py filtra correctamente NaN/NaT
"""
import pandas as pd
import numpy as np
from datetime import date
import pytest

# Ajusta el import según tu estructura real
try:
    from db import cargar_sesiones_raw, cargar_lesiones, _validar_numericos
except ImportError:
    pytest.skip("db module not available", allow_module_level=True)


# Mocking the Supabase client and its methods for isolation
class MockSupabaseResponse:
    def __init__(self, data):
        self.data = data
        self.error = None

class MockSupabaseClient:
    def __init__(self):
        self._data = {}

    def table(self, table_name):
        self._current_table = table_name
        return self

    def select(self, columns):
        self._selected_columns = columns
        return self
        
    def order(self, column, desc=False):
        self._order_column = column
        self._order_desc = desc
        return self

    def execute(self):
        if self._current_table == "sesiones_vmp":
            return MockSupabaseResponse(self._data.get("sesiones_vmp", []))
        elif self._current_table == "lesiones":
            return MockSupabaseResponse(self._data.get("lesiones", []))
        return MockSupabaseResponse([])

    def set_data(self, table_name, data):
        self._data[table_name] = data

# Mocking logging to capture output without actual logging setup
class MockLog:
    def warning(self, msg, *args): self.log_messages.append(f"WARNING: {msg % args}")
    def info(self, msg, *args): self.log_messages.append(f"INFO: {msg % args}")
    def error(self, msg, *args): self.log_messages.append(f"ERROR: {msg % args}")
    def __init__(self): self.log_messages = []

# Temporarily patch the Supabase client getter and logger
import sys
sys.modules['_get_client'] = lambda: MockSupabaseClient() # Mock the client getter
# We need to ensure _validar_numericos is available and the logger is mocked
# If _validar_numericos is defined in db.py, it should be importable directly.
# We also need to ensure the logger used within db.py is mocked.
# For simplicity, let's assume the logging is handled via a global 'log' object
# or that importing 'db' brings in its logger setup.

# --- Mocking logger used in db.py ---
# This is a simplified approach. A more robust solution would use unittest.mock.patch.
# We assume 'log' is a global object used within db.py
# If db.py uses `import logging; log = logging.getLogger(__name__)`, this mock won't work directly without patching.
# For this context, we'll assume we can inject a mock log object if needed, or that the function doesn't critically depend on logger side effects for tests.

# Let's try to import the actual functions and assume the logger is handled.
# If it fails, we might need a more sophisticated mocking strategy.
try:
    from db import _validar_numericos, cargar_sesiones_raw, cargar_lesiones
    # Attempt to get the actual logger if possible, or use a mock
    import logging
    log = logging.getLogger(__name__) # Use a real logger for the test file itself
except ImportError as e:
    pytest.skip(f"Could not import from db: {e}", allow_module_level=True)

# ─────────────────────────────────────────────────────────────────────────────
# Tests para _validar_numericos
# ─────────────────────────────────────────────────────────────────────────────

def test_validar_numericos_elimina_nan():
    """Verifica que _validar_numericos elimine filas con NaN en columnas especificadas."""
    df_input = pd.DataFrame({
        "col_num1": [1, 2, np.nan, 4, 5],
        "col_num2": [1.1, np.nan, 3.3, 4.4, 5.5],
        "col_str": ["a", "b", "c", "d", "e"]
    })
    cols_to_validate = ["col_num1", "col_num2"]
    
    df_output = _validar_numericos(df_input, cols_to_validate)
    
    assert len(df_output) == 3
    assert not df_output["col_num1"].isnull().any()
    assert not df_output["col_num2"].isnull().any()
    assert "col_str" in df_output.columns

def test_validar_numericos_sin_nan():
    """Verifica que _validar_numericos no modifique un DataFrame sin NaN."""
    df_input = pd.DataFrame({
        "col_num1": [1, 2, 3, 4, 5],
        "col_num2": [1.1, 2.2, 3.3, 4.4, 5.5],
        "col_str": ["a", "b", "c", "d", "e"]
    })
    cols_to_validate = ["col_num1", "col_num2"]
    
    df_output = _validar_numericos(df_input, cols_to_validate)
    
    pd.testing.assert_frame_equal(df_input, df_output)

def test_validar_numericos_columna_inexistente():
    """Verifica que _validar_numericos maneje columnas no existentes."""
    df_input = pd.DataFrame({
        "col_num1": [1, 2, np.nan, 4, 5],
        "col_str": ["a", "b", "c", "d", "e"]
    })
    cols_to_validate = ["col_num1", "non_existent_col"]
    
    # Should not raise an error and should still filter based on existing columns
    df_output = _validar_numericos(df_input, cols_to_validate)
    
    assert len(df_output) == 4
    assert not df_output["col_num1"].isnull().any()
    assert "col_str" in df_output.columns

# ─────────────────────────────────────────────────────────────────────────────
# Tests para cargar_sesiones_raw
# ─────────────────────────────────────────────────────────────────────────────

def test_cargar_sesiones_raw_filtra_nat_en_fecha():
    """Verifica que cargar_sesiones_raw descarta filas con NaT en fecha"""
    mock_client = MockSupabaseClient()
    data = [
        {"fecha": "2023-01-01", "vmp_hoy": 5.0, "vmp_ref": 4.8},
        {"fecha": None, "vmp_hoy": 5.1, "vmp_ref": 4.9}, # NaT representation
        {"fecha": "2023-01-03", "vmp_hoy": 5.2, "vmp_ref": 5.0},
        {"fecha": pd.NaT, "vmp_hoy": 5.3, "vmp_ref": 5.1}, # Actual pd.NaT
    ]
    mock_client.set_data("sesiones_vmp", data)
    
    # Temporarily replace the global _get_client for this test
    # This is a common pattern in testing modules that rely on global setup
    original_get_client = sys.modules['_get_client']
    sys.modules['_get_client'] = lambda: mock_client
    
    df = cargar_sesiones_raw()
    
    # Restore the original _get_client
    sys.modules['_get_client'] = original_get_client
    
    assert len(df) == 2
    assert not df["fecha"].isnull().any()
    assert df["fecha"].iloc[0] == pd.to_datetime("2023-01-01").date()
    assert df["fecha"].iloc[1] == pd.to_datetime("2023-01-03").date()

def test_cargar_sesiones_raw_filtra_nan_en_vmp():
    """Verifica que cargar_sesiones_raw descarta filas con NaN en vmp_hoy."""
    mock_client = MockSupabaseClient()
    data = [
        {"fecha": "2023-01-01", "vmp_hoy": 5.0, "vmp_ref": 4.8},
        {"fecha": "2023-01-02", "vmp_hoy": np.nan, "vmp_ref": 4.9}, # NaN
        {"fecha": "2023-01-03", "vmp_hoy": 5.2, "vmp_ref": 5.0},
        {"fecha": "2023-01-04", "vmp_hoy": 5.3, "vmp_ref": np.nan}, # NaN in vmp_ref, should also be filtered by _validar_numericos
    ]
    mock_client.set_data("sesiones_vmp", data)
    
    original_get_client = sys.modules['_get_client']
    sys.modules['_get_client'] = lambda: mock_client
    
    df = cargar_sesiones_raw()
    
    sys.modules['_get_client'] = original_get_client
    
    assert len(df) == 2
    assert not df["vmp_hoy"].isnull().any()
    assert not df["vmp_ref"].isnull().any() # vmp_ref should also be filtered by _validar_numericos
    assert df["fecha"].iloc[0] == pd.to_datetime("2023-01-01").date()
    assert df["vmp_hoy"].iloc[0] == 5.0
    assert df["fecha"].iloc[1] == pd.to_datetime("2023-01-03").date()
    assert df["vmp_hoy"].iloc[1] == 5.2

def test_cargar_sesiones_raw_empty_data():
    """Verifica que cargar_sesiones_raw maneje data vacía."""
    mock_client = MockSupabaseClient()
    mock_client.set_data("sesiones_vmp", [])
    
    original_get_client = sys.modules['_get_client']
    sys.modules['_get_client'] = lambda: mock_client
    
    df = cargar_sesiones_raw()
    
    sys.modules['_get_client'] = original_get_client
    
    assert df.empty

def test_cargar_sesiones_raw_no_valid_data_after_filter():
    """Verifica que cargar_sesiones_raw maneje caso donde todo se filtra."""
    mock_client = MockSupabaseClient()
    data = [
        {"fecha": None, "vmp_hoy": np.nan, "vmp_ref": np.nan},
        {"fecha": pd.NaT, "vmp_hoy": np.nan, "vmp_ref": np.nan},
    ]
    mock_client.set_data("sesiones_vmp", data)
    
    original_get_client = sys.modules['_get_client']
    sys.modules['_get_client'] = lambda: mock_client
    
    df = cargar_sesiones_raw()
    
    sys.modules['_get_client'] = original_get_client
    
    assert df.empty

# ─────────────────────────────────────────────────────────────────────────────
# Tests para cargar_lesiones
# ─────────────────────────────────────────────────────────────────────────────

def test_cargar_lesiones_filtra_nat():
    """Verifica que cargar_lesiones descarta filas con NaT en fechas"""
    mock_client = MockSupabaseClient()
    data = [
        {"fecha_inicio": "2022-01-01", "fecha_alta": "2022-02-01", "descripcion": "Lesión A"},
        {"fecha_inicio": None, "fecha_alta": "2022-02-15", "descripcion": "Lesión B"}, # NaT representation
        {"fecha_inicio": "2022-03-10", "fecha_alta": "2022-04-01", "descripcion": "Lesión C"},
        {"fecha_inicio": pd.NaT, "fecha_alta": "2022-04-10", "descripcion": "Lesión D"}, # Actual pd.NaT
    ]
    mock_client.set_data("lesiones", data)
    
    original_get_client = sys.modules['_get_client']
    sys.modules['_get_client'] = lambda: mock_client
    
    df = cargar_lesiones()
    
    sys.modules['_get_client'] = original_get_client
    
    assert len(df) == 2
    assert not df["fecha_inicio"].isnull().any()
    assert not df["fecha_alta"].isnull().any()
    assert df["descripcion"].iloc[0] == "Lesión A"
    assert df["descripcion"].iloc[1] == "Lesión C"

def test_cargar_lesiones_empty_data():
    """Verifica que cargar_lesiones maneje data vacía."""
    mock_client = MockSupabaseClient()
    mock_client.set_data("lesiones", [])
    
    original_get_client = sys.modules['_get_client']
    sys.modules['_get_client'] = lambda: mock_client
    
    df = cargar_lesiones()
    
    sys.modules['_get_client'] = original_get_client
    
    assert df.empty

def test_cargar_lesiones_no_valid_data_after_filter():
    """Verifica que cargar_lesiones maneje caso donde todo se filtra."""
    mock_client = MockSupabaseClient()
    data = [
        {"fecha_inicio": None, "fecha_alta": "2022-02-15"},
        {"fecha_inicio": pd.NaT, "fecha_alta": pd.NaT},
    ]
    mock_client.set_data("lesiones", data)
    
    original_get_client = sys.modules['_get_client']
    sys.modules['_get_client'] = lambda: mock_client
    
    df = cargar_lesiones()
    
    sys.modules['_get_client'] = original_get_client
    
    assert df.empty
