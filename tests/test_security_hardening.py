"""
tests/test_security_hardening.py
NMF-Optimizer v4.3 — Security Hardening Tests
"""
import re
import pandas as pd
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# Task 1 — Dependency pinning (V-DEPS)
# ─────────────────────────────────────────────────────────────────────────────

PINNED_DEPS = {
    "streamlit":    "1.38.0",
    "supabase":     "2.4.6",
    "scikit-fuzzy": "0.4.2",
    "scipy":        "1.11.4",
    "pandas":       "2.1.4",
    "numpy":        "1.26.4",
    "matplotlib":   "3.7.3",
    "openpyxl":     "3.1.2",
    "networkx":     "3.2.1",
    "plotly":       "5.18.0",
}


def _parse_requirements() -> dict[str, tuple[str, str]]:
    """Devuelve {nombre_paquete: (operador, version)} del requirements.txt raíz."""
    req_path = Path(__file__).parent.parent / "requirements.txt"
    result: dict[str, tuple[str, str]] = {}
    for line in req_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = re.match(r"^([A-Za-z0-9_\-]+)\s*([>=<~!]+)\s*(.+)$", line)
        if m:
            result[m.group(1).lower()] = (m.group(2), m.group(3))
    return result


class TestDependencyPinning:
    def test_all_deps_use_exact_pin(self):
        """Ninguna dependencia debe usar >= en producción."""
        parsed = _parse_requirements()
        loose = [name for name, (op, _ver) in parsed.items() if op != "=="]
        assert loose == [], f"Dependencias con versión suelta (deben usar ==): {loose}"

    def test_pinned_versions_match_audit(self):
        """Versiones auditadas coinciden exactamente con requirements.txt."""
        parsed = _parse_requirements()
        mismatches = []
        for pkg, expected_ver in PINNED_DEPS.items():
            entry = parsed.get(pkg.lower())
            if entry is None:
                mismatches.append(f"{pkg}: NOT FOUND en requirements.txt")
            elif entry[1] != expected_ver:
                mismatches.append(f"{pkg}: expected=={expected_ver}, got {entry[0]}{entry[1]}")
        assert mismatches == [], "\n".join(mismatches)


# ─────────────────────────────────────────────────────────────────────────────
# Task 2 — Import row limit backstop (V-DOS)
# ─────────────────────────────────────────────────────────────────────────────

MAX_IMPORT_ROWS = 500   # espejo de la constante en db.py


class TestImportRowLimit:
    """
    Verifica que db.importar_dataframe y db.importar_wellness_dataframe
    rechacen DataFrames que superen MAX_IMPORT_ROWS sin tocar la BD.
    """

    def _make_vmp_df(self, n: int) -> pd.DataFrame:
        return pd.DataFrame({
            "Nombre":  [f"Atleta_{i}" for i in range(n)],
            "Fecha":   ["2026-01-01"] * n,
            "VMP_Hoy": [0.85] * n,
        })

    def _make_wellness_df(self, n: int) -> pd.DataFrame:
        return pd.DataFrame({
            "Nombre":  [f"Atleta_{i}" for i in range(n)],
            "Fecha":   ["2026-01-01"] * n,
            "Sueno":   [3] * n,
            "Fatiga":  [3] * n,
            "Estres":  [3] * n,
            "Dolor":   [3] * n,
            "Humor":   [4] * n,
            "Notas":   [""] * n,
        })

    def test_vmp_import_rejects_oversized_df(self):
        """importar_dataframe con >500 filas devuelve (0, 0, [error]) sin tocar BD."""
        import db
        df_big = self._make_vmp_df(MAX_IMPORT_ROWS + 1)
        insertados, omitidos, errores = db.importar_dataframe(df_big)
        assert insertados == 0
        assert omitidos == 0
        assert len(errores) == 1
        assert "501" in errores[0] or "500" in errores[0] or "límite" in errores[0].lower()

    def test_vmp_import_accepts_max_rows(self, monkeypatch):
        """importar_dataframe con exactamente 500 filas NO debe rechazarse."""
        import db
        monkeypatch.setattr(db, "insertar_sesion", lambda *a, **kw: (True, "ok"))
        df_ok = self._make_vmp_df(MAX_IMPORT_ROWS)
        insertados, omitidos, errores = db.importar_dataframe(df_ok)
        limit_errors = [e for e in errores if "límite" in e.lower() or str(MAX_IMPORT_ROWS) in e]
        assert limit_errors == []

    def test_wellness_import_rejects_oversized_df(self):
        """importar_wellness_dataframe con >500 filas devuelve (0, 0, [error])."""
        import db
        df_big = self._make_wellness_df(MAX_IMPORT_ROWS + 1)
        insertados, omitidos, errores = db.importar_wellness_dataframe(df_big)
        assert insertados == 0
        assert omitidos == 0
        assert len(errores) == 1
        assert "501" in errores[0] or "500" in errores[0] or "límite" in errores[0].lower()


# ─────────────────────────────────────────────────────────────────────────────
# Task 3 — UI guard in app.py (V-DOS UI)
# ─────────────────────────────────────────────────────────────────────────────

class TestImportUIGuard:
    def test_app_imports_max_import_rows_from_db(self):
        """app.py debe referenciar db.MAX_IMPORT_ROWS (única fuente de verdad)."""
        app_src = (Path(__file__).parent.parent / "app.py").read_text()
        assert "db.MAX_IMPORT_ROWS" in app_src or "MAX_IMPORT_ROWS" in app_src, (
            "app.py no referencia MAX_IMPORT_ROWS — el guard UI podría estar ausente"
        )

    def test_db_max_import_rows_is_500(self):
        """Confirma el valor exacto de la constante."""
        import db
        assert db.MAX_IMPORT_ROWS == 500
