"""
tests/test_audit_fixes.py — Suite de auditoría para app.py y db.py
Cubre las 6 fallas del informe. Usa únicamente stdlib + ast: sin Streamlit, sin Supabase.
Ejecutar: python3 -m unittest tests/test_audit_fixes.py -v
"""
import ast
import os
import unittest

# ── Cargar fuentes como texto (sin ejecutar) ──────────────────────────────────
_HERE   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APP_SRC  = open(os.path.join(_HERE, "app.py"),  encoding="utf-8").read()
DB_SRC   = open(os.path.join(_HERE, "db.py"),   encoding="utf-8").read()
APP_TREE = ast.parse(APP_SRC)
DB_TREE  = ast.parse(DB_SRC)


def _func_names(tree):
    return {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}


def _module_level_imports(tree):
    names = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            names.update(a.name for a in node.names)
        if isinstance(node, ast.ImportFrom):
            names.update(a.name for a in node.names)
    return names


def _imports_inside_functions(tree):
    found = []
    for fn in ast.walk(tree):
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for child in ast.walk(fn):
            if child is fn:
                continue
            if isinstance(child, ast.Import):
                for alias in child.names:
                    found.append((fn.name, alias.name))
            if isinstance(child, ast.ImportFrom):
                found.append((fn.name, child.module or ""))
    return found


# =============================================================================
# TASK 1 — Dead code & inline imports
# =============================================================================
class TestTask1DeadCodeAndImports(unittest.TestCase):

    def test_estado_from_score_removed(self):
        """_estado_from_score es dead code — debe eliminarse de app.py."""
        self.assertNotIn("_estado_from_score", _func_names(APP_TREE),
                         "_estado_from_score sigue definida; eliminar la función completa")

    def test_diving_load_not_imported_inside_function(self):
        """diving_load debe importarse en el header, no dentro de funciones."""
        bad = [(fn, mod) for fn, mod in _imports_inside_functions(APP_TREE)
               if "diving_load" in mod]
        self.assertEqual(bad, [],
                         f"Import de diving_load dentro de función: {bad}")

    def test_fuzzy_diving_not_imported_inside_function(self):
        """fuzzy_diving debe importarse en el header, no dentro de funciones."""
        bad = [(fn, mod) for fn, mod in _imports_inside_functions(APP_TREE)
               if "fuzzy_diving" in mod]
        self.assertEqual(bad, [],
                         f"Import de fuzzy_diving dentro de función: {bad}")

    def test_diving_imports_present_at_module_level(self):
        """carga_bruta_sesion y conjunto_dominante_ci deben estar en imports de módulo."""
        top = _module_level_imports(APP_TREE)
        self.assertIn("carga_bruta_sesion",    top)
        self.assertIn("conjunto_dominante_ci", top)


# =============================================================================
# TASK 2 — Eliminar matplotlib; usar fig_membership_fuzzy (Plotly)
# =============================================================================
class TestTask2MatplotlibRemoved(unittest.TestCase):

    def test_fig_membership_matplotlib_removed(self):
        """La función fig_membership (matplotlib) debe eliminarse."""
        self.assertNotIn("fig_membership", _func_names(APP_TREE))

    def test_matplotlib_not_imported(self):
        """matplotlib no debe aparecer en ningún import de app.py."""
        for node in ast.walk(APP_TREE):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    self.assertNotIn("matplotlib", alias.name)
            if isinstance(node, ast.ImportFrom):
                self.assertNotIn("matplotlib", str(node.module or ""))

    def test_fig_membership_fuzzy_imported_at_module_level(self):
        """fig_membership_fuzzy debe importarse en el header de app.py."""
        top = _module_level_imports(APP_TREE)
        self.assertIn("fig_membership_fuzzy", top)

    def test_st_pyplot_not_called(self):
        """st.pyplot no debe usarse."""
        for node in ast.walk(APP_TREE):
            if isinstance(node, ast.Attribute):
                if (node.attr == "pyplot"
                        and isinstance(node.ctx, ast.Load)
                        and isinstance(node.value, ast.Name)
                        and node.value.id == "st"):
                    self.fail("st.pyplot sigue presente — reemplazar por st.plotly_chart")


# =============================================================================
# TASK 3 — RBAC: Funciones de Pertenencia solo para rol analitico
# =============================================================================
class TestTask3RBACMembership(unittest.TestCase):

    def test_membership_expander_guarded_by_rol(self):
        """
        El expander 'Funciones de Pertenencia' debe estar dentro de un bloque
        if cuya condición contenga 'rol_usuario' y 'analitico'.
        """
        lines = APP_SRC.splitlines()
        expander_line = None
        for i, line in enumerate(lines):
            if "Funciones de Pertenencia" in line and "expander" in line:
                expander_line = i
                break
        self.assertIsNotNone(expander_line, "No se encontró el expander de Funciones de Pertenencia")

        context = "\n".join(lines[max(0, expander_line - 6): expander_line])
        self.assertIn("rol_usuario", context,
                      "Falta guardia 'rol_usuario' antes del expander de membresía")
        self.assertIn("analitico", context,
                      "Falta 'analitico' en la guardia antes del expander de membresía")


# =============================================================================
# TASK 4 — Cache invalidation tras guardar Wellness
# =============================================================================
class TestTask4WellnessCacheInvalidation(unittest.TestCase):

    def _scan_after_key(self, key, target):
        lines = APP_SRC.splitlines()
        in_btn = False
        for i, line in enumerate(lines):
            if key in line:
                in_btn = True
            if in_btn and target in line:
                return True
            if in_btn and i > 0 and line.startswith("def "):
                return False
        return False

    def test_cache_clear_after_wellness_save(self):
        """Debe haber st.cache_data.clear() después de btn_guardar_well."""
        self.assertTrue(
            self._scan_after_key("btn_guardar_well", "cache_data.clear"),
            "Falta st.cache_data.clear() después de guardar Wellness"
        )

    def test_rerun_after_wellness_save(self):
        """Debe haber st.rerun() después de btn_guardar_well."""
        self.assertTrue(
            self._scan_after_key("btn_guardar_well", "st.rerun"),
            "Falta st.rerun() después de guardar Wellness"
        )


# =============================================================================
# TASK 5 — Persistencia de Clavados
# =============================================================================
class TestTask5CargaSesionPersistence(unittest.TestCase):

    def test_insertar_carga_sesion_defined_in_db(self):
        """db.py debe definir insertar_carga_sesion."""
        self.assertIn("insertar_carga_sesion", _func_names(DB_TREE))

    def test_save_button_key_in_sub_carga(self):
        """app.py debe tener btn_guardar_carga."""
        self.assertIn("btn_guardar_carga", APP_SRC)

    def test_cache_clear_after_carga_save(self):
        """Debe haber st.cache_data.clear() después de btn_guardar_carga."""
        lines = APP_SRC.splitlines()
        in_btn = False
        for i, line in enumerate(lines):
            if "btn_guardar_carga" in line:
                in_btn = True
            if in_btn and "cache_data.clear" in line:
                return
            if in_btn and i > 0 and line.startswith("def "):
                break
        self.fail("Falta st.cache_data.clear() después de guardar carga de clavados")

    def test_cargas_sesion_table_used_in_db(self):
        """insertar_carga_sesion debe escribir en la tabla cargas_sesion."""
        self.assertIn("cargas_sesion", DB_SRC)

    def test_insertar_carga_sesion_returns_bool_tuple(self):
        """insertar_carga_sesion debe retornar (bool, str)."""
        func_lines, in_func = [], False
        for line in DB_SRC.splitlines():
            if "def insertar_carga_sesion" in line:
                in_func = True
            if in_func:
                func_lines.append(line)
                if len(func_lines) > 5 and line.startswith("def "):
                    break
        func_src = "\n".join(func_lines[:60])
        self.assertIn("False", func_src)
        self.assertIn("True",  func_src)


# =============================================================================
# TASK 6 — Performance: historial O(N³) → cached wrapper
# =============================================================================
class TestTask6HistorialCached(unittest.TestCase):

    def test_cached_batch_function_defined(self):
        """app.py debe definir calcular_historial_batch_cached."""
        self.assertIn("calcular_historial_batch_cached", _func_names(APP_TREE))

    def test_cached_batch_has_cache_data_decorator(self):
        """calcular_historial_batch_cached debe tener @st.cache_data."""
        for node in ast.walk(APP_TREE):
            if (isinstance(node, ast.FunctionDef)
                    and node.name == "calcular_historial_batch_cached"):
                deco_attrs = []
                for d in node.decorator_list:
                    if isinstance(d, ast.Attribute):
                        deco_attrs.append(d.attr)
                    elif isinstance(d, ast.Call) and isinstance(d.func, ast.Attribute):
                        deco_attrs.append(d.func.attr)
                self.assertIn("cache_data", deco_attrs,
                              "calcular_historial_batch_cached necesita @st.cache_data")
                return
        self.fail("calcular_historial_batch_cached no encontrada en el AST")

    def test_tab_dashboard_does_not_call_pipeline_historial_directly(self):
        """tab_dashboard no debe llamar a pipeline_historial directamente."""
        func_src, in_func = "", False
        for line in APP_SRC.splitlines():
            if "def tab_dashboard(" in line:
                in_func = True
            if in_func:
                func_src += line + "\n"
                if line.startswith("def ") and "tab_dashboard" not in line:
                    break
        self.assertNotIn("pipeline_historial(", func_src,
                         "tab_dashboard llama a pipeline_historial directamente (O(N³))")

    def test_tuple_used_for_atletas_arg(self):
        """atletas debe pasarse como tuple() para ser hashable por cache_data."""
        self.assertIn("tuple(", APP_SRC,
                      "atletas debe convertirse a tuple() para st.cache_data")


# =============================================================================
# REGRESIÓN — no romper lo que ya funcionaba
# =============================================================================
class TestRegression(unittest.TestCase):

    def test_app_syntax_valid(self):
        try:
            ast.parse(APP_SRC)
        except SyntaxError as e:
            self.fail(f"app.py sintaxis inválida: {e}")

    def test_db_syntax_valid(self):
        try:
            ast.parse(DB_SRC)
        except SyntaxError as e:
            self.fail(f"db.py sintaxis inválida: {e}")

    def test_wellness_insert_no_generated_column(self):
        """insertar_wellness no debe insertar w_norm (GENERATED ALWAYS AS)."""
        in_func, insert_lines, capturing = False, [], False
        for line in DB_SRC.splitlines():
            if "def insertar_wellness" in line:
                in_func = True
            if in_func and ".insert({" in line:
                capturing = True
                insert_lines = [line]
                continue
            if capturing:
                insert_lines.append(line)
                if "}).execute()" in line:
                    break
        insert_text = "\n".join(insert_lines)
        self.assertNotIn('"w_norm"', insert_text,
                         "insertar_wellness no debe insertar 'w_norm' (columna GENERATED)")

    def test_insertar_sesion_still_defined(self):
        """insertar_sesion no debe haberse eliminado."""
        self.assertIn("insertar_sesion", _func_names(DB_TREE))

    def test_insertar_wellness_still_defined(self):
        """insertar_wellness no debe haberse eliminado."""
        self.assertIn("insertar_wellness", _func_names(DB_TREE))


if __name__ == "__main__":
    unittest.main(verbosity=2)
