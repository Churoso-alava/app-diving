#!/bin/bash

set -e

# FASE 1: Crear db.py (shim) y utils/__init__.py
# Blocker phase para desbloquear el resto del plan

echo "=== FASE 1: Crear db.py shim y utils/__init__.py ==="
echo ""

# Verificar que estamos en el repo correcto
if [ ! -d "data" ]; then
    echo "ERROR: No se encuentra el directorio 'data'. ¿Estás en el root del repositorio?"
    exit 1
fi

echo "✓ Directorio 'data' encontrado"
echo ""

# 1. Crear db.py en raíz como shim
echo "1. Creando db.py como shim de re-exportación..."
cat > db.py << 'EOF'
"""
Shim module for test compatibility.
Re-exports data.db module to satisfy v4.3 test suite expectations.
Tests import from root-level db.py; production imports from data.db.
"""

from data.db import (
    MAX_IMPORT_ROWS,
    cargar_atletas,
    cargar_sesiones_raw,
    insertar_sesion,
    insertar_wellness,
    insertar_carga_grupal_batch,
    importar_dataframe,
    importar_wellness_dataframe,
    wellness_masivo_template,
)

__all__ = [
    "MAX_IMPORT_ROWS",
    "cargar_atletas",
    "cargar_sesiones_raw",
    "insertar_sesion",
    "insertar_wellness",
    "insertar_carga_grupal_batch",
    "importar_dataframe",
    "importar_wellness_dataframe",
    "wellness_masivo_template",
]
EOF

echo "✓ db.py creado en raíz"
echo ""

# 2. Crear utils/__init__.py
echo "2. Creando utils/__init__.py..."
mkdir -p utils
cat > utils/__init__.py << 'EOF'
"""
Utils package placeholder for test compatibility.
"""
EOF

echo "✓ utils/__init__.py creado"
echo ""

# 3. Validación: importar db y verificar MAX_IMPORT_ROWS
echo "3. Validando importación y MAX_IMPORT_ROWS..."
python3 << 'PYEOF'
import sys
try:
    import db
    assert hasattr(db, 'MAX_IMPORT_ROWS'), "MAX_IMPORT_ROWS not found in db module"
    assert db.MAX_IMPORT_ROWS == 500, f"MAX_IMPORT_ROWS is {db.MAX_IMPORT_ROWS}, expected 500"
    print("✓ db.MAX_IMPORT_ROWS == 500")
except Exception as e:
    print(f"✗ Validation failed: {e}")
    sys.exit(1)
PYEOF

if [ $? -ne 0 ]; then
    echo "ERROR: Validación fallida"
    exit 1
fi
echo ""

# 4. Validación: importar utils
echo "4. Validando importación de utils..."
python3 -c "import utils" || (echo "ERROR: No se puede importar utils"; exit 1)
echo "✓ utils importado exitosamente"
echo ""

# 5. Mostrar estado de archivos
echo "5. Estado de archivos creados:"
echo ""
echo "db.py (raíz):"
wc -l db.py | awk '{print "  " $1 " líneas"}'
echo ""
echo "utils/__init__.py:"
wc -l utils/__init__.py | awk '{print "  " $1 " líneas"}'
echo ""

# 6. Gate test: python -c 'import db; assert db.MAX_IMPORT_ROWS == 500'
echo "6. Ejecutando gate test oficial..."
python3 -c "import db; assert db.MAX_IMPORT_ROWS == 500" && echo "✓ Gate test PASSED" || exit 1
echo ""

# 7. Gate test: python -c 'import utils'
echo "7. Ejecutando gate test para utils..."
python3 -c "import utils" && echo "✓ Gate test utils PASSED" || exit 1
echo ""

# 8. Resumen y commit
echo "=== FASE 1 COMPLETADA ==="
echo ""
echo "Archivos creados:"
echo "  - db.py (shim de re-exportación)"
echo "  - utils/__init__.py (package vacío)"
echo ""
echo "Gate tests: PASSED"
echo ""
echo "Próximo paso:"
echo "  git add db.py utils/__init__.py"
echo "  git commit -m 'Phase 1: Create db.py shim and utils/__init__.py'"
echo ""
