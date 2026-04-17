#!/bin/bash

set -e

# FASE 2: Dependency Pinning
# Reescribir requirements.txt con == pins exactos
# Actualizar PINNED_DEPS en tests/test_security_hardening.py con pillow

echo "=== FASE 2: Dependency Pinning ==="
echo ""

# Verificar que estamos en el repo correcto
if [ ! -f "requirements.txt" ]; then
    echo "ERROR: No se encuentra requirements.txt. ¿Estás en el root del repositorio?"
    exit 1
fi

echo "✓ requirements.txt encontrado"
echo ""

# 1. Hacer backup de requirements.txt original
echo "1. Haciendo backup de requirements.txt..."
cp requirements.txt requirements.txt.backup
echo "✓ Backup creado en requirements.txt.backup"
echo ""

# 2. Reescribir requirements.txt con pins exactos
echo "2. Reescribiendo requirements.txt con pins exactos (==)..."
cat > requirements.txt << 'EOF'
streamlit==1.38.0
supabase==2.4.6
scikit-fuzzy==0.4.2
scipy==1.11.4
pandas==2.1.4
numpy==1.26.4
matplotlib==3.7.3
openpyxl==3.1.2
networkx==3.2.1
plotly==5.18.0
pillow==10.3.0
EOF

echo "✓ requirements.txt reescrito con 11 dependencias con == pins"
echo ""

# 3. Mostrar requirements.txt reescrito
echo "3. Contenido de requirements.txt:"
echo ""
cat requirements.txt
echo ""

# 4. Verificar que tests/test_security_hardening.py existe
echo "4. Verificando tests/test_security_hardening.py..."
if [ ! -f "tests/test_security_hardening.py" ]; then
    echo "ERROR: No se encuentra tests/test_security_hardening.py"
    exit 1
fi
echo "✓ tests/test_security_hardening.py encontrado"
echo ""

# 5. Actualizar PINNED_DEPS en test_security_hardening.py
echo "5. Actualizando PINNED_DEPS en test_security_hardening.py..."

# Crear el nuevo contenido de PINNED_DEPS
python3 << 'PYEOF'
import re

with open('tests/test_security_hardening.py', 'r') as f:
    content = f.read()

# Buscar el bloque PINNED_DEPS
pinned_deps_pattern = r'(PINNED_DEPS\s*=\s*\{[^}]*?\})'

# Nueva definición con pillow incluido
new_pinned_deps = """PINNED_DEPS = {
    'streamlit': '1.38.0',
    'supabase': '2.4.6',
    'scikit-fuzzy': '0.4.2',
    'scipy': '1.11.4',
    'pandas': '2.1.4',
    'numpy': '1.26.4',
    'matplotlib': '3.7.3',
    'openpyxl': '3.1.2',
    'networkx': '3.2.1',
    'plotly': '5.18.0',
    'pillow': '10.3.0',
}"""

# Reemplazar el bloque PINNED_DEPS
if re.search(pinned_deps_pattern, content, re.DOTALL):
    new_content = re.sub(pinned_deps_pattern, new_pinned_deps, content, flags=re.DOTALL)
    with open('tests/test_security_hardening.py', 'w') as f:
        f.write(new_content)
    print("✓ PINNED_DEPS actualizado con 11 paquetes incluyendo pillow")
else:
    print("⚠ No se encontró el bloque PINNED_DEPS. Verificando estructura...")
    # Fallback: buscar cualquier referencia a PINNED_DEPS y advertir
    if 'PINNED_DEPS' in content:
        print("⚠ PINNED_DEPS existe pero no coincide con el patrón esperado")
        print("⚠ Actualización manual requerida en tests/test_security_hardening.py")
        exit(1)
PYEOF

if [ $? -ne 0 ]; then
    echo "ERROR: No se pudo actualizar PINNED_DEPS"
    exit 1
fi
echo ""

# 6. Verificar que el archivo fue actualizado
echo "6. Verificando actualización de PINNED_DEPS..."
if grep -q "'pillow': '10.3.0'" tests/test_security_hardening.py; then
    echo "✓ pillow añadido a PINNED_DEPS"
else
    echo "ERROR: pillow no encontrado en PINNED_DEPS después de actualización"
    exit 1
fi
echo ""

# 7. Mostrar PINNED_DEPS actualizado (primeras líneas)
echo "7. Verificando PINNED_DEPS en test_security_hardening.py:"
echo ""
python3 << 'PYEOF'
import re
with open('tests/test_security_hardening.py', 'r') as f:
    content = f.read()
# Extraer y mostrar PINNED_DEPS
match = re.search(r'(PINNED_DEPS\s*=\s*\{[^}]*?\})', content, re.DOTALL)
if match:
    pinned_deps_text = match.group(1)
    # Mostrar solo las primeras 300 caracteres
    print(pinned_deps_text[:300])
    print("...")
PYEOF

echo ""

# 8. Gate test: verificar que todas las dependencias en requirements.txt tienen ==
echo "8. Validando que todas las dependencias tienen == (no ~=, >=, etc)..."
python3 << 'PYEOF'
import sys

with open('requirements.txt', 'r') as f:
    lines = [line.strip() for line in f if line.strip() and not line.startswith('#')]

bad_lines = [line for line in lines if '==' not in line]
if bad_lines:
    print(f"ERROR: Las siguientes líneas no usan ==:")
    for line in bad_lines:
        print(f"  {line}")
    sys.exit(1)

print("✓ Todas las dependencias usan == exactamente")

# Contar dependencias
print(f"✓ Total de dependencias pinneadas: {len(lines)}")
PYEOF

if [ $? -ne 0 ]; then
    echo "ERROR: Validación de pins fallida"
    exit 1
fi
echo ""

# 9. Gate test oficial: pytest test_security_hardening.py::TestDependencyPinning -v
echo "9. Ejecutando gate test oficial: TestDependencyPinning..."
python3 -m pytest tests/test_security_hardening.py::TestDependencyPinning -v 2>&1 | head -50

if python3 -m pytest tests/test_security_hardening.py::TestDependencyPinning -v >/dev/null 2>&1; then
    echo ""
    echo "✓ Gate test TestDependencyPinning PASSED"
else
    echo ""
    echo "ERROR: Gate test TestDependencyPinning FAILED"
    exit 1
fi
echo ""

# 10. Resumen
echo "=== FASE 2 COMPLETADA ==="
echo ""
echo "Cambios realizados:"
echo "  - requirements.txt: reescrito con 11 paquetes con == pins exactos"
echo "  - tests/test_security_hardening.py: PINNED_DEPS actualizado con pillow"
echo ""
echo "Gate test: TestDependencyPinning PASSED"
echo ""
echo "Próximo paso:"
echo "  git add requirements.txt tests/test_security_hardening.py"
echo "  git commit -m 'Phase 2: Pin all requirements to exact versions; add pillow to audit dict'"
echo ""
