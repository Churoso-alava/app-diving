import sys
import os
from pathlib import Path

# Añadir la raíz del proyecto al PYTHONPATH para todos los tests
root_dir = Path(__file__).parent.parent.absolute()
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))
