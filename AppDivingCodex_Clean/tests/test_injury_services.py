import pytest
import pandas as pd
from datetime import date
from core.services import (
    registrar_lesion_servicio,
    obtener_lesiones_activas_servicio,
    obtener_historial_lesiones_servicio,
    actualizar_estado_lesion_servicio
)
# We will mock the database functions in future tasks.
