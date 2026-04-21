# NMF-Optimizer v4.4.1 (Revitalized)

**Estado:** 🟢 Estable | **Tests:** 89/89 PASSED

## Novedades v4.4.1
- **Estabilidad Total:** Manejo de NaN/NaT en todos los gráficos y tablas.
- **Fail-Fast Cache:** Eliminación de fugas de memoria por fallos silenciosos de caché.
- **Robustez UI:** Error handling independiente para pestañas de Lesiones e Historial.
- **Auditoría Pasada:** Cumple con el 100% de los requisitos de `test_audit_fixes.py`.

## Features
*   **Data Entry:** VMP (individual/group/CSV), Wellness (sliders/mass), Carga Grupal (training load).
*   **Dashboard:** Group/Individual analysis (fatigue, KPIs, VMP trends).
*   **Lesiones:** Injury tracking.
*   **Historial:** View/edit data.
*   **Config:** User roles, mesocycle settings (Analítico/Operativo).

## Instalación
```bash
pip install -r requirements.txt
```

## Configuración
Asegurar credenciales de Supabase en `.streamlit/secrets.toml`.

## Ejecución
```bash
streamlit run app.py
```

## Tests
```bash
python -m unittest discover tests/
```

## Database Schema (Supabase)
Tablas requeridas: `atletas`, `sesiones_vmp`, `wellness`, `cargas_grupales`, `cargas_grupales_atletas`, `lesiones`.
