# AppDivingCodex (v4.5)

Sistema modular y refactorizado para el monitoreo de carga interna y análisis de fatiga en atletas.

## 🏗️ Arquitectura
- `/core`: Lógica de negocio (biomecánica, motor difuso Mamdani).
- `/data`: Capa de persistencia (Supabase).
- `/ui`: Interfaz de usuario (Streamlit) y visualizaciones (Plotly).
- `/components`: Componentes UI reutilizables.
- `/docs`: Documentación técnica y de proyecto.
  - `/planning`: Planes de implementación y especificaciones.
  - `/science`: Fundamentos fisiológicos y científicos.
  - `/audits`: Reportes de auditoría y validación.
  - `/project-health`: Logs de deuda técnica y mejora.
  - `/migrations`: Scripts de base de datos.
- `/tests`: Pruebas unitarias e integrales.

## 🚀 Instalación
1. Clonar el repositorio.
2. Instalar dependencias:
   ```bash
   pip install -r requirements.txt
   ```
3. Configurar variables de entorno o `.streamlit/secrets.toml` con las credenciales de Supabase (`SUPABASE_URL`, `SUPABASE_KEY`).

## 💻 Ejecución
```bash
streamlit run app.py
```

## 🧪 Tests
Para ejecutar las pruebas:
```bash
python -m pytest tests/
```

---
*Mantenido por el equipo de desarrollo de AppDivingCodex.*
