# Registro de Deuda Técnica y Oportunidades de Mejora

Este documento rastrea las fallas detectadas, discrepancias estadísticas y oportunidades de mejora para el sistema "Gem".

| ID | Fecha | Componente | Descripción de la Falla / Problema | Estado | Impacto | Oportunidad de Mejora |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **001** | 2024-05-22 | `z_meso` | Distribución de datos no normal en atletas (Test Shapiro-Wilk: P < 0.05). | **COMPLETADO** | `z_meso` (Z-score) estadísticamente sesgado para ciertos atletas. | Migrado de Z-score estándar a MAD (Median Absolute Deviation). |
| **002** | 2024-05-22 | Lógica Fuzzy | Falta de ponderación dinámica del peso de las variables según la confianza estadística. | PENDIENTE | El motor procesa valores sesgados como verdades absolutas. | Implementar atenuación de variables basada en flags de confianza estadística en `evaluar_atleta`. |
| **003** | 2024-05-22 | DQI | El DQI mide cantidad/densidad, no validez estadística (normalidad) de los datos. | PENDIENTE | Un DQI "Alto" puede dar falsa confianza en métricas calculadas sobre distribuciones aberrantes. | Vincular el DQI al resultado del test de normalidad para degradar la calidad si la distribución es aberrante. |
| **004** | 2024-05-22 | ACWR | ACWR basado puramente en velocidad interna. | PENDIENTE | Ignora la acumulación pura de volumen (riesgo de lesiones por sobreuso). | Añadir un "ACWR de Carga Externa" como variable secundaria o un limitador de volumen si el DQI es bajo. |
| **005** | 2024-05-22 | Regresión Lineal | La regresión lineal (`beta_aguda`/`beta_28`) asume fatiga lineal, suavizando cambios abruptos. | **COMPLETADO** | Riesgo de retrasar la detección de fatiga aguda/traumática por suavización excesiva. | Implementada regresión robusta (Theil-Sen) en `core/services.py`. |

---
