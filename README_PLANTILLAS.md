# Plantillas CSV — Reportes Semanales y Mensuales
## NMF-Optimizer · Club Tornados

Todas las plantillas admiten carga masiva desde la pestaña **📤 Importar CSV** del dashboard.  
Formato de fecha: `YYYY-MM-DD`. Separador: coma (`,`). Codificación: UTF-8.

---

## 1. `plantilla_vmp_sesiones.csv` — VMP Individual por Sesión

Uso diario o por sesión de entrenamiento.

```
Nombre,Fecha,VMP_Hoy,Notas
Juanes,2025-03-15,0.523,Test matutino
Maria,2025-03-15,0.481,Post-viaje
Carlos,2025-03-16,0.510,
```

| Campo | Tipo | Rango | Requerido |
|-------|------|-------|-----------|
| Nombre | texto | — | ✅ |
| Fecha | fecha (YYYY-MM-DD) | ≤ hoy | ✅ |
| VMP_Hoy | decimal | 0.100–2.500 m/s | ✅ |
| Notas | texto | — | ❌ |

**Alias detectados automáticamente:**
- `Nombre` → atleta, deportista, jugador
- `Fecha` → date, fecha_sesion
- `VMP_Hoy` → vmp, velocidad, vel_media, mean_velocity

---

## 2. `plantilla_wellness_diario.csv` — Cuestionario Hooper (diario)

Un registro por atleta por día. La escala es **inversa** para Sueño/Fatiga/Estrés/Dolor (1=óptimo) y **directa** para Humor (7=óptimo).

```
Nombre,Fecha,Sueno,Fatiga,Estres,Dolor,Humor,Notas
Juanes,2025-03-15,2,3,2,1,6,Post-partido
Maria,2025-03-15,4,5,3,3,4,
Carlos,2025-03-16,1,2,1,1,7,
```

| Campo | Tipo | Escala | Requerido |
|-------|------|--------|-----------|
| Nombre | texto | — | ✅ |
| Fecha | fecha | ≤ hoy | ✅ |
| Sueno | entero | 1–7 (1=óptimo) | ✅ |
| Fatiga | entero | 1–7 (1=óptimo) | ✅ |
| Estres | entero | 1–7 (1=óptimo) | ✅ |
| Dolor | entero | 1–7 (1=sin dolor) | ✅ |
| Humor | entero | 1–7 (7=óptimo) | ✅ |
| Notas | texto | — | ❌ |

**W_norm se calcula automáticamente** en el servidor al importar.

---

## 3. `plantilla_resumen_semanal.csv` — Resumen Semanal de VMP

Para reportes de entrenador con promedios calculados externamente (Excel/Google Sheets).  
El sistema lo trata como una sesión representativa con fecha = último día de la semana.

```
Nombre,Semana,Fecha_Fin,N_Sesiones,VMP_Promedio,VMP_Minimo,VMP_Maximo,Notas
Juanes,1,2025-01-12,4,0.523,0.498,0.551,Pre-temporada
Maria,1,2025-01-12,3,0.481,0.462,0.495,Regresó de lesión
Carlos,1,2025-01-12,5,0.510,0.495,0.528,
```

| Campo | Tipo | Descripción |
|-------|------|-------------|
| Nombre | texto | Atleta |
| Semana | entero | Número de semana del mesociclo (1–12) |
| Fecha_Fin | fecha | Último día de la semana reportada |
| N_Sesiones | entero | Sesiones completadas esa semana |
| VMP_Promedio | decimal (m/s) | Media semanal — se importa como VMP_Hoy |
| VMP_Minimo | decimal (m/s) | Referencia diagnóstica (no se importa al modelo) |
| VMP_Maximo | decimal (m/s) | Referencia diagnóstica |
| Notas | texto | Contexto del entrenador |

> **Nota de importación:** El sistema toma `VMP_Promedio` → `VMP_Hoy` y `Fecha_Fin` → `Fecha`.

---

## 4. `plantilla_resumen_mensual.csv` — Resumen Mensual / Mesociclo

Para análisis longitudinal. Un registro por atleta por mes.

```
Nombre,Mes,Anio,Fecha_Referencia,N_Sesiones,VMP_Promedio,W_norm_Promedio,ACWR_Promedio,Notas
Juanes,3,2025,2025-03-31,16,0.519,0.72,1.05,Mesociclo de fuerza
Maria,3,2025,2025-03-31,12,0.475,0.65,0.98,Post-lesión parcial
```

| Campo | Tipo | Descripción |
|-------|------|-------------|
| Nombre | texto | Atleta |
| Mes | entero | Mes (1–12) |
| Anio | entero | Año |
| Fecha_Referencia | fecha | Último día del mes (para ordenar cronológicamente) |
| N_Sesiones | entero | Total sesiones en el mes |
| VMP_Promedio | decimal | Media mensual VMP (m/s) |
| W_norm_Promedio | decimal [0–1] | Media mensual Wellness normalizado |
| ACWR_Promedio | decimal | ACWR medio del mes (referencia) |
| Notas | texto | Resumen del entrenador |

> Este resumen **no se importa directamente al modelo** — es un archivo de análisis histórico descargable desde la tabla de resultados.

---

## 5. `plantilla_carga_clavados.csv` — Carga de Entrenamiento (Clavados)

```
Nombre,Fecha,N_Clavados,Altura_Promedio,DD_Promedio,Tipo_Predominante,L_Norm,W_norm,CI,Notas
Juanes,2025-03-15,12,7.5,2.8,HEAD,65.4,0.72,82.1,Sesión técnica
Maria,2025-03-15,8,5.0,2.2,PIKE,38.2,0.58,55.8,
```

| Campo | Tipo | Rango | Descripción |
|-------|------|-------|-------------|
| Nombre | texto | — | Atleta |
| Fecha | fecha | — | Fecha de la sesión |
| N_Clavados | entero | 1–30 | Clavados ejecutados |
| Altura_Promedio | decimal | 1.0–10.0 m | Altura media FINA |
| DD_Promedio | decimal | 1.2–4.4 | Grado dificultad FINA medio |
| Tipo_Predominante | texto | HEAD/FEET/TWIST/PIKE/SYNC | Tipo más frecuente |
| L_Norm | decimal | 0–100 | Carga normalizada (calculada externamente) |
| W_norm | decimal | 0–1 | Wellness normalizado Hooper |
| CI | decimal | 0–200 | Carga Integrada = L_norm × (2 − W_norm) |
| Notas | texto | — | Contexto de la sesión |

---

## Guía Rápida de Errores Comunes

| Error | Causa | Solución |
|-------|-------|----------|
| "Columnas faltantes" | Nombre de columna distinto | Usar alias listados o nombres exactos |
| "VMP fuera de rango" | VMP > 2.50 m/s | Verificar sensor; rango válido: 0.10–2.50 |
| "Hooper fuera de rango" | Valor fuera de 1–7 | Todos los ítems Hooper deben ser enteros 1–7 |
| "Fecha inválida" | Formato incorrecto | Usar YYYY-MM-DD (ej. 2025-03-15) |
| "nombre vacío" | Celda en blanco | Completar nombre del atleta en cada fila |
