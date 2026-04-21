# AUDITORIA INICIAL - NMF-Optimizer

## Estado de Archivos Críticos

### 1. db.py
- [x] `_validar_numericos` existe y se usa.
- [x] `cargar_sesiones_raw` filtra NaN/NaT en 'fecha'.
- [x] `cargar_lesiones` filtra NaN/NaT en 'fecha_inicio' y 'fecha_alta'.

### 2. visualization/charts.py
- [x] `fig_vmp_tendencia` valida NaN/NaT.
- [x] `fig_semaforo_historico` valida NaN/NaT.
- [x] Sintaxis `if-elif` correcta en `fig_semaforo_historico`.

### 3. app.py
- [ ] `_cache_resource` silencia errores (L57-68). **Arreglo pendiente.**
- [ ] `_cache_data_ttl` silencia errores. **Arreglo pendiente.**
- [ ] `render_tab_lesiones` NO tiene try-except (L653). **Arreglo pendiente.**
- [ ] `tab_historial` NO tiene try-except (L656). **Arreglo pendiente.**
- [ ] `calcular_membresias_atleta` acceso inseguro a `_fat_v` (L123-128). **Arreglo pendiente.**
- [x] Importaciones principales verificadas.

## Errores Identificados (Fase 0)
1. **Sintaxis:** Error en `app.py` línea 340 (Corregido).
2. **Dependencias:** `scikit-fuzzy` (0.5.0) está instalado pero `diagnose.py` no lo detectó por error en el script.
3. **Fail Fast:** Cache wrappers en `app.py` ocultan problemas reales.
4. **Robustez UI:** Falta de try-except en el renderizado de pestañas nuevas.

## Plan de Acción Inmediato
1. Ejecutar **Checkpoint 1.2**: Arreglos críticos en `app.py`.
2. Ejecutar **Checkpoint 1.3**: Validación de plotting functions restantes.
