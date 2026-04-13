"""
logic/services.py — NMF-Optimizer v4.4
Capa de lógica de negocio. Sin dependencias de Streamlit ni base de datos.

Responsabilidades:
  - Calcular las 5 variables de entrada del motor Mamdani a partir de sesiones_vmp.
  - Manejar nulos por días sin entrenamiento (reindex diario con rolling windows).
  - Exponer pipeline_diagnostico() como punto de entrada unificado.

Variables de entrada del motor Mamdani v4.1:
  1. acwr       — Promedio VMP 7d / Promedio VMP 28d
  2. delta_pct  — Variación % VMP hoy vs MMC28
  3. z_meso     — Z-Score dentro del mesociclo (últimos 28d)
  4. beta_aguda — Pendiente regresión lineal VMP en ventana 7d
  5. beta_28    — Pendiente regresión lineal VMP en ventana 28d
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTES
# ─────────────────────────────────────────────────────────────────────────────

# Data Quality Index — pesos y referencias
_DQI_W7:  float = 0.40
_DQI_W28: float = 0.60
_REF_7D:  int   = 3      # sesiones mínimas en 7d para DQI perfecto
_REF_28D: int   = 12     # sesiones mínimas en 28d para DQI perfecto

# Tolerancia para min_periods de rolling (50% de la frecuencia esperada)
_TOL: float = 0.50

# Clamp de pendiente: evitar que gaps largos generen slopes explosivos
_GAP_MIN: float = 1.0
_GAP_MAX: float = 7.0

# Rango fisiológico VMP (m/s)
VMP_MIN: float = 0.100
VMP_MAX: float = 2.500


# ─────────────────────────────────────────────────────────────────────────────
# VALIDACIÓN DE ENTRADA
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SessionInput:
    """Valida una sesión antes de insertar en la base de datos."""
    nombre: str
    fecha:  str      # ISO 8601: "YYYY-MM-DD"
    vmp:    float
    notas:  str = field(default="")

    _VMP_MIN: float = field(default=VMP_MIN, init=False, repr=False)
    _VMP_MAX: float = field(default=VMP_MAX, init=False, repr=False)

    def __post_init__(self) -> None:
        errors: list[str] = []
        if not self.nombre or not self.nombre.strip():
            errors.append("nombre no puede estar vacío")
        if not (self._VMP_MIN <= self.vmp <= self._VMP_MAX):
            errors.append(
                f"VMP {self.vmp:.3f} fuera de rango fisiológico "
                f"[{self._VMP_MIN}, {self._VMP_MAX}] m/s"
            )
        try:
            pd.Timestamp(self.fecha)
        except Exception:
            errors.append(f"fecha inválida: '{self.fecha}'")
        if errors:
            raise ValueError("SessionInput inválido: " + "; ".join(errors))

    def to_dict(self) -> dict:
        return {
            "nombre": self.nombre.strip(),
            "fecha":  self.fecha,
            "vmp":    self.vmp,
            "notas":  self.notas.strip(),
        }


# ─────────────────────────────────────────────────────────────────────────────
# FASE 2 — CÁLCULO DE MÉTRICAS TEMPORALES
# ─────────────────────────────────────────────────────────────────────────────

def calcular_metricas(
    df: pd.DataFrame,
    atleta: str,
    ventana_meso: int = 28,
    perfil: dict | None = None,
) -> dict | None:
    """
    Calcula las 5 variables de entrada del motor Mamdani y métricas auxiliares.

    Parámetros
    ----------
    df          : DataFrame de sesiones_vmp con columnas snake_case
                  (nombre, fecha, vmp_hoy). Origen: data.db.cargar_sesiones_raw().
    atleta      : Nombre exacto del atleta (case-sensitive).
    ventana_meso: Días del mesociclo para Z-Score (default 28).
    perfil      : Dict opcional con {"edad": int}.

    Retorna None si el atleta tiene < 4 sesiones.

    Manejo de nulos (FASE 5):
    - Se reindexan las fechas al calendario diario continuo.
    - Los días sin entrenamiento quedan como NaN.
    - Los rolling windows usan min_periods adaptativo → nunca fallan por NaN.
    - polyfit se aplica solo sobre valores no-NaN.
    """
    # ── Filtrar y ordenar ─────────────────────────────────────────────────────
    sub = (
        df[df["nombre"] == atleta]
        .copy()
        .sort_values("fecha")
    )
    # Agrupar por fecha: máxima VMP del día (doble sesión)
    sub = (
        sub.groupby("fecha", as_index=False)["vmp_hoy"]
        .max()
        .sort_values("fecha")
        .reset_index(drop=True)
    )

    n = len(sub)
    if n < 4:
        log.debug("Atleta '%s': %d sesiones, mínimo 4.", atleta, n)
        return None

    # ── Serie temporal diaria (Fase 5: manejo de nulos) ───────────────────────
    idx        = pd.to_datetime(sub["fecha"])
    vmp_series = pd.Series(sub["vmp_hoy"].values, index=idx, dtype=float)

    # Reindex a calendario diario continuo → NaN donde no hubo sesión
    date_range = pd.date_range(start=idx.min(), end=idx.max(), freq="D")
    vmp_daily  = vmp_series.reindex(date_range)

    last_vmp  = float(vmp_series.iloc[-1])
    last_date = idx.max()
    hoy       = pd.Timestamp.today().normalize()

    # ── min_periods adaptativos basados en frecuencia de entrenamiento ────────
    dias_span = max((last_date - idx.min()).days, 1)
    freq_day  = n / dias_span                    # sesiones/día promedio
    mp_7d  = max(1, round(max(1, freq_day * 7)  * _TOL))
    mp_28d = max(2, round(max(2, freq_day * 28) * _TOL))

    # ── ACWR: MMA7 / MMC28 ───────────────────────────────────────────────────
    mma7_s  = vmp_daily.rolling("7D",  min_periods=mp_7d).mean()
    mmc28_s = vmp_daily.rolling("28D", min_periods=mp_28d).mean()

    mma7  = float(mma7_s.iloc[-1])  if not pd.isna(mma7_s.iloc[-1])  else last_vmp
    mmc28 = float(mmc28_s.iloc[-1]) if not pd.isna(mmc28_s.iloc[-1]) else last_vmp

    acwr = float(np.clip(mma7 / mmc28 if mmc28 > 0 else 1.0, 0.50, 1.80))

    # ── Delta %: variación VMP hoy vs MMC28 ─────────────────────────────────
    delta_pct = float(
        np.clip(((mmc28 - last_vmp) / mmc28) * 100 if mmc28 > 0 else 0.0, -20, 40)
    )

    # ── Z-Score mesociclo (últimos ventana_meso días) ────────────────────────
    cutoff_meso = last_date - pd.Timedelta(days=ventana_meso - 1)
    win_meso    = vmp_daily[vmp_daily.index >= cutoff_meso].dropna()
    if len(win_meso) >= 4 and float(win_meso.std()) > 0:
        z_meso = (last_vmp - float(win_meso.mean())) / float(win_meso.std())
    else:
        z_meso = 0.0
    z_meso = float(np.clip(z_meso, -4.0, 4.0))

    # ── Pendientes vía polyfit (Fase 5: solo valores no-NaN) ─────────────────
    def _pendiente_calendar(dias_back: int, min_n: int) -> float:
        """
        Regresión lineal sobre la ventana [last_date - dias_back, last_date].
        Usa solo días CON sesión (dropna). Retorna 0.0 si hay < min_n puntos.
        Resultado en unidades de VMP/sesión (escalado por gap promedio).
        """
        cutoff = vmp_daily.index[-1] - pd.Timedelta(days=dias_back - 1)
        win    = vmp_daily[vmp_daily.index >= cutoff].dropna()
        if len(win) < min_n:
            return 0.0
        x       = (win.index - win.index[0]).days.values.astype(float)
        slope_d = np.polyfit(x, win.values, 1)[0]      # pendiente en m/s·día⁻¹
        # Escalar por gap promedio → m/s por sesión
        avg_gap = float(np.clip(x[-1] / max(len(x) - 1, 1), _GAP_MIN, _GAP_MAX))
        return float(slope_d * avg_gap)

    beta_aguda = float(np.clip(_pendiente_calendar(7,  2), -0.25, 0.25))
    beta_28    = float(np.clip(_pendiente_calendar(28, 3), -0.25, 0.25))

    # ── DQI — Data Quality Index ──────────────────────────────────────────────
    n_7d  = int(vmp_daily[vmp_daily.index >= (last_date - pd.Timedelta(days=6))].notna().sum())
    n_28d = int(vmp_daily[vmp_daily.index >= (last_date - pd.Timedelta(days=27))].notna().sum())
    dqi   = _DQI_W7 * min(1.0, n_7d / _REF_7D) + _DQI_W28 * min(1.0, n_28d / _REF_28D)

    if   dqi >= 0.80: calidad = "alta"
    elif dqi >= 0.50: calidad = "media"
    elif dqi >= 0.20: calidad = "baja"
    else:             calidad = "insuficiente"

    # ── SWC personal (pequeño cambio mínimo detectable) ──────────────────────
    edad_atleta  = (perfil or {}).get("edad", 18)
    ultimas_8    = sub.tail(8)["vmp_hoy"].values
    sd_personal  = float(np.std(ultimas_8)) if len(ultimas_8) >= 4 else 0.0
    mult_swc     = 1.5 if edad_atleta < 15 else 1.0
    swc_personal = sd_personal * mult_swc

    # ── Flag de ruido biológico ───────────────────────────────────────────────
    caida_absoluta     = mmc28 - last_vmp
    es_ruido_biologico = (
        caida_absoluta > 0
        and swc_personal > 0
        and caida_absoluta < swc_personal
    )

    # ── Sesiones descendentes consecutivas ───────────────────────────────────
    vmp_rev = sub["vmp_hoy"].values[::-1]
    n_desc  = 0
    for i in range(len(vmp_rev) - 1):
        if vmp_rev[i] < vmp_rev[i + 1]:
            n_desc += 1
        else:
            break

    # ── Estadísticos adicionales ─────────────────────────────────────────────
    cv = (np.std(vmp_series.values) / np.mean(vmp_series.values) * 100
          if np.mean(vmp_series.values) > 0 else 0.0)
    _, p_n = stats.shapiro(vmp_series.values) if n >= 8 else (None, None)

    return {
        # Identidad
        "atleta":              atleta,
        "edad_atleta":         edad_atleta,
        "n_sesiones":          n,
        "ultima_fecha":        str(last_date)[:10],
        "dias_sin_datos":      int((hoy - last_date).days),
        # VMP bruta
        "vmp_hoy":             last_vmp,
        "mma7":                mma7,
        "mmc28":               mmc28,
        # ── 5 variables Mamdani ──
        "acwr":                acwr,
        "delta_pct":           delta_pct,
        "z_meso":              z_meso,
        "beta_aguda":          beta_aguda,
        "beta_28":             beta_28,
        # Calidad de dato
        "dqi":                 round(dqi, 3),
        "calidad_dato":        calidad,
        # SWC / ruido biológico
        "swc_personal":        round(swc_personal, 4),
        "sd_personal":         round(sd_personal, 4),
        "caida_absoluta":      round(caida_absoluta, 4),
        "es_ruido_biologico":  es_ruido_biologico,
        # Tendencia
        "n_sesiones_desc":     n_desc,
        # Historial para gráficos
        "historial":           vmp_series.values.tolist(),
        "fechas":              [str(d)[:10] for d in vmp_series.index],
        # Estadísticos
        "cv_pct":              float(cv),
        "p_normalidad":        float(p_n) if p_n is not None else None,
    }


def detectar_tendencia_mpv(df: pd.DataFrame, ventana: int = 3) -> bool:
    """True si las últimas `ventana` sesiones muestran VMP estrictamente decreciente."""
    sub = df.sort_values("fecha")
    if len(sub) < ventana:
        return False
    vals = sub.tail(ventana)["vmp_hoy"].values
    return bool(all(vals[i] > vals[i + 1] for i in range(len(vals) - 1)))


# ─────────────────────────────────────────────────────────────────────────────
# FASE 3 — PIPELINE MAMDANI
# ─────────────────────────────────────────────────────────────────────────────

def pipeline_diagnostico(
    atleta: str,
    df_raw: pd.DataFrame,
    simulador,
    ventana_meso: int = 28,
    perfil: dict | None = None,
) -> dict | None:
    """
    Pipeline completo: datos → métricas → motor Mamdani → resultado.

    Flujo
    -----
    1. Filtra sesiones del atleta desde df_raw (snake_case).
    2. Calcula las 5 variables + métricas auxiliares.
    3. Aplica filtro SWC (neutraliza delta_pct si es ruido biológico).
    4. Corre el motor Mamdani y categoriza el índice de fatiga.
    5. Genera advertencias clínicas.

    Parámetros
    ----------
    atleta      : Nombre del atleta.
    df_raw      : DataFrame completo de sesiones_vmp (desde db.cargar_sesiones_raw).
    simulador   : ControlSystemSimulation ya construido (cachear en app.py).
    ventana_meso: Ventana en días para Z-Score mesociclo.
    perfil      : {"edad": int} opcional para multiplicador SWC pediátrico.

    Retorna None si el atleta no tiene datos suficientes.
    Retorna dict con: indice_fatiga, estado, color, accion, advertencias,
                      contexto_cientifico, nota_swc + todas las métricas.
    """
    metricas = calcular_metricas(df_raw, atleta, ventana_meso, perfil)
    if metricas is None:
        return None

    # ── Filtro SWC pre-motor ─────────────────────────────────────────────────
    if metricas["es_ruido_biologico"]:
        metricas_fuzzy = {**metricas, "delta_pct": 0.0}
        nota_swc = (
            f"⬇ Caída {metricas['caida_absoluta']:.3f} m/s < SWC "
            f"{metricas['swc_personal']:.3f} m/s → variabilidad biológica normal."
        )
    else:
        metricas_fuzzy = metricas
        nota_swc = ""

    # ── Motor Mamdani ────────────────────────────────────────────────────────
    try:
        simulador.input["acwr"]       = metricas_fuzzy["acwr"]
        simulador.input["delta_pct"]  = metricas_fuzzy["delta_pct"]
        simulador.input["z_meso"]     = metricas_fuzzy["z_meso"]
        simulador.input["beta_aguda"] = metricas_fuzzy["beta_aguda"]
        simulador.input["beta_28"]    = metricas_fuzzy["beta_28"]
        simulador.compute()
        indice = float(simulador.output["fatiga"])
    except Exception as exc:
        log.warning("Motor fuzzy falló para '%s': %s — fallback 50.0", atleta, exc)
        indice = 50.0

    # ── Categorización del índice ─────────────────────────────────────────────
    if   indice >= 75:
        estado, color           = "🟢 ÓPTIMO",           "#16a34a"
        accion_primaria         = "Entrenamiento normal"
        contexto_cientifico     = "Adaptación positiva. Puede progresar carga con monitoreo."
    elif indice >= 50:
        estado, color           = "🟡 ALERTA TEMPRANA",  "#ca8a04"
        accion_primaria         = "Reducir carga 10–15%"
        contexto_cientifico     = "Monitoreo estrecho 48 h. Re-evaluar si persiste caída de VMP."
    elif indice >= 25:
        estado, color           = "🟠 FATIGA ACUMULADA", "#ea580c"
        accion_primaria         = "Sesión regenerativa únicamente"
        contexto_cientifico     = "Sin carga intensa. Priorizar recuperación y sueño."
    else:
        estado, color           = "🔴 CRÍTICO",           "#dc2626"
        accion_primaria         = "Descanso obligatorio"
        contexto_cientifico     = "Evaluación médica si el estado persiste > 24 h."

    # ── Advertencias clínicas ─────────────────────────────────────────────────
    advertencias: list[str] = []

    # Estrés pediátrico
    if metricas["edad_atleta"] < 15 and metricas["delta_pct"] > 20:
        advertencias.append("🚨 Estrés pediátrico — revisar carga semanal")

    # Ajuste por calidad de dato
    if metricas["calidad_dato"] == "insuficiente":
        indice = min(indice, 62.0)
        advertencias.append("⚠️ Datos insuficientes — decisión con baja confianza")
    elif metricas["calidad_dato"] == "baja" and metricas["dias_sin_datos"] > 7:
        advertencias.append(f"⚠️ Sin datos hace {metricas['dias_sin_datos']} días")

    # Tendencia descendente sostenida
    if metricas["n_sesiones_desc"] >= 3:
        advertencias.append("📉 Tendencia descendente ≥ 3 sesiones consecutivas")

    accion = accion_primaria
    if advertencias:
        accion = advertencias[0] + " · " + accion_primaria

    return {
        **metricas,
        "indice_fatiga":       round(indice, 1),
        "estado":              estado,
        "color":               color,
        "accion":              accion,
        "accion_primaria":     accion_primaria,
        "advertencias":        advertencias,
        "contexto_cientifico": contexto_cientifico,
        "nota_swc":            nota_swc,
    }


def pipeline_batch(
    df_raw: pd.DataFrame,
    simulador,
    ventana_meso: int = 28,
) -> pd.DataFrame:
    """
    Ejecuta pipeline_diagnostico para todos los atletas en df_raw.
    Retorna DataFrame con una fila por atleta (todas las métricas + resultado fuzzy).
    """
    atletas = sorted(df_raw["nombre"].dropna().unique())
    resultados = []
    for atleta in atletas:
        res = pipeline_diagnostico(atleta, df_raw, simulador, ventana_meso)
        if res:
            resultados.append(res)
    return pd.DataFrame(resultados)


def calcular_historial_fatiga(
    df_raw: pd.DataFrame,
    atleta: str,
    simulador,
    ventana_meso: int = 28,
) -> pd.DataFrame:
    """
    Calcula el índice de fatiga para cada sesión histórica del atleta.
    Itera acumulando sesiones (O(n²), aceptable hasta ~150 sesiones).

    Retorna DataFrame con columnas: fecha, fatiga, estado, dqi.
    """
    sub = (
        df_raw[df_raw["nombre"] == atleta]
        .sort_values("fecha")
        .reset_index(drop=True)
    )
    resultados = []
    for i in range(4, len(sub) + 1):
        df_slice = sub.iloc[:i].copy()
        res = pipeline_diagnostico(atleta, df_slice, simulador, ventana_meso)
        if res:
            resultados.append({
                "fecha":  str(df_slice["fecha"].iloc[-1])[:10],
                "fatiga": res["indice_fatiga"],
                "estado": res["estado"],
                "dqi":    res["dqi"],
            })
    return pd.DataFrame(resultados)
