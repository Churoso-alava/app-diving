"""
services.py — Lógica de métricas de fatiga neuromuscular v4.1
Sin dependencias de Streamlit ni base de datos.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats
from fuzzy import evaluar_atleta

log = logging.getLogger(__name__)

# =============================================================================
#  CONSTANTES DQI (Data Quality Index)
# =============================================================================

_DQI_W7  = 0.40
_DQI_W28 = 0.60
_REF_7D  = 3
_REF_28D = 12
_TOL     = 0.50
_GAP_MIN, _GAP_MAX = 1.0, 7.0


# =============================================================================
#  FASE 4 — VALIDACIÓN DE ENTRADA (SessionInput)
# =============================================================================

@dataclass
class SessionInput:
    """Valida una sesión antes de insertar en la base de datos."""
    nombre: str
    fecha:  str          # ISO 8601: "YYYY-MM-DD"
    vmp:    float
    notas:  str = field(default="")

    VMP_MIN: float = field(default=0.100, init=False, repr=False)
    VMP_MAX: float = field(default=2.500, init=False, repr=False)

    def __post_init__(self):
        errors: list[str] = []
        if not self.nombre or not self.nombre.strip():
            errors.append("nombre no puede estar vacío")
        if not (self.VMP_MIN <= self.vmp <= self.VMP_MAX):
            errors.append(
                f"VMP {self.vmp:.3f} fuera del rango fisiológico "
                f"[{self.VMP_MIN}, {self.VMP_MAX}] m/s"
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


# =============================================================================
#  CÁLCULO DE MÉTRICAS v4.1
# =============================================================================

def calcular_metricas(
    df: pd.DataFrame,
    atleta: str,
    ventana_meso: int = 28,
    perfil: dict | None = None,
) -> dict | None:
    """
    Calcula métricas de fatiga neuromuscular a partir de la VMP de la fase
    propulsiva del CMJ (única variable de entrada al sistema).

    Retorna None si el atleta tiene menos de 4 sesiones registradas.

    Correcciones v4.1:
      [C1] delta_pct calculado vs MMC28 (no vs MMA7).
      [C2] SWC personal: SD últimas 8 sesiones × multiplicador por edad.
      [C3] es_ruido_biologico: flag para neutralizar delta_pct en el motor.
      [C4] n_sesiones_desc: sesiones VMP consecutivamente decrecientes.
    """
    sub = df[df["Nombre"] == atleta].copy().sort_values("Fecha")
    sub = (
        sub.groupby("Fecha", as_index=False)["VMP_Hoy"]
        .max()
        .sort_values("Fecha")
        .reset_index(drop=True)
    )

    n = len(sub)
    if n < 4:
        log.debug("Atleta '%s' con %d sesiones: insuficiente (mínimo 4).", atleta, n)
        return None

    idx        = pd.to_datetime(sub["Fecha"])
    vmp_series = pd.Series(sub["VMP_Hoy"].values, index=idx, dtype=float)
    date_range = pd.date_range(start=idx.min(), end=idx.max(), freq="D")
    vmp_daily  = vmp_series.reindex(date_range)

    last_vmp   = float(vmp_series.iloc[-1])
    last_date  = idx.max()
    first_date = idx.min()

    dias_span = max((last_date - first_date).days, 1)
    freq_day  = n / dias_span

    mp_7d  = max(1, round(max(1, freq_day * 7)  * _TOL))
    mp_28d = max(2, round(max(2, freq_day * 28) * _TOL))

    mma7_s  = vmp_daily.rolling("7D",  min_periods=mp_7d).mean()
    mmc28_s = vmp_daily.rolling("28D", min_periods=mp_28d).mean()

    mma7  = float(mma7_s.iloc[-1])  if not pd.isna(mma7_s.iloc[-1])  else last_vmp
    mmc28 = float(mmc28_s.iloc[-1]) if not pd.isna(mmc28_s.iloc[-1]) else last_vmp

    acwr      = mma7 / mmc28 if mmc28 > 0 else 1.0
    delta_pct = ((mmc28 - last_vmp) / mmc28) * 100 if mmc28 > 0 else 0.0   # [C1]

    cutoff_meso = last_date - pd.Timedelta(days=ventana_meso - 1)
    win_meso    = vmp_daily[vmp_daily.index >= cutoff_meso].dropna()
    z_meso = (
        (last_vmp - float(win_meso.mean())) / float(win_meso.std())
        if len(win_meso) >= 4 and float(win_meso.std()) > 0
        else 0.0
    )

    def _beta_calendar(serie_daily, days_back, min_n):
        cutoff = serie_daily.index[-1] - pd.Timedelta(days=days_back - 1)
        win    = serie_daily[serie_daily.index >= cutoff].dropna()
        if len(win) < min_n:
            return 0.0
        x       = (win.index - win.index[0]).days.values.astype(float)
        slope_d = np.polyfit(x, win.values, 1)[0]
        avg_gap = np.clip(x[-1] / max(len(x) - 1, 1), _GAP_MIN, _GAP_MAX)
        return float(slope_d * avg_gap)

    beta_aguda = _beta_calendar(vmp_daily, 7,  2)
    beta_28    = _beta_calendar(vmp_daily, 28, 3)

    n_7d  = int(vmp_daily[vmp_daily.index >= (last_date - pd.Timedelta(days=6))].notna().sum())
    n_28d = int(vmp_daily[vmp_daily.index >= (last_date - pd.Timedelta(days=27))].notna().sum())
    dqi   = _DQI_W7 * min(1.0, n_7d / _REF_7D) + _DQI_W28 * min(1.0, n_28d / _REF_28D)

    if   dqi >= 0.80: calidad = "alta"
    elif dqi >= 0.50: calidad = "media"
    elif dqi >= 0.20: calidad = "baja"
    else:             calidad = "insuficiente"

    hoy = pd.Timestamp.today().normalize()
    cv  = (
        (np.std(vmp_series.values) / np.mean(vmp_series.values)) * 100
        if np.mean(vmp_series.values) > 0 else 0.0
    )
    _, p_n = stats.shapiro(vmp_series.values) if n >= 8 else (None, None)

    # [C2] SWC PERSONAL
    edad_atleta  = perfil.get("edad", 18) if perfil else 18
    ultimas_8    = sub.tail(8)["VMP_Hoy"].values
    sd_personal  = float(np.std(ultimas_8)) if len(ultimas_8) >= 4 else 0.0
    mult_swc     = 1.5 if edad_atleta < 15 else 1.0
    swc_personal = sd_personal * mult_swc

    # [C3] FLAG DE RUIDO BIOLÓGICO
    caida_absoluta     = mmc28 - last_vmp
    es_ruido_biologico = (
        caida_absoluta > 0
        and swc_personal > 0
        and caida_absoluta < swc_personal
    )

    # [C4] SESIONES DESCENDENTES CONSECUTIVAS
    vmp_ord = sub["VMP_Hoy"].values[::-1]
    n_desc  = 0
    for i in range(len(vmp_ord) - 1):
        if vmp_ord[i] < vmp_ord[i + 1]:
            n_desc += 1
        else:
            break

    return {
        "atleta":              atleta,
        "edad_atleta":         edad_atleta,
        "n_sesiones":          n,
        "ultima_fecha":        str(last_date)[:10],
        "dias_sin_datos":      int((hoy - last_date).days),
        "vmp_hoy":             last_vmp,
        "mma7":                mma7,
        "mmc28":               mmc28,
        "acwr":                float(np.clip(acwr, 0.50, 1.80)),
        "delta_pct":           float(np.clip(delta_pct, -20, 40)),
        "z_meso":              float(np.clip(z_meso, -4.0, 4.0)),
        "beta_aguda":          float(np.clip(beta_aguda, -0.25, 0.25)),
        "beta_28":             float(np.clip(beta_28,   -0.25, 0.25)),
        "dqi":                 round(dqi, 3),
        "calidad_dato":        calidad,
        "historial":           vmp_series.values.tolist(),
        "fechas":              vmp_series.index.tolist(),
        "cv_pct":              float(cv),
        "p_normalidad":        float(p_n) if p_n else None,
        "swc_personal":        round(swc_personal, 4),
        "sd_personal":         round(sd_personal, 4),
        "caida_absoluta":      round(caida_absoluta, 4),
        "es_ruido_biologico":  es_ruido_biologico,
        "n_sesiones_desc":     n_desc,
    }


def detectar_tendencia_mpv(df_atleta: pd.DataFrame, ventana: int = 3) -> bool:
    """True si las últimas `ventana` sesiones muestran VMP estrictamente decreciente."""
    if len(df_atleta) < ventana:
        return False
    ultimas = (
        df_atleta.sort_values("Fecha")
        .tail(ventana)["VMP_Hoy"]
        .values
    )
    return bool(all(ultimas[i] > ultimas[i + 1] for i in range(len(ultimas) - 1)))


def calcular_historial_fatiga(df, atleta, simulador):
    resultados = []

    sub = df[df["Nombre"] == atleta].sort_values("Fecha")

    for i in range(4, len(sub) + 1):
        df_slice = sub.iloc[:i]

        metricas = calcular_metricas(df_slice, atleta)
        if not metricas:
            continue

        resultado = evaluar_atleta(simulador, metricas)

        resultados.append({
            "fecha": df_slice["Fecha"].iloc[-1],
            "fatiga": resultado["indice_fatiga"],
            "dqi": metricas["dqi"]
        })

    return pd.DataFrame(resultados)
