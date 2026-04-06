import streamlit as st

# =============================================================================
#  SECCIÓN 1 — CONFIGURACIÓN DE PÁGINA  (debe ir ANTES de cualquier st.*)
# =============================================================================

st.set_page_config(
    page_title="Dashboard Fatiga · Club Tornados",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
  .block-container { padding-top: 1.2rem; }
  .stTabs [data-baseweb="tab-list"] { gap: 8px; }
  .stTabs [data-baseweb="tab"] {
    background: #1e293b; border-radius: 6px 6px 0 0;
    color: #94a3b8; padding: 8px 20px; font-weight: 600;
  }
  .stTabs [aria-selected="true"] { background: #0f172a; color: #38bdf8; }
  .stDataFrame { font-size: 13px; }
  div[data-testid="metric-container"] {
    background: #1e293b; border-radius: 8px; padding: 12px; border-left: 3px solid #334155;
  }
  .form-card {
    background: #1e293b; border-radius: 10px; padding: 20px;
    border: 1px solid #334155; margin-bottom: 16px;
  }
</style>
""", unsafe_allow_html=True)


# =============================================================================
#  SECCIÓN 2 — PROTECCIÓN CON CONTRASEÑA
# =============================================================================

def check_password() -> bool:
    if "password_correct" not in st.session_state:
        st.session_state.password_correct = False

    if not st.session_state.password_correct:
        st.write("🔒 Ingresa la contraseña para acceder")
        password = st.text_input("Contraseña:", type="password")
        expected = st.secrets.get("APP_PASSWORD", "ENDCLAVADOS2026")

        if password == expected:
            st.session_state.password_correct = True
            st.rerun()
        elif password:
            st.error("❌ Contraseña incorrecta")
            return False
        return False
    return True

if not check_password():
    st.stop()


# =============================================================================
#  IMPORTACIONES
# =============================================================================

import numpy as np
import pandas as pd
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt
from scipy import stats
import warnings
from datetime import date, timedelta

from database import (
    cargar_sesiones,
    cargar_atletas,
    insertar_sesion,
    actualizar_sesion,
    eliminar_sesion,
    importar_dataframe,
)

warnings.filterwarnings("ignore")


# =============================================================================
#  SECCIÓN 3 — CARGA DE DATOS CON CACHÉ (TTL 30 s)
# =============================================================================

@st.cache_data(ttl=30)
def cargar_sesiones_cached() -> pd.DataFrame:
    return cargar_sesiones()

@st.cache_data(ttl=30)
def cargar_atletas_cached() -> list:
    return cargar_atletas()


# =============================================================================
#  SECCIÓN 4 — MODELO MATEMÁTICO v4.1
#  CORRECCIONES APLICADAS:
#  [C1] delta_pct ahora se calcula contra MMC28 (baseline crónico),
#       no contra MMA7 (que incluye días fatigados y sesga el denominador).
#  [C2] SWC personal: filtro pre-motor basado en SD de las últimas 8 sesiones.
#       Adultos: 1.0×SD  |  Jóvenes <15 a: 1.5×SD (mayor variabilidad inherente).
#  [C3] es_ruido_biologico: flag que neutraliza delta_pct antes del motor fuzzy
#       cuando la caída es menor al SWC personal (resuelve falso positivo Ariana).
#  [C4] n_sesiones_descendentes: contador de sesiones VMP consecutivamente
#       decrecientes para el módulo de tendencia inter-sesión del SNC.
# =============================================================================

# Constantes del DQI (Data Quality Index)
_DQI_W7  = 0.40
_DQI_W28 = 0.60
_REF_7D  = 3
_REF_28D = 12
_TOL     = 0.50
_GAP_MIN, _GAP_MAX = 1.0, 7.0


def calcular_metricas(
    df: pd.DataFrame,
    atleta: str,
    ventana_meso: int = 28,
    perfil: dict | None = None,   # dict con clave 'edad' para ajuste pediátrico
) -> dict | None:
    """
    Calcula métricas de fatiga neuromuscular a partir de la VMP de la fase
    propulsiva del CMJ (única variable de entrada al sistema).

    NOTA: Esta función NO utiliza RSImod ni tiempo de contacto.
    La VMP aquí es la velocidad media en la fase concéntrica del salto libre
    (sin carga externa), que es el punto más sensible al estado del SNC en la
    curva carga-velocidad (González-Badillo et al., 2022).
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

    acwr = mma7 / mmc28 if mmc28 > 0 else 1.0

    # ── [C1] CORRECCIÓN: delta_pct vs MMC28 (baseline crónico estable) ────────
    # Antes: ((mma7 - last_vmp) / mma7) — INCORRECTO: MMA7 incluye días
    # fatigados y sesga el denominador hacia abajo, amplificando la caída.
    # Ahora: ((mmc28 - last_vmp) / mmc28) — CORRECTO: MMC28 representa la
    # capacidad física base del atleta (González-Badillo et al., 2022).
    delta_pct = ((mmc28 - last_vmp) / mmc28) * 100 if mmc28 > 0 else 0.0

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
        x        = (win.index - win.index[0]).days.values.astype(float)
        slope_d  = np.polyfit(x, win.values, 1)[0]
        avg_gap  = np.clip(x[-1] / max(len(x) - 1, 1), _GAP_MIN, _GAP_MAX)
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

    # ── [C2] SWC PERSONAL ─────────────────────────────────────────────────────
    # Smallest Worthwhile Change calculado sobre las últimas 8 sesiones.
    # Multiplicador 1.5 para jóvenes <15 a (mayor variabilidad inherente;
    # sin norma publicada — multiplicador conservador, Consensus 2026).
    edad_atleta  = perfil.get("edad", 18) if perfil else 18
    ultimas_8    = sub.tail(8)["VMP_Hoy"].values
    sd_personal  = float(np.std(ultimas_8)) if len(ultimas_8) >= 4 else 0.0
    mult_swc     = 1.5 if edad_atleta < 15 else 1.0
    swc_personal = sd_personal * mult_swc

    # ── [C3] FLAG DE RUIDO BIOLÓGICO ─────────────────────────────────────────
    # Si la caída absoluta (MMC28 - VMP_hoy) no supera el SWC personal,
    # la variación está dentro del ruido biológico normal inter-sesión
    # (CV% CMJ 4–8%; Jukic 2022; Alba-Jiménez 2022).
    # Este flag neutraliza delta_pct antes de alimentar el motor fuzzy.
    caida_absoluta    = mmc28 - last_vmp   # positivo = cayó, negativo = mejoró
    es_ruido_biologico = (
        caida_absoluta > 0                  # solo si hay caída real
        and swc_personal > 0                # solo si hay suficiente historial
        and caida_absoluta < swc_personal   # dentro del ruido
    )

    # ── [C4] CONTADOR DE SESIONES DESCENDENTES (tendencia inter-sesión SNC) ───
    # Cuenta cuántas sesiones consecutivas (desde la más reciente hacia atrás)
    # muestran VMP decreciente. Proxy del deterioro gradual del SNC
    # (Weakley 2019: efectos 24 h entre sesiones son desconocidos →
    # usamos patrón acumulado como señal conservadora).
    vmp_ord = sub["VMP_Hoy"].values[::-1]   # más reciente primero
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
        # ── v4.1 ──────────────────────────────────────────────────────────────
        "swc_personal":        round(swc_personal, 4),
        "sd_personal":         round(sd_personal, 4),
        "caida_absoluta":      round(caida_absoluta, 4),
        "es_ruido_biologico":  es_ruido_biologico,
        "n_sesiones_desc":     n_desc,
    }


# =============================================================================
#  MÓDULO DE TENDENCIA INTER-SESIÓN (VMP como proxy del SNC)
#  Basado en: Weakley 2019 — efectos 24 h entre sesiones son desconocidos.
#  Se usa tendencia acumulada ≥3 sesiones como señal conservadora, NO como
#  disparador de alerta clínica aislado.
# =============================================================================

def detectar_tendencia_mpv(df_atleta: pd.DataFrame, ventana: int = 3) -> bool:
    """
    Retorna True si las últimas `ventana` sesiones muestran VMP
    estrictamente decreciente.

    Cada caída individual puede ser sub-SWC (ruido biológico), pero el
    patrón acumulado de ≥3 sesiones es una señal proxy del estado del SNC
    que justifica revisar la carga semanal.
    """
    if len(df_atleta) < ventana:
        return False
    ultimas = (
        df_atleta.sort_values("Fecha")
        .tail(ventana)["VMP_Hoy"]
        .values
    )
    return bool(all(ultimas[i] > ultimas[i + 1] for i in range(len(ultimas) - 1)))


# =============================================================================
#  SECCIÓN 5 — MOTOR FUZZY MAMDANI v4.1
# =============================================================================

@st.cache_resource
def construir_sistema_fuzzy():
    """Universos + funciones de pertenencia Mamdani. Sin cambios en universos."""
    u_acwr  = np.arange(0.50, 1.81,  0.01)
    u_delta = np.arange(-20,  41,    0.5)
    u_zmeso = np.arange(-4,   4.1,   0.1)
    u_ba    = np.arange(-0.25, 0.26,  0.001)
    u_b28   = np.arange(-0.25, 0.26,  0.001)
    u_fat   = np.arange(0, 101, 1)

    acwr_v  = ctrl.Antecedent(u_acwr,  "acwr")
    delta_v = ctrl.Antecedent(u_delta, "delta_pct")
    zmeso_v = ctrl.Antecedent(u_zmeso, "z_meso")
    ba_v    = ctrl.Antecedent(u_ba,    "beta_aguda")
    b28_v   = ctrl.Antecedent(u_b28,   "beta_28")
    fat_v   = ctrl.Consequent(u_fat,   "fatiga")

    acwr_v["bajo"]     = fuzz.trapmf(u_acwr, [0.50, 0.50, 0.70, 0.85])
    acwr_v["optimo"]   = fuzz.trapmf(u_acwr, [0.78, 0.88, 1.20, 1.30])
    acwr_v["alto"]     = fuzz.trimf (u_acwr, [1.25, 1.40, 1.55])
    acwr_v["excesivo"] = fuzz.trapmf(u_acwr, [1.45, 1.55, 1.80, 1.80])

    delta_v["ganancia"]   = fuzz.trapmf(u_delta, [-20, -20, -5,  0])
    delta_v["tolerable"]  = fuzz.trapmf(u_delta, [ -2,   0,  8, 12])
    delta_v["vigilancia"] = fuzz.trimf (u_delta, [ 10,  15, 22])
    delta_v["alarma"]     = fuzz.trapmf(u_delta, [ 18,  22, 40, 40])

    zmeso_v["muy_bajo"] = fuzz.trapmf(u_zmeso, [-4.0, -4.0, -2.2, -1.5])
    zmeso_v["bajo"]     = fuzz.trimf (u_zmeso, [-2.0, -1.2, -0.5])
    zmeso_v["normal"]   = fuzz.trimf (u_zmeso, [-0.8,  0.0,  0.8])
    zmeso_v["elevado"]  = fuzz.trapmf(u_zmeso, [ 0.6,  1.2,  4.0,  4.0])

    ba_v["neg_fuerte"]   = fuzz.trapmf(u_ba, [-0.25, -0.25, -0.06, -0.025])
    ba_v["neg_moderada"] = fuzz.trimf (u_ba, [-0.04, -0.015, -0.005])
    ba_v["estable"]      = fuzz.trimf (u_ba, [-0.008,  0.0,   0.008])
    ba_v["positiva"]     = fuzz.trapmf(u_ba, [ 0.005,  0.025, 0.25,  0.25])

    b28_v["deterioro"] = fuzz.trapmf(u_b28, [-0.25, -0.25, -0.03, -0.01])
    b28_v["estable"]   = fuzz.trimf (u_b28, [-0.012,  0.0,   0.012])
    b28_v["mejora"]    = fuzz.trapmf(u_b28, [ 0.010,  0.03,  0.25,  0.25])

    fat_v["critico"]          = fuzz.trapmf(u_fat, [  0,  0, 18, 28])
    fat_v["fatiga_acumulada"] = fuzz.trimf (u_fat, [ 25, 37, 52])
    fat_v["alerta_temprana"]  = fuzz.trimf (u_fat, [ 50, 62, 76])
    fat_v["optimo"]           = fuzz.trapmf(u_fat, [ 75, 88, 100, 100])

    return acwr_v, delta_v, zmeso_v, ba_v, b28_v, fat_v


def construir_reglas(acwr_v, delta_v, zmeso_v, ba_v, b28_v, fat_v):
    """
    23 reglas IF-THEN del motor Mamdani v4.1.

    CAMBIOS vs v4:
      [MOD] R04: ACWR excesivo ahora requiere TRIPLE condición
            (excesivo + z muy_bajo + beta_aguda neg_fuerte) para ser CRÍTICO.
            Antes: solo excesivo + z bajo/muy_bajo → demasiado sensible en
            periodos de overreaching planificado.
      [ELI] R05 ELIMINADA: zmeso muy_bajo solo → crítico.
            Era veto absoluto que marcaba como crítico a atletas en tapering
            o recuperación activa donde Z bajo es fisiológicamente esperable.
      [NUE] R22: beta_aguda positiva + z_meso bajo → alerta_temprana.
            Atleta en recuperación activa: tendencia semanal al alza pero
            base mensual aún deprimida. Evita clasificar como crítico un
            proceso de supercompensación.
      [NUE] R23: ACWR excesivo + z_meso normal → fatiga_acumulada (no crítico).
            Atleta que soporta carga alta con rendimiento aún dentro del
            rango normal. Riesgo latente, no crisis aguda.
    """
    return [
        # ── BLOQUE CRÍTICO (4 reglas; R05 eliminada) ─────────────────────────
        ctrl.Rule(
            acwr_v["bajo"] & delta_v["alarma"] & ba_v["neg_fuerte"],
            fat_v["critico"]
        ),  # R01
        ctrl.Rule(
            acwr_v["bajo"] & b28_v["deterioro"] & ba_v["neg_fuerte"],
            fat_v["critico"]
        ),  # R02
        ctrl.Rule(
            delta_v["alarma"] & b28_v["deterioro"] & zmeso_v["muy_bajo"],
            fat_v["critico"]
        ),  # R03
        # R04 MODIFICADA — triple condición (antes: 2 condiciones)
        ctrl.Rule(
            acwr_v["excesivo"] & zmeso_v["muy_bajo"] & ba_v["neg_fuerte"],
            fat_v["critico"]
        ),  # R04 v4.1

        # ── BLOQUE FATIGA ACUMULADA (8 reglas) ───────────────────────────────
        ctrl.Rule(
            acwr_v["bajo"] & delta_v["alarma"],
            fat_v["fatiga_acumulada"]
        ),  # R05 (renumerada desde R06)
        ctrl.Rule(
            acwr_v["bajo"] & delta_v["vigilancia"] & b28_v["deterioro"],
            fat_v["fatiga_acumulada"]
        ),  # R06
        ctrl.Rule(
            acwr_v["optimo"] & ba_v["neg_fuerte"] & zmeso_v["muy_bajo"],
            fat_v["fatiga_acumulada"]
        ),  # R07
        ctrl.Rule(
            delta_v["alarma"] & ba_v["neg_moderada"],
            fat_v["fatiga_acumulada"]
        ),  # R08
        ctrl.Rule(
            acwr_v["alto"] & delta_v["vigilancia"] & b28_v["deterioro"],
            fat_v["fatiga_acumulada"]
        ),  # R09
        # R10: ACWR excesivo pero rendimiento normal → fatiga acumulada latente
        ctrl.Rule(
            acwr_v["excesivo"] & zmeso_v["normal"],
            fat_v["fatiga_acumulada"]
        ),  # R10 [NUE R23 original]
        ctrl.Rule(
            b28_v["deterioro"]
            & (ba_v["neg_fuerte"] | ba_v["neg_moderada"])
            & (zmeso_v["bajo"] | zmeso_v["muy_bajo"]),
            fat_v["fatiga_acumulada"]
        ),  # R11 — SSENF enmascarado por carga controlada
        ctrl.Rule(
            acwr_v["excesivo"] & (zmeso_v["bajo"] | zmeso_v["muy_bajo"])
            & ba_v["neg_moderada"],
            fat_v["fatiga_acumulada"]
        ),  # R12 — ACWR excesivo + Z bajo + tendencia moderada

        # ── BLOQUE ALERTA TEMPRANA (6 reglas) ────────────────────────────────
        ctrl.Rule(
            delta_v["vigilancia"] & ba_v["neg_moderada"] & zmeso_v["normal"],
            fat_v["alerta_temprana"]
        ),  # R13
        ctrl.Rule(
            acwr_v["optimo"] & zmeso_v["bajo"] & delta_v["vigilancia"],
            fat_v["alerta_temprana"]
        ),  # R14
        ctrl.Rule(
            acwr_v["alto"] & ba_v["estable"] & delta_v["tolerable"],
            fat_v["alerta_temprana"]
        ),  # R15
        ctrl.Rule(
            b28_v["deterioro"] & delta_v["tolerable"] & zmeso_v["bajo"],
            fat_v["alerta_temprana"]
        ),  # R16
        ctrl.Rule(
            ba_v["neg_fuerte"] & b28_v["estable"] & zmeso_v["normal"],
            fat_v["alerta_temprana"]
        ),  # R17 — caída aguda sobre base sólida
        # R18: recuperación activa — [NUE R22 original]
        # Beta positiva + Z bajo: atleta mejorando semana a semana pero base
        # mensual aún deprimida. Supercompensación, no crisis.
        ctrl.Rule(
            ba_v["positiva"] & zmeso_v["bajo"],
            fat_v["alerta_temprana"]
        ),  # R18 [NUE]

        # ── BLOQUE ÓPTIMO (5 reglas) ──────────────────────────────────────────
        ctrl.Rule(
            acwr_v["optimo"] & delta_v["tolerable"] & ba_v["positiva"] & b28_v["mejora"],
            fat_v["optimo"]
        ),  # R19
        ctrl.Rule(
            acwr_v["optimo"] & delta_v["ganancia"] & b28_v["estable"],
            fat_v["optimo"]
        ),  # R20
        ctrl.Rule(
            acwr_v["optimo"] & zmeso_v["normal"] & ba_v["estable"] & delta_v["tolerable"],
            fat_v["optimo"]
        ),  # R21
        # R22: protección tapering / supercompensación
        ctrl.Rule(
            (acwr_v["bajo"] | acwr_v["optimo"])
            & zmeso_v["elevado"]
            & (ba_v["estable"] | ba_v["positiva"]),
            fat_v["optimo"]
        ),  # R22
        ctrl.Rule(
            acwr_v["optimo"] & zmeso_v["elevado"] & b28_v["mejora"],
            fat_v["optimo"]
        ),  # R23
    ]


@st.cache_resource
def construir_motor_fuzzy():
    vars_tuple = construir_sistema_fuzzy()
    acwr_v, delta_v, zmeso_v, ba_v, b28_v, fat_v = vars_tuple
    reglas    = construir_reglas(acwr_v, delta_v, zmeso_v, ba_v, b28_v, fat_v)
    sistema   = ctrl.ControlSystem(reglas)
    simulador = ctrl.ControlSystemSimulation(sistema)
    return vars_tuple, simulador


def evaluar_atleta(simulador, metricas: dict) -> dict:
    """
    Corre el motor difuso aplicando el filtro SWC pre-motor (v4.1).

    FILTRO SWC [C3]:
    Si es_ruido_biologico=True, la caída de VMP está dentro del ruido
    inter-sesión normal (CV% CMJ 4–8%; Jukic 2022; Alba-Jiménez 2022).
    En ese caso se envía delta_pct=0 al motor para neutralizar la señal
    antes de que dispare una alerta incorrecta (caso Ariana).

    El resto de variables (ACWR, Z-meso, betas) se envían sin modificar
    para que el motor pueda detectar fatiga por otras vías si existe.
    """
    # ── [C3] FILTRO SWC PRE-MOTOR ─────────────────────────────────────────────
    if metricas.get("es_ruido_biologico", False):
        # Neutralizar delta_pct: la caída es ruido, no señal real
        metricas_fuzzy = {**metricas, "delta_pct": 0.0}
        nota_swc = (
            f"⬇ Caída {metricas['caida_absoluta']:.3f} m/s < SWC "
            f"{metricas['swc_personal']:.3f} m/s → variabilidad biológica normal."
        )
    else:
        metricas_fuzzy = metricas
        nota_swc = ""

    try:
        simulador.input["acwr"]       = metricas_fuzzy["acwr"]
        simulador.input["delta_pct"]  = metricas_fuzzy["delta_pct"]
        simulador.input["z_meso"]     = metricas_fuzzy["z_meso"]
        simulador.input["beta_aguda"] = metricas_fuzzy["beta_aguda"]
        simulador.input["beta_28"]    = metricas_fuzzy["beta_28"]
        simulador.compute()
        indice = simulador.output["fatiga"]
    except Exception:
        indice = 50.0

    if   indice >= 75: estado = "🟢 ÓPTIMO";           color = "#16a34a"; accion = "Entrenamiento normal. Posible progresión."
    elif indice >= 50: estado = "🟡 ALERTA TEMPRANA";  color = "#ca8a04"; accion = "Reducir 10–15%. Monitoreo estrecho."
    elif indice >= 25: estado = "🟠 FATIGA ACUMULADA"; color = "#ea580c"; accion = "Sesión regenerativa. Sin carga intensa."
    else:              estado = "🔴 CRÍTICO";           color = "#dc2626"; accion = "Descanso obligatorio / evaluación médica."

    # Alerta pediátrica (documento base §3)
    if metricas.get("edad_atleta", 18) < 15 and metricas["delta_pct"] > 20:
        accion = "🚨 RIESGO ESTRÉS PEDIÁTRICO — " + accion

    # Resiliencia DQI
    calidad        = metricas.get("calidad_dato", "alta")
    dias_sin_datos = metricas.get("dias_sin_datos", 0)
    if calidad == "insuficiente":
        indice = min(indice, 62.0)
        accion = "⚠ Datos insuficientes (<4 ses recientes). " + accion
    elif calidad == "baja" and dias_sin_datos > 7:
        accion = f"⚠ Datos desactualizados ({dias_sin_datos}d). " + accion

    return {
        **metricas,
        "indice_fatiga": round(indice, 1),
        "estado":        estado,
        "color":         color,
        "accion":        accion,
        "nota_swc":      nota_swc,
    }


# =============================================================================
#  SECCIÓN 6 — GRÁFICOS
# =============================================================================

def fig_semaforo(df_res: pd.DataFrame) -> plt.Figure:
    df_p = df_res.sort_values("indice_fatiga").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(10, max(4, len(df_p) * 0.9)))
    fig.patch.set_facecolor("#0f172a")
    ax.set_facecolor("#0f172a")
    ax.axis("off")
    for i, r in df_p.iterrows():
        y = i * 1.1
        ax.barh(y, 100, height=0.72, color="#1e293b", left=0, zorder=1)
        ax.barh(y, r["indice_fatiga"], height=0.72, color=r["color"], alpha=0.88, zorder=2)
        ax.text(-1.5, y, r["atleta"], va="center", ha="right",
                fontsize=11, color="white", fontweight="bold")
        ax.text(r["indice_fatiga"] + 1.5, y, f"{r['indice_fatiga']:.0f}",
                va="center", fontsize=10, color=r["color"], fontweight="bold")
        ax.text(72, y, r["estado"].split(" ", 1)[1],
                va="center", fontsize=8.5, color=r["color"])
        ax.text(105, y, r.get("ultima_fecha", ""), va="center",
                fontsize=7.5, color="#64748b")
    ax.set_xlim(-22, 125)
    ax.set_ylim(-0.6, len(df_p) * 1.1)
    ax.set_title("Índice de Fatiga  [0 = Crítico → 100 = Óptimo]",
                 color="white", fontsize=11, pad=12)
    plt.tight_layout()
    return fig


def fig_tendencia(m: dict) -> plt.Figure:
    vmp    = np.array(m["historial"])
    fechas = m["fechas"]
    n      = len(vmp)

    # Rolling sobre base diaria interpolada igual que en calcular_metricas
    mma7s  = pd.Series(vmp).rolling(7,  min_periods=3).mean().values
    mmc28s = pd.Series(vmp).rolling(28, min_periods=7).mean().values

    fig, ax = plt.subplots(figsize=(10, 4))
    fig.patch.set_facecolor("#0f172a")
    ax.set_facecolor("#1e293b")
    for sp in ax.spines.values():
        sp.set_color("#334155")
    ax.tick_params(colors="#94a3b8", labelsize=8)

    xs = range(n)
    ax.plot(xs, vmp,    color="#38bdf8", lw=1.5, alpha=0.6,
            label="VMP Fase Propulsiva CMJ (m/s)")
    ax.scatter(xs, vmp, color="#38bdf8", s=22, zorder=5)
    ax.plot(xs, mma7s,  color="#fb923c", lw=2,
            label="Fatiga Aguda — MMA7 (Últimos 7 días)")
    ax.plot(xs, mmc28s, color="#a78bfa", lw=2,
            label="Baseline Crónico — MMC28 (Últimos 28 días)")

    ax.axvline(n - 1, color=m["color"], lw=1.5, linestyle="--", alpha=0.7)
    ax.scatter([n - 1], [vmp[-1]], color=m["color"], s=70, zorder=6)

    # Zona de riesgo basada en MMC28 (v4.1: umbral vs baseline crónico)
    mmc28_val = m.get("mmc28", mma7s[-1] if len(mma7s) > 0 else vmp[-1])
    if mmc28_val > 0:
        umbral_alerta = mmc28_val * (1 - 0.15)   # SWC ≈ 15% sobre MMC28
        umbral_critico = mmc28_val * (1 - 0.25)  # Zona crítica ≥25%
        ax.axhline(umbral_alerta,  color="#f59e0b", lw=1, linestyle=":", alpha=0.8,
                   label="Umbral Alerta (~15% caída sobre MMC28)")
        ax.axhline(umbral_critico, color="#f87171", lw=1, linestyle=":", alpha=0.8,
                   label="Umbral Crítico (~25% caída sobre MMC28)")

    xticks = list(range(0, n, max(1, n // 10)))
    ax.set_xticks(xticks)
    ax.set_xticklabels(
        [str(fechas[i])[:10] if i < len(fechas) else "" for i in xticks],
        rotation=35, ha="right", fontsize=7.5, color="#94a3b8"
    )
    ax.set_ylabel("VMP Fase Propulsiva CMJ (m/s)", color="#94a3b8", fontsize=9)
    ax.set_title(
        f"Evolución VMP del CMJ — {m['atleta']}  "
        f"[Δ vs MMC28: {m.get('delta_pct', 0):+.1f}%]",
        color="white", fontsize=12, fontweight="bold"
    )
    ax.legend(fontsize=8, labelcolor="white",
              facecolor="#0f172a", edgecolor="#334155")
    plt.tight_layout()
    return fig


def fig_membership(vars_tuple) -> plt.Figure:
    acwr_v, delta_v, zmeso_v, ba_v, b28_v, fat_v = vars_tuple
    configs = [
        (acwr_v,  ["bajo","optimo","alto","excesivo"],               "ACWR"),
        (delta_v, ["ganancia","tolerable","vigilancia","alarma"],     "Δ% vs MMC28"),
        (zmeso_v, ["muy_bajo","bajo","normal","elevado"],             "Z-Score Mesociclo"),
        (ba_v,    ["neg_fuerte","neg_moderada","estable","positiva"], "Pendiente β₇"),
        (b28_v,   ["deterioro","estable","mejora"],                   "Pendiente β₂₈"),
        (fat_v,   ["critico","fatiga_acumulada","alerta_temprana","optimo"], "SALIDA: Fatiga"),
    ]
    colores = ["#f87171", "#fb923c", "#34d399", "#38bdf8"]
    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    fig.patch.set_facecolor("#0f172a")
    for ax, (var, labels, title) in zip(axes.flat, configs):
        ax.set_facecolor("#1e293b")
        ax.tick_params(colors="#94a3b8", labelsize=7)
        for sp in ax.spines.values():
            sp.set_color("#334155")
        for label, color in zip(labels, colores):
            mf = fuzz.interp_membership(var.universe, var[label].mf, var.universe)
            ax.plot(var.universe, mf, color=color, lw=2, label=label)
            ax.fill_between(var.universe, mf, alpha=0.07, color=color)
        ax.set_title(title, color="white", fontsize=9, fontweight="bold", pad=6)
        ax.legend(fontsize=6.5, labelcolor="white", facecolor="#0f172a",
                  edgecolor="#334155", loc="upper right")
        ax.set_ylim(-0.05, 1.1)
    plt.suptitle("Funciones de Pertenencia — Modelo Fuzzy Mamdani v4.1",
                 color="white", fontsize=12, fontweight="bold", y=1.01)
    plt.tight_layout()
    return fig


# =============================================================================
#  SECCIÓN 7 — SIDEBAR
# =============================================================================

def render_sidebar() -> dict:
    with st.sidebar:
        st.markdown("## ⚡ Club Tornados")
        st.markdown("**Dashboard de Fatiga v4.1**")
        st.divider()

        if st.button("🔄 Actualizar Datos Ahora", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
        st.caption("Caché: refresco automático cada 30 s.")
        st.divider()

        st.markdown("### ⚙️ Parámetros del Modelo")
        ventana_meso = st.slider(
            "Ventana Mesociclo — Z-score (días)", 14, 42, 28,
            help="Ventana reciente para Z-score. NO usar historial total."
        )
        st.divider()

        st.markdown("### 📊 Variables del Modelo v4.1")
        st.markdown("""
| # | Variable | Ventana | Cambio v4.1 |
|---|----------|---------|-------------|
| 1 | ACWR | MMA₇ / MMC₂₈ | — |
| 2 | Δ% VMP | **VMP vs MMC₂₈** | ✅ Era vs MMA₇ |
| 3 | Z-score | Mesociclo | — |
| 4 | β Aguda | 7 sesiones | — |
| 5 | β Tendencia | 28 sesiones | — |
| 6 | SWC | 8 sesiones | ✅ Nuevo filtro |
""")
        st.divider()
        st.caption(
            "Mamdani · 5 entradas · 23 reglas · COG defuzz.\n"
            "Filtro SWC pre-motor activo.\n"
            "Variable: VMP fase propulsiva CMJ."
        )
    return {"ventana_meso": ventana_meso}


# =============================================================================
#  SECCIÓN 8 — TAB: DASHBOARD
# =============================================================================

def tab_dashboard(df_raw: pd.DataFrame, simulador, vars_tuple, cfg: dict):
    atletas    = sorted(df_raw["Nombre"].unique())
    metricas_l = [calcular_metricas(df_raw, a, cfg["ventana_meso"]) for a in atletas]
    metricas_l = [m for m in metricas_l if m]
    resultados = [evaluar_atleta(simulador, m) for m in metricas_l]
    df_res     = pd.DataFrame(resultados)

    total    = len(df_res)
    criticos = (df_res["indice_fatiga"] < 25).sum()
    fatiga   = ((df_res["indice_fatiga"] >= 25) & (df_res["indice_fatiga"] < 50)).sum()
    alerta   = ((df_res["indice_fatiga"] >= 50) & (df_res["indice_fatiga"] < 75)).sum()
    optimos  = (df_res["indice_fatiga"] >= 75).sum()
    ruido    = df_res["es_ruido_biologico"].sum() if "es_ruido_biologico" in df_res.columns else 0

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("👥 Atletas",          total)
    c2.metric("🔴 Críticos",         criticos)
    c3.metric("🟠 Fatiga Acum.",      fatiga)
    c4.metric("🟡 Alerta Temp.",      alerta)
    c5.metric("🟢 Óptimos",           optimos)
    c6.metric("🔵 Filtro SWC activo", int(ruido),
              help="Atletas donde la caída de VMP está dentro del ruido biológico normal (< SWC personal). No genera alerta.")

    # ── Semáforo ──────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("## 🚦 Semáforo de Fatiga — Todos los Atletas")
    st.pyplot(fig_semaforo(df_res))

    # ── Tabla resumen ─────────────────────────────────────────────────────────
    st.markdown("## 📋 Tabla de Resultados")
    cols = [
        "atleta", "vmp_hoy", "mmc28", "acwr", "delta_pct",
        "z_meso", "beta_aguda", "beta_28",
        "swc_personal", "es_ruido_biologico",
        "dqi", "calidad_dato", "indice_fatiga", "estado", "accion", "ultima_fecha",
    ]
    df_t = (
        df_res[cols]
        .rename(columns={
            "atleta":             "Atleta",
            "vmp_hoy":            "VMP Hoy",
            "mmc28":              "MMC28 (base)",
            "acwr":               "ACWR",
            "delta_pct":          "Δ% vs MMC28",
            "z_meso":             "Z Meso",
            "beta_aguda":         "β₇",
            "beta_28":            "β₂₈",
            "swc_personal":       "SWC",
            "es_ruido_biologico": "Ruido Bio.",
            "dqi":                "DQI",
            "calidad_dato":       "Calidad",
            "indice_fatiga":      "Índice",
            "estado":             "Estado",
            "accion":             "Acción",
            "ultima_fecha":       "Última Sesión",
        })
        .sort_values("Índice")
    )
    st.dataframe(
        df_t.style.format({
            "VMP Hoy":    "{:.3f}",
            "MMC28 (base)": "{:.3f}",
            "ACWR":       "{:.3f}",
            "Δ% vs MMC28": "{:+.1f}%",
            "Z Meso":     "{:+.2f}",
            "β₇":         "{:+.4f}",
            "β₂₈":        "{:+.4f}",
            "SWC":        "{:.4f}",
            "DQI":        "{:.2f}",
            "Índice":     "{:.1f}",
        }),
        use_container_width=True, hide_index=True,
    )

    csv_bytes = df_t.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Descargar resultados (CSV)",
        data=csv_bytes,
        file_name=f"fatiga_tornados_{date.today()}.csv",
        mime="text/csv",
    )

    # ── Análisis individual ───────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("## 🔍 Análisis Individual")
    atletas_ord = df_res.sort_values("indice_fatiga")["atleta"].tolist()
    sel = st.selectbox("Selecciona un atleta:", atletas_ord)

    if sel:
        row = df_res[df_res["atleta"] == sel].iloc[0]
        m   = calcular_metricas(df_raw, sel, cfg["ventana_meso"])
        if m is None:
            st.warning("Datos insuficientes (mínimo 4 sesiones).")
            return

        m["estado"]        = row["estado"]
        m["color"]         = row["color"]
        m["indice_fatiga"] = row["indice_fatiga"]
        m["mmc28"]         = row["mmc28"]

        # ── Tendencia inter-sesión SNC ─────────────────────────────────────
        sub_atleta = df_raw[df_raw["Nombre"] == sel]
        if detectar_tendencia_mpv(sub_atleta, ventana=3):
            st.warning(
                "📉 **Tendencia VMP descendente en 3 sesiones consecutivas.** "
                "Proxy de deterioro gradual del SNC — revisar carga semanal. "
                "*(Weakley 2019: efectos 24 h entre sesiones aún sin norma publicada)*"
            )

        # ── Filtro SWC activo ─────────────────────────────────────────────
        if row.get("nota_swc"):
            st.info(f"🔵 **Filtro SWC:** {row['nota_swc']}")

        col_info, col_vars = st.columns([1, 2])
        with col_info:
            color = row["color"]
            calidad_badge = {
                "alta":         "🟢 Confianza Alta",
                "media":        "🟡 Confianza Media",
                "baja":         "🟠 Confianza Baja",
                "insuficiente": "🔴 Datos Insuficientes",
            }.get(row.get("calidad_dato", "media"), "")

            st.markdown(f"""
            <div style="background:#1e293b;border-radius:12px;padding:20px;
                 border-left:5px solid {color};text-align:center;">
              <div style="font-size:32px;font-weight:900;color:{color};">{row['indice_fatiga']:.0f}</div>
              <div style="font-size:10px;color:#64748b;text-transform:uppercase;letter-spacing:1px;">Índice de Fatiga</div>
              <div style="font-size:15px;font-weight:700;color:{color};margin-top:8px;">{row['estado']}</div>
              <div style="font-size:12px;color:#94a3b8;margin-top:8px;line-height:1.5;">{row['accion']}</div>
              <div style="font-size:11px;color:#475569;margin-top:10px;">Última sesión: {row.get('ultima_fecha','—')}</div>
              <div style="font-size:11px;color:#94a3b8;margin-top:5px;">DQI: {row.get('dqi',0):.2f} ({calidad_badge})</div>
              <div style="font-size:11px;color:#38bdf8;margin-top:5px;">SWC personal: ±{row.get('swc_personal',0):.4f} m/s</div>
            </div>
            """, unsafe_allow_html=True)

        with col_vars:
            st.markdown("#### Variables del modelo")
            d1, d2, d3 = st.columns(3)
            d1.metric(
                "ACWR", f"{row['acwr']:.3f}",
                help="Ratio carga aguda/crónica (MMA7/MMC28). Ideal 0.88–1.20."
            )
            d2.metric(
                "Δ% vs MMC28", f"{row['delta_pct']:+.1f}%",
                help="Caída de VMP hoy respecto al baseline crónico de 28 días (MMC28). "
                     "v4.1: antes se comparaba contra MMA7, lo que sesgaba el resultado."
            )
            d3.metric(
                "Z-Score Mesociclo", f"{row['z_meso']:+.2f}",
                help="Desviación estándar del rendimiento de hoy respecto al mesociclo completo."
            )

            d4, d5, d6 = st.columns(3)
            d4.metric(
                "β₇ (Tendencia semanal)", f"{row['beta_aguda']:+.4f}",
                help="Pendiente de VMP en los últimos 7 días. Positivo = mejorando."
            )
            d5.metric(
                "β₂₈ (Tendencia mensual)", f"{row['beta_28']:+.4f}",
                help="Pendiente de VMP en los últimos 28 días. Adaptación al plan."
            )
            d6.metric(
                "Sesiones consecutivas ↓", int(row.get("n_sesiones_desc", 0)),
                help="Sesiones seguidas con VMP decreciente. ≥3 activa aviso de tendencia SNC."
            )

            st.markdown(
                f"**Variable de entrada:** VMP fase propulsiva CMJ  ·  "
                f"**Baseline:** MMC28 = `{row['mmc28']:.3f}` m/s  ·  "
                f"**SWC personal:** `{row.get('swc_personal',0):.4f}` m/s  ·  "
                f"**Sesiones totales:** {int(row['n_sesiones'])}"
            )

        st.markdown("#### Evolución VMP del CMJ")
        st.pyplot(fig_tendencia(m))

        with st.expander("📅 Ver historial de sesiones (últimas 20)"):
            sub = df_raw[df_raw["Nombre"] == sel][["Fecha", "VMP_Hoy"]].tail(20)
            st.dataframe(
                sub.sort_values("Fecha", ascending=False)
                   .style.format({"VMP_Hoy": "{:.3f}"}),
                use_container_width=True, hide_index=True,
            )

    with st.expander("📐 Ver Funciones de Pertenencia del Modelo"):
        st.pyplot(fig_membership(vars_tuple))

    return df_res


# =============================================================================
#  SECCIÓN 9 — TAB: INGRESO DE DATOS
# =============================================================================

def tab_ingreso(atletas_lista: list[str], df_raw: pd.DataFrame):
    st.markdown("### ➕ Registrar Sesión")
    st.caption(
        "**Variable registrada:** VMP de la fase propulsiva del CMJ (m/s). "
        "No se registra RSImod ni tiempo de contacto."
    )
    st.markdown('<div class="form-card">', unsafe_allow_html=True)

    col1, col2, col3 = st.columns([2, 2, 2])
    with col1:
        atleta_sel = st.selectbox("Atleta", atletas_lista, key="form_atleta")
    with col2:
        fecha_sel  = st.date_input("Fecha", value=date.today(), key="form_fecha",
                                   max_value=date.today())
    with col3:
        vmp_val = st.number_input(
            "VMP Fase Propulsiva CMJ (m/s)",
            min_value=0.100, max_value=4.999,
            value=0.500, step=0.001, format="%.3f", key="form_vmp",
            help="Velocidad media en la fase concéntrica del salto. "
                 "Rango fisiológico normal CMJ libre: 0.80–2.50 m/s."
        )

    # Sanity check VMP CMJ (documento base §3)
    if vmp_val > 2.50:
        st.warning(
            f"⚠️ VMP {vmp_val:.3f} m/s es inusualmente alta para CMJ libre. "
            "Verifica que el sensor esté midiendo la fase propulsiva correcta."
        )

    notas_val = st.text_input("Notas (opcional)", key="form_notas",
                               placeholder="Observaciones de la sesión...")
    st.markdown('</div>', unsafe_allow_html=True)

    if not df_raw.empty:
        ya_existe = (
            (df_raw["Nombre"] == atleta_sel) &
            (df_raw["Fecha"]  == pd.Timestamp(fecha_sel))
        ).any()
        if ya_existe:
            st.warning(
                f"⚠️ Ya existe un registro para **{atleta_sel}** el **{fecha_sel}**. "
                "Guarda solo si deseas agregar una segunda sesión en el mismo día."
            )

    if st.button("💾 Guardar Sesión", type="primary"):
        ok, msg = insertar_sesion(atleta_sel, fecha_sel, vmp_val, notas_val)
        if ok:
            st.success(msg)
            st.cache_data.clear()
        else:
            st.error(msg)

    st.markdown("---")
    st.markdown("### ⚡ Registro Rápido Multi-Atleta (mismo día)")
    st.caption(
        "VMP fase propulsiva CMJ para cada atleta. Deja en **0.000** a quienes no participaron."
    )

    fecha_multi = st.date_input("Fecha de la sesión", value=date.today(),
                                 max_value=date.today(), key="multi_fecha")
    n_cols = 3
    rows   = [atletas_lista[i:i+n_cols] for i in range(0, len(atletas_lista), n_cols)]

    vmp_multi: dict[str, float] = {}
    for fila in rows:
        cols_ui = st.columns(n_cols)
        for col, nombre in zip(cols_ui, fila):
            with col:
                val = st.number_input(
                    nombre, min_value=0.0, max_value=4.999, value=0.0,
                    step=0.001, format="%.3f", key=f"multi_{nombre}",
                    help="0.000 = no participó (se omite)"
                )
                if val > 0:
                    vmp_multi[nombre] = val

    if st.button("💾 Guardar Todos", type="primary", key="multi_save"):
        if not vmp_multi:
            st.warning("No hay valores VMP ingresados.")
        else:
            errores = []
            for nombre, vmp in vmp_multi.items():
                ok, msg = insertar_sesion(nombre, fecha_multi, vmp)
                if not ok:
                    errores.append(f"{nombre}: {msg}")
            if errores:
                st.warning("Algunos registros no se pudieron guardar:\n" + "\n".join(errores))
            else:
                st.success(f"✅ {len(vmp_multi)} sesiones guardadas correctamente.")
                st.cache_data.clear()


# =============================================================================
#  SECCIÓN 10 — TAB: HISTORIAL Y EDICIÓN
# =============================================================================

def tab_historial(df_raw: pd.DataFrame, atletas_lista: list[str]):
    st.markdown("### 📝 Editar / Eliminar Sesiones")

    col1, col2 = st.columns([2, 3])
    with col1:
        atleta_ed = st.selectbox("Atleta", atletas_lista, key="ed_atleta")
    with col2:
        fecha_desde = st.date_input("Desde", value=date.today() - timedelta(days=30),
                                     key="ed_desde")

    sub = df_raw[
        (df_raw["Nombre"] == atleta_ed) &
        (df_raw["Fecha"]  >= pd.Timestamp(fecha_desde))
    ].sort_values("Fecha", ascending=False)

    if sub.empty:
        st.info("No hay registros en el rango seleccionado.")
        return

    if "notas" not in sub.columns:
        sub = sub.copy()
        sub["notas"] = ""

    sub_display = sub[["Fecha", "VMP_Hoy", "notas", "id"]].copy()
    sub_display["Fecha"] = sub_display["Fecha"].dt.strftime("%Y-%m-%d")

    st.dataframe(
        sub_display.rename(columns={
            "Fecha": "Fecha", "VMP_Hoy": "VMP CMJ (m/s)", "notas": "Notas", "id": "ID"
        }).style.format({"VMP CMJ (m/s)": "{:.3f}"}),
        use_container_width=True, hide_index=True,
    )

    st.markdown("---")
    opciones = {
        f"{row['Fecha'].strftime('%Y-%m-%d')} · {row['VMP_Hoy']:.3f} m/s": row["id"]
        for _, row in sub.iterrows()
    }
    sel_label = st.selectbox("Selecciona sesión a modificar:", list(opciones.keys()), key="ed_sel")
    sel_id    = opciones[sel_label]
    sel_row   = sub[sub["id"] == sel_id].iloc[0]

    col_e1, col_e2 = st.columns([2, 3])
    with col_e1:
        nuevo_vmp = st.number_input(
            "Nuevo VMP CMJ (m/s)", min_value=0.100, max_value=4.999,
            value=float(sel_row["VMP_Hoy"]), step=0.001, format="%.3f", key="ed_vmp"
        )
    with col_e2:
        nuevas_notas = st.text_input(
            "Notas", value=str(sel_row.get("notas", "") or ""), key="ed_notas"
        )

    col_btn1, col_btn2, _ = st.columns([1, 1, 3])
    with col_btn1:
        if st.button("✏️ Actualizar", type="primary"):
            ok, msg = actualizar_sesion(sel_id, nuevo_vmp, nuevas_notas)
            if ok:
                st.success(msg)
                st.cache_data.clear()
                st.rerun()
            else:
                st.error(msg)
    with col_btn2:
        if st.button("🗑️ Eliminar", type="secondary"):
            ok, msg = eliminar_sesion(sel_id)
            if ok:
                st.success(msg)
                st.cache_data.clear()
                st.rerun()
            else:
                st.error(msg)


# =============================================================================
#  SECCIÓN 11 — TAB: IMPORTACIÓN MASIVA
# =============================================================================

def tab_importacion():
    st.markdown("### 📤 Importar desde CSV / Excel")
    st.markdown("""
El archivo debe tener mínimo estas tres columnas (nombres exactos o equivalentes detectados automáticamente):

| Columna | Ejemplo | Descripción |
|---------|---------|-------------|
| `Nombre` | Juanes | Nombre del atleta |
| `Fecha`  | 2025-03-15 | Fecha de la sesión |
| `VMP_Hoy` | 0.487 | VMP fase propulsiva CMJ (m/s) |
""")
    archivo = st.file_uploader("Subir archivo", type=["csv", "xlsx"])

    if archivo:
        try:
            df_imp = (
                pd.read_csv(archivo) if archivo.name.endswith(".csv")
                else pd.read_excel(archivo)
            )

            col_map = {}
            for c in df_imp.columns:
                cl = c.lower().strip()
                if "nombre" in cl or "atleta" in cl: col_map[c] = "Nombre"
                elif "fecha" in cl or "date"  in cl: col_map[c] = "Fecha"
                elif "vmp"   in cl or "vel"   in cl: col_map[c] = "VMP_Hoy"
            df_imp = df_imp.rename(columns=col_map)

            if not all(c in df_imp.columns for c in ["Nombre", "Fecha", "VMP_Hoy"]):
                st.error("No se encontraron las columnas requeridas. Verifica el archivo.")
                return

            # Sanity check masivo (documento base §3)
            anomalias = df_imp[df_imp["VMP_Hoy"] > 2.50]
            if not anomalias.empty:
                st.warning(
                    f"⚠️ {len(anomalias)} filas con VMP > 2.50 m/s detectadas. "
                    "Revisa que los datos correspondan a VMP del CMJ y no de otro ejercicio."
                )

            st.success(
                f"Archivo válido: {len(df_imp)} filas · "
                f"{df_imp['Nombre'].nunique()} atletas detectados."
            )
            st.dataframe(df_imp.head(10), use_container_width=True)

            if st.button("⬆️ Importar a Base de Datos", type="primary"):
                with st.spinner("Importando..."):
                    ins, omi, errs = importar_dataframe(df_imp)
                st.success(f"✅ Insertados: {ins} · Omitidos (duplicados): {omi}")
                if errs:
                    st.warning("Errores:\n" + "\n".join(errs))
                st.cache_data.clear()

        except Exception as e:
            st.error(f"Error al leer el archivo: {e}")


# =============================================================================
#  SECCIÓN 12 — MAIN
# =============================================================================

def main():
    st.markdown(
        "<h1 style='text-align:center;color:#38bdf8;'>⚡ Dashboard de Fatiga</h1>"
        "<h3 style='text-align:center;color:#64748b;margin-top:-10px;'>"
        "Club Tornados · Modelo Fuzzy Mamdani v4.1 · Filtro SWC Activo</h3>",
        unsafe_allow_html=True,
    )

    cfg = render_sidebar()

    with st.spinner("Conectando con base de datos..."):
        df_raw        = cargar_sesiones_cached()
        atletas_lista = cargar_atletas_cached()

    if df_raw.empty:
        st.warning(
            "Base de datos vacía. Ve a la pestaña **➕ Ingreso de Datos** "
            "para registrar las primeras sesiones."
        )
    else:
        n_atletas = df_raw["Nombre"].nunique()
        ultima    = df_raw["Fecha"].max().strftime("%d/%m/%Y")
        st.success(
            f"✅ **{len(df_raw)} registros** · **{n_atletas} atletas** · "
            f"Última sesión: **{ultima}**"
        )

    vars_tuple, simulador = construir_motor_fuzzy()

    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Dashboard",
        "➕ Ingreso de Datos",
        "✏️ Historial / Edición",
        "📤 Importar CSV",
    ])

    with tab1:
        if not df_raw.empty:
            tab_dashboard(df_raw, simulador, vars_tuple, cfg)
        else:
            st.info("Sin datos. Registra sesiones en **➕ Ingreso de Datos**.")

    with tab2:
        tab_ingreso(atletas_lista, df_raw)

    with tab3:
        if not df_raw.empty:
            tab_historial(df_raw, atletas_lista)
        else:
            st.info("No hay sesiones registradas aún.")

    with tab4:
        tab_importacion()

    st.markdown("---")
    st.caption(
        "Modelo Fuzzy Mamdani v4.1 · 5 variables · 23 reglas · Defuzzificación COG · "
        "Variable: VMP fase propulsiva CMJ (no RSImod) · "
        "Δ% calculado vs MMC28 (baseline crónico) · "
        "Filtro SWC dinámico (1.0×SD adultos / 1.5×SD <15 a) · "
        "Tendencia inter-sesión SNC (≥3 sesiones) · "
        "Umbrales pediátricos activos (VL ≤15%, SWC ×1.5) · "
        "Base científica: Jukic 2022 · González-Badillo 2022 · Moura 2023 · Weakley 2019"
    )


if __name__ == "__main__":
    main()
