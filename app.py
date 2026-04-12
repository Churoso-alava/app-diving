"""
app.py — Interfaz Streamlit (solo UI) v4.2
Lógica de negocio → services.py | Motor difuso → fuzzy.py | Base de datos → db.py

Cambios v4.2 (correcciones de auditoría):
  [A1] Dead code _estado_from_score eliminado
  [A2] Imports diving_load / fuzzy_diving movidos al header
  [A3] fig_membership (matplotlib) → fig_membership_fuzzy (Plotly)
  [A4] Expander membresía fuzzy restringido a rol analitico (RBAC)
  [A5] st.cache_data.clear() + st.rerun() tras guardar Wellness
  [A6] calcular_historial_batch_cached: O(N³) → @st.cache_data(ttl=30)
  [A7] Botón 💾 Guardar Carga en sub_carga + insertar_carga_sesion
"""
import logging
import warnings
from datetime import date, timedelta

import numpy as np
import pandas as pd
import skfuzzy as fuzz
import streamlit as st

import db
import fuzzy as fz
from services import (
    SessionInput,
    calcular_metricas,
    detectar_tendencia_mpv,
    pipeline_batch,
    pipeline_historial,
)
from visualization.components import render_kpi_row, render_athlete_bars, render_athlete_profile
from visualization.charts import (
    fig_vmp_tendencia,
    fig_semaforo_historico,
    fig_historial_barras_atleta,
    fig_membership_fuzzy,           # [A3] Plotly — reemplaza matplotlib
)
# [A2] Imports estáticos de clavados — no en caliente dentro de funciones
from diving_load import (
    carga_bruta_sesion as _cbs,
    normalizar_carga as _nc,
    calcular_wellness as _cw_fn,
    carga_integrada as _ci_fn,
)
from fuzzy_diving import mf_ci, CONJUNTOS_CI, conjunto_dominante_ci

warnings.filterwarnings("ignore")

# =============================================================================
#  LOGGING
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# =============================================================================
#  CONFIGURACIÓN DE PÁGINA
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
  [data-testid="stAppViewContainer"] { background-color: #0D1117; }
  [data-testid="stSidebar"] { background-color: #161B22; }
  h1, h2, h3, p, label { color: #E6EDF3 !important; }
</style>
""", unsafe_allow_html=True)


# =============================================================================
#  AUTENTICACIÓN
# =============================================================================

def check_password() -> bool:
    if st.session_state.get("password_correct"):
        return True

    st.write("🔒 Ingresa la contraseña para acceder")
    password = st.text_input("Contraseña:", type="password")

    try:
        expected = st.secrets["APP_PASSWORD"]
    except KeyError:
        st.error("⚠️ APP_PASSWORD no configurado en secrets.toml")
        return False

    if password == expected:
        st.session_state.password_correct = True
        try:
            st.session_state.rol_usuario = st.secrets["ROL_USUARIO"]
        except KeyError:
            st.session_state.rol_usuario = "operativo"
        log.info("Login exitoso. Rol: %s", st.session_state.rol_usuario)
        st.rerun()
    elif password:
        st.error("❌ Contraseña incorrecta")
        return False
    return False


if not check_password():
    st.stop()


# =============================================================================
#  CARGA DE DATOS CON CACHÉ
# =============================================================================

@st.cache_data(ttl=30)
def cargar_sesiones_cached() -> pd.DataFrame:
    return db.cargar_sesiones()


@st.cache_data(ttl=30)
def cargar_atletas_cached() -> list:
    return db.cargar_atletas()


# =============================================================================
#  MOTOR FUZZY
# =============================================================================

@st.cache_resource
def construir_motor_fuzzy_cached():
    return fz.construir_motor_fuzzy()


# =============================================================================
#  [A6] HISTORIAL BATCH CACHEADO — O(N³) → @st.cache_data(ttl=30)
# =============================================================================

@st.cache_data(ttl=30, show_spinner=False)
def calcular_historial_batch_cached(
    df_raw: pd.DataFrame,
    atletas: tuple,          # tuple = hashable para st.cache_data
    ventana_meso: int,
) -> dict:
    """
    Calcula el historial de fatiga para todos los atletas y lo cachea 30 s.
    Evita el bloqueo O(N³) en cada render del dashboard.

    El simulador se obtiene desde cache_resource interno para no incluir
    objetos no-hashables en la clave de caché de st.cache_data.
    """
    _vars_tuple, _sim = construir_motor_fuzzy_cached()
    result = {}
    for atleta in atletas:
        df_h = pipeline_historial(df_raw, atleta, _sim, ventana_meso)
        if not df_h.empty:
            result[atleta] = df_h
    return result


# =============================================================================
#  SIDEBAR
# =============================================================================

def render_sidebar() -> dict:
    with st.sidebar:
        st.markdown("## ⚡ Club Tornados")
        st.markdown("**Dashboard de Fatiga v4.2**")
        st.divider()

        if st.button("🔄 Actualizar Datos Ahora", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
        st.caption("Caché: refresco automático cada 30 s.")
        st.divider()

        st.markdown("### ⚙️ Parámetros del Modelo")
        rol = st.session_state.get("rol_usuario", "operativo")
        if rol == "analitico":
            ventana_meso = st.slider(
                "Ventana Mesociclo — Z-score (días)", 14, 42, 28,
                help="Ventana reciente para Z-score. NO usar historial total.",
            )
        else:
            ventana_meso = 28
            st.caption("⚙️ **Ventana mesociclo:** 28 días (fija).\nParámetro bloqueado en perfil operativo.")

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
#  TAB: DASHBOARD
# =============================================================================

_STATUS_MAP = {
    "🔴": "CRÍTICO",
    "🟠": "FATIGA ACUMULADA",
    "🟡": "ALERTA TEMPRANA",
    "🟢": "ÓPTIMO",
}


def _clean_estado(estado_raw: str) -> str:
    for emoji, label in _STATUS_MAP.items():
        if emoji in str(estado_raw):
            return label
    return str(estado_raw)


# [A1] _estado_from_score eliminada — era dead code


def _preparar_df_atletas(df_res: pd.DataFrame) -> list[dict]:
    resultado = []
    for _, row in df_res.iterrows():
        resultado.append({
            "nombre": row["atleta"],
            "score":  float(row["indice_fatiga"]),
            "estado": _clean_estado(row.get("estado", "")),
            "fecha":  str(row.get("ultima_fecha", "")),
        })
    return resultado


def tab_dashboard(df_raw: pd.DataFrame, simulador, vars_tuple, cfg: dict):
    atletas = sorted(df_raw["Nombre"].unique())
    df_res  = pipeline_batch(df_raw, simulador, cfg["ventana_meso"])

    if df_res.empty:
        st.warning("No hay atletas con suficientes sesiones (mínimo 4).")
        return

    # ── KPI Cards ────────────────────────────────────────────────────────────
    total    = len(df_res)
    criticos = int((df_res["indice_fatiga"] < 25).sum())
    fatiga   = int(((df_res["indice_fatiga"] >= 25) & (df_res["indice_fatiga"] < 50)).sum())
    alerta   = int(((df_res["indice_fatiga"] >= 50) & (df_res["indice_fatiga"] < 75)).sum())
    optimos  = int((df_res["indice_fatiga"] >= 75).sum())

    render_kpi_row(
        total=total,
        criticos=criticos,
        fatiga_acum=fatiga,
        alerta_temp=alerta,
        optimos=optimos,
    )

    # ── Semáforo ──────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("## 🚦 Semáforo de Fatiga — Todos los Atletas")
    render_athlete_bars(_preparar_df_atletas(df_res))

    # ── [A6] Histórico individual por atleta — cacheado ───────────────────────
    st.markdown("---")
    st.markdown("## 📊 Historial de Fatiga — Barras por Atleta")
    st.caption("Últimas 12 sesiones evaluadas · Líneas de referencia: 25 (crítico) · 50 (alerta) · 75 (óptimo)")
    try:
        with st.spinner("Calculando historial de fatiga..."):
            frames_dict = calcular_historial_batch_cached(
                df_raw,
                tuple(sorted(atletas)),   # tuple hashable para cache_data
                cfg["ventana_meso"],
            )

        if frames_dict:
            _n_cols = 3
            _atletas_list = list(frames_dict.keys())
            for _chunk in [_atletas_list[i:i + _n_cols] for i in range(0, len(_atletas_list), _n_cols)]:
                _grid_cols = st.columns(_n_cols)
                for _col, _ath in zip(_grid_cols, _chunk):
                    with _col:
                        _df_h = frames_dict[_ath].copy()
                        _df_h["fecha"] = _df_h["fecha"].astype(str)
                        st.plotly_chart(
                            fig_historial_barras_atleta(_df_h, _ath),
                            use_container_width=True,
                            config={"displayModeBar": False},
                        )
        else:
            st.info("El historial estará disponible cuando haya múltiples sesiones registradas.")
    except Exception as e:
        log.warning("No se pudo renderizar histórico: %s", e)
        st.info("El historial estará disponible cuando haya múltiples sesiones registradas.")

    # ── Tabla de resultados ───────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("## 📋 Tabla de Resultados")
    modo_analitico = st.toggle(
        "🔬 Modo Analítico (variables del modelo)",
        value=st.session_state.get("modo_tabla_analitico", False),
        key="modo_tabla_analitico",
        help="Activa para ver ACWR, β₇, β₂₈, Z-score, SWC y DQI.",
    )
    cols = (
        ["atleta", "vmp_hoy", "mmc28", "acwr", "delta_pct", "z_meso",
         "beta_aguda", "beta_28", "swc_personal", "es_ruido_biologico",
         "dqi", "calidad_dato", "indice_fatiga", "estado", "accion_primaria", "ultima_fecha"]
        if modo_analitico else
        ["atleta", "accion_primaria", "advertencias", "indice_fatiga", "estado", "ultima_fecha"]
    )
    rename_map = {
        "atleta": "Atleta", "vmp_hoy": "VMP Hoy", "mmc28": "MMC28 (base)",
        "acwr": "ACWR", "delta_pct": "Δ% vs MMC28", "z_meso": "Z Meso",
        "beta_aguda": "β₇", "beta_28": "β₂₈", "swc_personal": "SWC",
        "es_ruido_biologico": "Ruido Bio.", "dqi": "DQI", "calidad_dato": "Calidad",
        "indice_fatiga": "Índice", "estado": "Estado",
        "accion_primaria": "Acción", "advertencias": "Alertas", "ultima_fecha": "Última Sesión",
    }
    df_t = df_res[cols].copy().rename(columns=rename_map).sort_values("Índice")
    if "Alertas" in df_t.columns:
        df_t["Alertas"] = df_t["Alertas"].apply(
            lambda v: " · ".join(v) if isinstance(v, list) and v else "—"
        )
    fmt = {"Índice": "{:.1f}"}
    if modo_analitico:
        fmt.update({
            "VMP Hoy": "{:.3f}", "MMC28 (base)": "{:.3f}", "ACWR": "{:.3f}",
            "Δ% vs MMC28": "{:+.1f}%", "Z Meso": "{:+.2f}",
            "β₇": "{:+.4f}", "β₂₈": "{:+.4f}", "SWC": "{:.4f}", "DQI": "{:.2f}",
        })
    st.dataframe(df_t.style.format(fmt), use_container_width=True, hide_index=True)
    st.download_button(
        "⬇️ Descargar resultados (CSV)",
        data=df_t.to_csv(index=False).encode("utf-8"),
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

        sub_atleta = df_raw[df_raw["Nombre"] == sel]
        if detectar_tendencia_mpv(sub_atleta, ventana=3):
            st.warning(
                "📉 **Tendencia VMP descendente en 3 sesiones consecutivas.** "
                "Proxy de deterioro gradual del SNC — revisar carga semanal. "
                "*(Weakley 2019)*"
            )

        if row.get("nota_swc"):
            st.info(f"🔵 **Filtro SWC:** {row['nota_swc']}")

        render_athlete_profile(
            nombre=sel,
            posicion=row.get("posicion", "Jugador"),
            disponible=bool(row.get("activo", True)),
            indice_fatiga=float(row["indice_fatiga"]),
            estado=_clean_estado(row["estado"]),
            recomendacion=row.get("accion_primaria", row.get("accion", "—")),
            ultima_sesion=str(row.get("ultima_fecha", "—")),
            metricas={
                "acwr":            {"valor": float(row["acwr"]),            "estado": ""},
                "delta_pct":       {"valor": float(row["delta_pct"]),       "estado": ""},
                "z_meso":          {"valor": float(row["z_meso"]),          "estado": ""},
                "beta7":           {"valor": float(row["beta_aguda"]),      "estado": ""},
                "beta28":          {"valor": float(row["beta_28"]),         "estado": ""},
                "sesiones_consec": {"valor": int(row.get("n_sesiones_desc", 0)), "estado": ""},
                "dqi":             {"valor": float(row.get("dqi", 0)),      "estado": row.get("calidad_dato", "")},
            },
        )

        # ── Gráfico VMP ───────────────────────────────────────────────────────
        st.markdown("#### Evolución VMP del CMJ")
        vmp       = np.array(m["historial"])
        fechas_dt = pd.to_datetime(m["fechas"])
        serie     = pd.Series(vmp, index=fechas_dt)
        mma7s     = serie.rolling("7D",  min_periods=3).mean().values
        mmc28s    = serie.rolling("28D", min_periods=7).mean().values

        df_atleta_plot = pd.DataFrame({
            "fecha":   fechas_dt,
            "vmp_hoy": vmp,
            "mma7":    mma7s,
            "mmc28":   mmc28s,
        })
        st.plotly_chart(
            fig_vmp_tendencia(
                df=df_atleta_plot,
                nombre_atleta=sel,
                delta_pct=float(m.get("delta_pct", 0)),
            ),
            use_container_width=True,
        )

        with st.expander("📅 Ver historial de sesiones (últimas 20)"):
            sub = df_raw[df_raw["Nombre"] == sel][["Fecha", "VMP_Hoy"]].tail(20)
            st.dataframe(
                sub.sort_values("Fecha", ascending=False).style.format({"VMP_Hoy": "{:.3f}"}),
                use_container_width=True,
                hide_index=True,
            )

    # [A3][A4] Funciones de Pertenencia — solo rol analitico, Plotly
    if st.session_state.get("rol_usuario") == "analitico":
        with st.expander("📐 Ver Funciones de Pertenencia del Modelo"):
            _acwr_v, _delta_v, _zmeso_v, _ba_v, _b28_v, _fat_v = vars_tuple
            u_fat = _fat_v.universe
            membership_vals = {
                "Óptimo":  fuzz.interp_membership(u_fat, _fat_v["optimo"].mf,           u_fat),
                "Alerta":  fuzz.interp_membership(u_fat, _fat_v["alerta_temprana"].mf,  u_fat),
                "Fatiga":  fuzz.interp_membership(u_fat, _fat_v["fatiga_acumulada"].mf, u_fat),
                "Crítico": fuzz.interp_membership(u_fat, _fat_v["critico"].mf,          u_fat),
            }
            st.plotly_chart(
                fig_membership_fuzzy(u_fat, membership_vals),
                use_container_width=True,
            )

    return df_res


# =============================================================================
#  TAB: INGRESO DE DATOS
# =============================================================================

def tab_ingreso(atletas_lista: list[str], df_raw: pd.DataFrame):
    """
    3 sub-pestañas independientes:
      1. 🏃 Velocidad (VMP)
      2. 💤 Wellness (Hooper Modificado)
      3. 🏋️ Carga Entrenamiento (Clavados + CI)
    """
    sub_vel, sub_well, sub_carga = st.tabs([
        "🏃 Velocidad (VMP)",
        "💤 Wellness",
        "🏋️ Carga Entrenamiento",
    ])

    # =========================================================================
    #  SUB-TAB 1 — VELOCIDAD (VMP)
    # =========================================================================
    with sub_vel:
        st.markdown("### ➕ Registrar Sesión VMP")
        st.caption(
            "**Variable registrada:** VMP de la fase propulsiva del CMJ (m/s). "
            "No se registra RSImod ni tiempo de contacto."
        )
        st.markdown('<div class="form-card">', unsafe_allow_html=True)

        col1, col2, col3 = st.columns([2, 2, 2])
        with col1:
            atleta_sel = st.selectbox("Atleta", atletas_lista, key="form_atleta")
        with col2:
            fecha_sel = st.date_input("Fecha", value=date.today(), key="form_fecha",
                                      max_value=date.today())
        with col3:
            vmp_val = st.number_input(
                "VMP Fase Propulsiva CMJ (m/s)",
                min_value=0.100, max_value=4.999,
                value=0.500, step=0.001, format="%.3f", key="form_vmp",
                help="Velocidad media fase concéntrica CMJ. Rango fisiológico: 0.80–2.50 m/s.",
            )

        if vmp_val > 2.50:
            st.error(
                f"🚫 VMP {vmp_val:.3f} m/s supera el límite fisiológico (2.50 m/s). "
                "Verifica el sensor. **El sistema rechazará este valor.**"
            )
        elif vmp_val > 1.80:
            st.warning(
                f"⚠️ VMP {vmp_val:.3f} m/s es inusualmente alta para CMJ libre. "
                "¿El sensor está midiendo la fase propulsiva correcta?"
            )

        notas_val = st.text_input("Notas (opcional)", key="form_notas",
                                  placeholder="Observaciones de la sesión...")
        st.markdown('</div>', unsafe_allow_html=True)

        if not df_raw.empty:
            ya_existe = (
                (df_raw["Nombre"] == atleta_sel) &
                (df_raw["Fecha"] == pd.Timestamp(fecha_sel))
            ).any()
            if ya_existe:
                st.warning(
                    f"⚠️ Ya existe un registro para **{atleta_sel}** el **{fecha_sel}**. "
                    "Guarda solo si deseas agregar una segunda sesión en el mismo día."
                )

        if st.button("💾 Guardar Sesión", type="primary", key="btn_guardar_vmp"):
            ok, msg = db.insertar_sesion(atleta_sel, fecha_sel, vmp_val, notas_val)
            if ok:
                st.success(msg)
                st.cache_data.clear()
            else:
                st.error(msg)

        st.markdown("---")
        st.markdown("### ⚡ Registro Rápido Multi-Atleta (mismo día)")
        st.caption("VMP fase propulsiva CMJ para cada atleta. Deja en **0.000** a quienes no participaron.")

        fecha_multi = st.date_input("Fecha de la sesión", value=date.today(),
                                    max_value=date.today(), key="multi_fecha")
        n_cols = 3
        rows   = [atletas_lista[i:i + n_cols] for i in range(0, len(atletas_lista), n_cols)]
        vmp_multi: dict[str, float] = {}
        for fila in rows:
            cols_ui = st.columns(n_cols)
            for col, nombre in zip(cols_ui, fila):
                with col:
                    val = st.number_input(
                        nombre, min_value=0.0, max_value=4.999, value=0.0,
                        step=0.001, format="%.3f", key=f"multi_{nombre}",
                        help="0.000 = no participó (se omite)",
                    )
                    if val > 0:
                        vmp_multi[nombre] = val

        if st.button("💾 Guardar Todos", type="primary", key="multi_save"):
            if not vmp_multi:
                st.warning("No hay valores VMP ingresados.")
            else:
                errores = []
                for nombre, vmp in vmp_multi.items():
                    ok, msg = db.insertar_sesion(nombre, fecha_multi, vmp)
                    if not ok:
                        errores.append(f"{nombre}: {msg}")
                if errores:
                    st.warning("Algunos registros no se pudieron guardar:\n" + "\n".join(errores))
                else:
                    st.success(f"✅ {len(vmp_multi)} sesiones guardadas correctamente.")
                    st.cache_data.clear()

    # =========================================================================
    #  SUB-TAB 2 — WELLNESS (Hooper Modificado)
    # =========================================================================
    with sub_well:
        st.markdown("### 💤 Cuestionario de Wellness (Hooper Modificado)")
        st.caption(
            "5 ítems en escala Likert 1–7. "
            "Escala inversa: **Sueño/Fatiga/Estrés/Dolor** → 1 = óptimo. "
            "Escala directa: **Humor** → 7 = óptimo."
        )

        st.markdown('<div class="form-card">', unsafe_allow_html=True)
        col_w0, col_w_fecha = st.columns([2, 2])
        with col_w0:
            atleta_well = st.selectbox("Atleta", atletas_lista, key="well_atleta")
        with col_w_fecha:
            fecha_well = st.date_input("Fecha", value=date.today(),
                                       max_value=date.today(), key="well_fecha")

        st.markdown("---")
        col_w1, col_w2, col_w3 = st.columns(3)
        with col_w1:
            w_sueno  = st.slider("😴 Sueño (1 = óptimo, 7 = pésimo)",  1, 7, 4, key="well_sueno")
            w_fatiga = st.slider("😓 Fatiga (1 = óptimo, 7 = máxima)",  1, 7, 4, key="well_fatiga")
        with col_w2:
            w_estres = st.slider("😰 Estrés (1 = óptimo, 7 = máximo)",  1, 7, 4, key="well_estres")
            w_dolor  = st.slider("🦵 Dolor muscular (1 = sin dolor)",    1, 7, 4, key="well_dolor")
        with col_w3:
            w_humor  = st.slider("😊 Humor (7 = óptimo, 1 = pésimo)",   1, 7, 4, key="well_humor")

        _w_s = (7 - w_sueno)  / 6.0
        _w_f = (7 - w_fatiga) / 6.0
        _w_e = (7 - w_estres) / 6.0
        _w_d = (7 - w_dolor)  / 6.0
        _w_h = (w_humor - 1)  / 6.0
        _w_norm_preview = (_w_s + _w_f + _w_e + _w_d + _w_h) / 5.0
        _color_w = (
            "#00C49A" if _w_norm_preview >= 0.65
            else "#E67E22" if _w_norm_preview >= 0.35
            else "#E74C3C"
        )
        st.markdown(
            f'<div style="margin-top:12px;">'
            f'<span style="font-size:13px;color:#8B949E;">W_norm (preview): </span>'
            f'<span style="font-size:20px;font-weight:700;color:{_color_w};">'
            f'{_w_norm_preview:.2f}</span>'
            f'<span style="font-size:11px;color:#8B949E;"> / 1.00</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

        notas_well = st.text_input("Notas (opcional)", key="well_notas",
                                   placeholder="Observaciones del atleta...")
        st.markdown('</div>', unsafe_allow_html=True)

        # [A5] Cache invalidation + rerun tras guardar Wellness
        if st.button("💾 Guardar Wellness", type="primary", key="btn_guardar_well"):
            ok, msg = db.insertar_wellness(
                nombre=atleta_well,
                fecha=fecha_well,
                sueno=w_sueno,
                fatiga_hooper=w_fatiga,
                estres=w_estres,
                dolor=w_dolor,
                humor=w_humor,
                notas=notas_well,
            )
            if ok:
                st.success(msg)
                st.cache_data.clear()
                st.rerun()
            else:
                st.error(msg)

        st.markdown("---")
        st.markdown("#### 📋 Últimos registros de Wellness")
        try:
            df_well_hist = db.cargar_wellness(atleta_well)
            if df_well_hist.empty:
                st.info("No hay registros de wellness para este atleta aún.")
            else:
                cols_show = [c for c in
                             ["fecha", "sueno", "fatiga_hooper", "estres", "dolor", "humor", "w_norm", "notas"]
                             if c in df_well_hist.columns]
                st.dataframe(
                    df_well_hist[cols_show].head(15).rename(columns={
                        "fecha": "Fecha", "sueno": "Sueño", "fatiga_hooper": "Fatiga",
                        "estres": "Estrés", "dolor": "Dolor", "humor": "Humor",
                        "w_norm": "W_norm", "notas": "Notas",
                    }).style.format({"W_norm": "{:.2f}"}),
                    use_container_width=True, hide_index=True,
                )
        except Exception as exc:
            st.warning(
                f"⚠️ No se pudo cargar el historial de wellness ({exc}). "
                "Verifica que la tabla `wellness` exista en Supabase."
            )

    # =========================================================================
    #  SUB-TAB 3 — CARGA DE ENTRENAMIENTO (Clavados)
    # =========================================================================
    with sub_carga:
        st.markdown("### 🏋️ Registro de Carga — Clavados (CI)")
        st.caption("Carga Integrada = L_norm × (2 − W_norm). Modelo FINA + Hooper. Ref: Pandey 2022.")

        st.markdown('<div class="form-card">', unsafe_allow_html=True)
        col_c0, col_c_fecha = st.columns([2, 2])
        with col_c0:
            atleta_carga = st.selectbox("Atleta", atletas_lista, key="carga_atleta")
        with col_c_fecha:
            fecha_carga = st.date_input("Fecha", value=date.today(),
                                        max_value=date.today(), key="carga_fecha")

        n_clavados_ui = st.number_input("Número de clavados ejecutados", min_value=0,
                                        max_value=30, value=0, step=1, key="n_clavados")
        clavados_input = []
        for i in range(int(n_clavados_ui)):
            st.markdown(f"**Clavado {i+1}**")
            col_h, col_dd, col_tipo = st.columns(3)
            with col_h:
                h = st.selectbox("Altura", [1.0, 3.0, 5.0, 7.5, 10.0],
                                 key=f"h_{i}", format_func=lambda x: f"{x}m")
            with col_dd:
                dd = st.number_input("DD (FINA)", min_value=1.2, max_value=4.4,
                                     value=2.0, step=0.1, key=f"dd_{i}")
            with col_tipo:
                tipo = st.selectbox("Tipo", ["HEAD", "FEET", "TWIST", "PIKE", "SYNC"],
                                   key=f"tipo_{i}")
            clavados_input.append({"altura": h, "dd": dd, "tipo": tipo})

        st.markdown("---")
        st.markdown("**Wellness vinculado a esta sesión (Hooper Modificado)**")
        col_cw1, col_cw2, col_cw3 = st.columns(3)
        with col_cw1:
            cw_sueno  = st.slider("Sueño (1=óptimo)",  1, 7, 4, key="cw_sueno")
            cw_fatiga = st.slider("Fatiga (1=óptimo)", 1, 7, 4, key="cw_fatiga")
        with col_cw2:
            cw_estres = st.slider("Estrés (1=óptimo)", 1, 7, 4, key="cw_estres")
            cw_dolor  = st.slider("Dolor (1=óptimo)",  1, 7, 4, key="cw_dolor")
        with col_cw3:
            cw_humor  = st.slider("Humor (7=óptimo)",  1, 7, 4, key="cw_humor")

        if clavados_input:
            # [A2] Variables ya importadas en el header — sin imports en caliente
            _l_bruta   = _cbs(clavados_input)
            _l_norm    = _nc(_l_bruta)
            _w_norm    = _cw_fn(sueno=cw_sueno, fatiga=cw_fatiga,
                                estres=cw_estres, dolor=cw_dolor, humor=cw_humor)
            _ci        = _ci_fn(_l_norm, _w_norm)
            _dominante = conjunto_dominante_ci(_ci)
            _color_map = {
                "RECUPERACION": "#2ecc71", "MANTENIMIENTO": "#f1c40f",
                "DESARROLLO": "#e67e22",   "SOBRECARGA": "#e74c3c",
            }
            _grado = mf_ci[_dominante](_ci)
            st.metric("Carga Integrada (CI)", f"{_ci:.1f} / 200",
                      delta=f"W_norm: {_w_norm:.2f} · L_norm: {_l_norm:.1f}%")
            st.markdown(
                f'<span style="color:{_color_map[_dominante]};font-weight:bold;">'
                f'Zona difusa dominante: {_dominante} (μ={_grado:.2f})</span>',
                unsafe_allow_html=True,
            )

            # [A7] Botón de guardado + persistencia en DB
            st.markdown("---")
            notas_carga = st.text_input(
                "Notas de la sesión (opcional)", key="carga_notas",
                placeholder="Contexto del entrenador...",
            )
            if st.button("💾 Guardar Carga", type="primary", key="btn_guardar_carga"):
                ok_c, msg_c = db.insertar_carga_sesion(
                    nombre=atleta_carga,
                    fecha=fecha_carga,
                    n_clavados=len(clavados_input),
                    l_bruta=_l_bruta,
                    l_norm=_l_norm,
                    w_norm=_w_norm,
                    ci=_ci,
                    zona_dominante=_dominante,
                    notas=notas_carga,
                )
                if ok_c:
                    st.success(msg_c)
                    st.cache_data.clear()
                else:
                    st.error(msg_c)
        else:
            st.info("Ingresa al menos 1 clavado para calcular y guardar la Carga Integrada.")

        st.markdown('</div>', unsafe_allow_html=True)


# =============================================================================
#  TAB: HISTORIAL Y EDICIÓN
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
        (df_raw["Fecha"] >= pd.Timestamp(fecha_desde))
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
            "Fecha": "Fecha", "VMP_Hoy": "VMP CMJ (m/s)", "notas": "Notas", "id": "ID",
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
            value=float(sel_row["VMP_Hoy"]), step=0.001, format="%.3f", key="ed_vmp",
        )
    with col_e2:
        nuevas_notas = st.text_input(
            "Notas", value=str(sel_row.get("notas", "") or ""), key="ed_notas",
        )

    if "confirm_delete_id" not in st.session_state:
        st.session_state.confirm_delete_id = None

    col_btn1, col_btn2, _ = st.columns([1, 1, 3])
    with col_btn1:
        if st.button("✏️ Actualizar", type="primary"):
            st.session_state.confirm_delete_id = None
            ok, msg = db.actualizar_sesion(sel_id, nuevo_vmp, nuevas_notas)
            if ok:
                st.success(msg); st.cache_data.clear(); st.rerun()
            else:
                st.error(msg)
    with col_btn2:
        if st.button("🗑️ Eliminar", type="secondary"):
            st.session_state.confirm_delete_id = sel_id

    if st.session_state.get("confirm_delete_id") == sel_id:
        st.warning(
            f"⚠️ **¿Confirmar eliminación?**  \n"
            f"Atleta: **{atleta_ed}** · "
            f"Fecha: **{sel_row['Fecha'].strftime('%Y-%m-%d')}** · "
            f"VMP: **{sel_row['VMP_Hoy']:.3f} m/s**  \n"
            "Esta acción es **irreversible**."
        )
        col_conf, col_cancel, _ = st.columns([1, 1, 3])
        with col_conf:
            if st.button("⚠️ Confirmar eliminación", type="primary", key="confirm_del_btn"):
                ok, msg = db.eliminar_sesion(sel_id)
                st.session_state.confirm_delete_id = None
                if ok:
                    st.success(msg); st.cache_data.clear(); st.rerun()
                else:
                    st.error(msg)
        with col_cancel:
            if st.button("✖ Cancelar", key="cancel_del_btn"):
                st.session_state.confirm_delete_id = None
                st.rerun()


# =============================================================================
#  TAB: IMPORTACIÓN MASIVA
# =============================================================================

def tab_importacion():
    sub_imp_vel, sub_imp_well = st.tabs(["🏃 Importar VMP", "💤 Importar Wellness"])

    with sub_imp_vel:
        st.markdown("### 📤 Importar VMP desde CSV / Excel")
        st.markdown("""
| Columna | Ejemplo | Descripción |
|---------|---------|-------------|
| `Nombre` | Juanes | Nombre del atleta |
| `Fecha`  | 2025-03-15 | Fecha de la sesión |
| `VMP_Hoy` | 0.487 | VMP fase propulsiva CMJ (m/s) |
""")
        archivo = st.file_uploader("Subir archivo VMP", type=["csv", "xlsx"], key="up_vmp")

        if archivo:
            try:
                df_imp = (
                    pd.read_csv(archivo) if archivo.name.endswith(".csv")
                    else pd.read_excel(archivo)
                )
                col_map = {}
                for c in df_imp.columns:
                    cl = c.lower().strip()
                    if "nombre" in cl or "atleta" in cl:
                        col_map[c] = "Nombre"
                    elif "fecha" in cl or "date" in cl:
                        col_map[c] = "Fecha"
                    elif "vmp" in cl or "vel" in cl:
                        col_map[c] = "VMP_Hoy"
                df_imp = df_imp.rename(columns=col_map)

                if not all(c in df_imp.columns for c in ["Nombre", "Fecha", "VMP_Hoy"]):
                    st.error("No se encontraron las columnas requeridas.")
                    return

                anomalias = df_imp[df_imp["VMP_Hoy"] > 2.50]
                if not anomalias.empty:
                    st.warning(f"⚠️ {len(anomalias)} filas con VMP > 2.50 m/s.")

                st.success(f"Archivo válido: {len(df_imp)} filas · {df_imp['Nombre'].nunique()} atletas.")
                st.dataframe(df_imp.head(10), use_container_width=True)

                if st.button("⬆️ Importar a Base de Datos", type="primary", key="btn_imp_vmp"):
                    with st.spinner("Importando..."):
                        ins, omi, errs = db.importar_dataframe(df_imp)
                    st.success(f"✅ Insertados: {ins} · Omitidos: {omi}")
                    if errs:
                        st.warning("Errores:\n" + "\n".join(errs))
                    st.cache_data.clear()

            except Exception as exc:
                log.error("tab_importacion VMP error: %s", exc)
                st.error(f"Error al leer el archivo: {exc}")

    with sub_imp_well:
        st.markdown("### 📤 Importar Wellness desde CSV / Excel")
        _plantilla_well = (
            "Nombre,Fecha,Sueno,Fatiga,Estres,Dolor,Humor,Notas\n"
            "Juanes,2025-03-15,2,3,2,1,6,Post-partido\n"
            "Maria,2025-03-15,4,5,3,3,4,\n"
        )
        st.download_button("⬇️ Descargar plantilla Wellness CSV",
                           data=_plantilla_well, file_name="plantilla_wellness.csv",
                           mime="text/csv", key="dl_plantilla_well")

        archivo_well = st.file_uploader("Subir archivo Wellness", type=["csv", "xlsx"], key="up_well")
        if archivo_well:
            try:
                df_well_imp = (
                    pd.read_csv(archivo_well) if archivo_well.name.endswith(".csv")
                    else pd.read_excel(archivo_well)
                )
                st.info(f"Vista previa: {len(df_well_imp)} filas")
                st.dataframe(df_well_imp.head(10), use_container_width=True)

                if st.button("⬆️ Importar Wellness a Base de Datos", type="primary", key="btn_imp_well"):
                    with st.spinner("Importando wellness..."):
                        ins, omi, errs = db.importar_wellness_dataframe(df_well_imp)
                    st.success(f"✅ Insertados: {ins} · Omitidos: {omi}")
                    if errs:
                        st.warning("Errores:\n" + "\n".join(errs[:20]))

            except Exception as exc:
                log.error("tab_importacion Wellness error: %s", exc)
                st.error(f"Error al leer el archivo: {exc}")


# =============================================================================
#  MAIN
# =============================================================================

def main():
    st.markdown(
        "<h1 style='text-align:center;color:#38bdf8;'>⚡ Dashboard de Fatiga</h1>"
        "<h3 style='text-align:center;color:#64748b;margin-top:-10px;'>"
        "Club Tornados · Modelo Fuzzy Mamdani v4.2 · Filtro SWC Activo</h3>",
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

    vars_tuple, simulador = construir_motor_fuzzy_cached()

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
        "Modelo Fuzzy Mamdani v4.2 · 5 variables · 23 reglas · Defuzzificación COG · "
        "Variable: VMP fase propulsiva CMJ (no RSImod) · "
        "Δ% calculado vs MMC28 (baseline crónico) · "
        "Filtro SWC dinámico (1.0×SD adultos / 1.5×SD <15 a) · "
        "Base científica: Jukic 2022 · González-Badillo 2022 · Moura 2023 · Weakley 2019"
    )


if __name__ == "__main__":
    main()
