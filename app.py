"""
app.py — Interfaz Streamlit (solo UI)
Lógica de negocio → services.py | Motor difuso → fuzzy.py | Base de datos → db.py
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
from services import SessionInput, calcular_metricas, detectar_tendencia_mpv

# ── Nuevos módulos de visualización ──────────────────────────────────────────
# ── Módulos de visualización (importar directamente desde raíz) ──────────────
# TEMPORALMENTE comentado - limpiar después
# from visualization.themes import get_global_css

# Usar CSS básico en lugar de archivo
def get_global_css() -> str:
    return """
<style>
[data-testid="stAppViewContainer"] {
    background-color: #0D1117;
}
[data-testid="stSidebar"] {
    background-color: #161B22;
}
h1, h2, h3, p, label {
    color: #E6EDF3 !important;
}
</style>
"""

# Imports deshabilitados - usar más tarde
# from visualization.charts import fig_vmp_tendencia, ...
# from visualization.components import render_kpi_row, ...

#error de sintaxis
#except ModuleNotFoundError:
    # Si falla, intenta desde archivos en raíz
    #print("⚠️ No se encontró paquete visualization/, intentando imports alternativos...")
    #try:
     #   from themes import get_global_css
      #  from charts import (
            fig_vmp_tendencia,
            fig_semaforo_barras,
            fig_semaforo_historico,
            fig_membership_fuzzy,
       # )
        #from components import (
         #   render_kpi_row,
            render_athlete_bars,
            render_athlete_profile,
        #)
        #print("✓ Imports desde raíz funcionan")
    #except ModuleNotFoundError as e:
     #   print(f"❌ Error crítico en imports: {e}")
      #  raise

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

# ── CSS: base legacy + nuevo glassmorphism ────────────────────────────────────
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

# Inyectar CSS glassmorphism dark del módulo de visualización
st.markdown(get_global_css(), unsafe_allow_html=True)


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
#  GRÁFICOS LEGACY (solo fig_membership se mantiene con matplotlib)
# =============================================================================

def fig_membership(vars_tuple):
    """Funciones de membresía — se mantiene matplotlib por complejidad del motor fuzzy."""
    import matplotlib.pyplot as plt
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
#  SIDEBAR
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

def _preparar_df_atletas(df_res: pd.DataFrame) -> list[dict]:
    """Convierte df_res al formato que esperan render_athlete_bars."""
    STATUS_MAP = {
        "🔴": "CRÍTICO",
        "🟠": "FATIGA ACUMULADA",
        "🟡": "ALERTA TEMPRANA",
        "🟢": "ÓPTIMO",
    }
    resultado = []
    for _, row in df_res.iterrows():
        estado_raw = row.get("estado", "")
        # Extraer estado limpio del string "🟢 ÓPTIMO" o similar
        estado_clean = estado_raw
        for emoji, label in STATUS_MAP.items():
            if emoji in estado_raw:
                estado_clean = label
                break
        resultado.append({
            "nombre": row["atleta"],
            "score":  float(row["indice_fatiga"]),
            "estado": estado_clean,
            "fecha":  str(row.get("ultima_fecha", "")),
        })
    return resultado


def tab_dashboard(df_raw: pd.DataFrame, simulador, vars_tuple, cfg: dict):
    atletas    = sorted(df_raw["Nombre"].unique())
    metricas_l = [calcular_metricas(df_raw, a, cfg["ventana_meso"]) for a in atletas]
    metricas_l = [m for m in metricas_l if m]
    resultados = [fz.evaluar_atleta(simulador, m) for m in metricas_l]
    df_res     = pd.DataFrame(resultados)

    total    = len(df_res)
    criticos = int((df_res["indice_fatiga"] < 25).sum())
    fatiga   = int(((df_res["indice_fatiga"] >= 25) & (df_res["indice_fatiga"] < 50)).sum())
    alerta   = int(((df_res["indice_fatiga"] >= 50) & (df_res["indice_fatiga"] < 75)).sum())
    optimos  = int((df_res["indice_fatiga"] >= 75).sum())

    # ── KPI Cards (nuevo diseño) ──────────────────────────────────────────────
    render_kpi_row(
        total=total,
        criticos=criticos,
        fatiga_acum=fatiga,
        alerta_temp=alerta,
        optimos=optimos,
    )

    st.markdown("---")
    st.markdown("## 🚦 Semáforo de Fatiga — Todos los Atletas")

    # ── Barras de atletas en 2 columnas (nuevo diseño) ────────────────────────
    atletas_lista_ui = _preparar_df_atletas(df_res)
    render_athlete_bars(atletas_lista_ui)

    # ── Histórico de semáforo (NUEVO — no existía) ────────────────────────────
    st.markdown("---")
    st.markdown("## 📈 Línea Histórica de Semáforo")
    try:
        df_hist = df_raw.copy()
        # Calcular métricas históricas por atleta y fecha para el gráfico
        # Usamos df_res como proxy del estado actual; para histórico completo
        # se necesitaría iterar sesión a sesión — aquí mostramos lo disponible.
        df_hist_plot = df_res[["atleta", "indice_fatiga", "estado", "ultima_fecha"]].copy()
        df_hist_plot = df_hist_plot.rename(columns={
            "atleta": "nombre",
            "indice_fatiga": "score",
            "ultima_fecha": "fecha",
        })
        # Limpiar estado para que coincida con STATUS_COLOR
        STATUS_MAP = {
            "🔴": "CRÍTICO", "🟠": "FATIGA ACUMULADA",
            "🟡": "ALERTA TEMPRANA", "🟢": "ÓPTIMO",
        }
        def _clean_estado(e):
            for emoji, label in STATUS_MAP.items():
                if emoji in str(e):
                    return label
            return str(e)
        df_hist_plot["estado"] = df_hist_plot["estado"].apply(_clean_estado)
        df_hist_plot["fecha"]  = df_hist_plot["fecha"].astype(str)
        df_hist_plot = df_hist_plot.dropna(subset=["score", "fecha"])
        st.plotly_chart(fig_semaforo_historico(df_hist_plot), use_container_width=True)
    except Exception as e:
        log.warning("No se pudo renderizar histórico de semáforo: %s", e)
        st.info("El gráfico histórico estará disponible cuando haya múltiples sesiones registradas.")

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

        m["estado"]        = row["estado"]
        m["color"]         = row["color"]
        m["indice_fatiga"] = row["indice_fatiga"]
        m["mmc28"]         = row["mmc28"]

        sub_atleta = df_raw[df_raw["Nombre"] == sel]
        if detectar_tendencia_mpv(sub_atleta, ventana=3):
            st.warning(
                "📉 **Tendencia VMP descendente en 3 sesiones consecutivas.** "
                "Proxy de deterioro gradual del SNC — revisar carga semanal. "
                "*(Weakley 2019)*"
            )

        if row.get("nota_swc"):
            st.info(f"🔵 **Filtro SWC:** {row['nota_swc']}")

        color = row["color"]
        calidad_badge = {
            "alta": "🟢 Confianza Alta", "media": "🟡 Confianza Media",
            "baja": "🟠 Confianza Baja", "insuficiente": "🔴 Datos Insuficientes",
        }.get(row.get("calidad_dato", "media"), "")

        advertencias_html = "".join(
            f'<div style="font-size:12px;color:#f87171;margin-top:6px;'
            f'background:#2d1b1b;border-radius:6px;padding:4px 8px;">{adv}</div>'
            for adv in (row.get("advertencias") or [])
        )

        # ── Panel de perfil atleta (nuevo diseño) ─────────────────────────────
        STATUS_MAP_CLEAN = {
            "🔴": "CRÍTICO", "🟠": "FATIGA ACUMULADA",
            "🟡": "ALERTA TEMPRANA", "🟢": "ÓPTIMO",
        }
        estado_clean = row["estado"]
        for emoji, label in STATUS_MAP_CLEAN.items():
            if emoji in str(row["estado"]):
                estado_clean = label
                break

        render_athlete_profile(
            nombre=sel,
            posicion=row.get("posicion", "Jugador"),
            disponible=bool(row.get("activo", True)),
            indice_fatiga=float(row["indice_fatiga"]),
            estado=estado_clean,
            recomendacion=row.get("accion_primaria", row.get("accion", "—")),
            ultima_sesion=str(row.get("ultima_fecha", "—")),
            metricas={
                "acwr":            {"valor": float(row["acwr"]),       "estado": ""},
                "delta_pct":       {"valor": float(row["delta_pct"]),  "estado": ""},
                "z_meso":          {"valor": float(row["z_meso"]),     "estado": ""},
                "beta7":           {"valor": float(row["beta_aguda"]), "estado": ""},
                "beta28":          {"valor": float(row["beta_28"]),    "estado": ""},
                "sesiones_consec": {"valor": int(row.get("n_sesiones_desc", 0)), "estado": ""},
            },
        )

        # ── Gráfico VMP (nuevo Plotly interactivo) ────────────────────────────
        st.markdown("#### Evolución VMP del CMJ")
        vmp    = np.array(m["historial"])
        fechas = [str(f)[:10] for f in m["fechas"]]
        mma7s  = pd.Series(vmp).rolling(7,  min_periods=3).mean().values
        mmc28s = pd.Series(vmp).rolling(28, min_periods=7).mean().values

        df_atleta_plot = pd.DataFrame({
            "fecha": fechas,
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
                use_container_width=True, hide_index=True,
            )

    with st.expander("📐 Ver Funciones de Pertenencia del Modelo"):
        st.pyplot(fig_membership(vars_tuple))

    return df_res


# =============================================================================
#  TAB: INGRESO DE DATOS
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
            (df_raw["Fecha"]  == pd.Timestamp(fecha_sel))
        ).any()
        if ya_existe:
            st.warning(
                f"⚠️ Ya existe un registro para **{atleta_sel}** el **{fecha_sel}**. "
                "Guarda solo si deseas agregar una segunda sesión en el mismo día."
            )

    if st.button("💾 Guardar Sesión", type="primary"):
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
    rows   = [atletas_lista[i:i+n_cols] for i in range(0, len(atletas_lista), n_cols)]
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
                if "nombre" in cl or "atleta" in cl:  col_map[c] = "Nombre"
                elif "fecha" in cl or "date"  in cl:  col_map[c] = "Fecha"
                elif "vmp"   in cl or "vel"   in cl:  col_map[c] = "VMP_Hoy"
            df_imp = df_imp.rename(columns=col_map)

            if not all(c in df_imp.columns for c in ["Nombre", "Fecha", "VMP_Hoy"]):
                st.error("No se encontraron las columnas requeridas. Verifica el archivo.")
                return

            anomalias = df_imp[df_imp["VMP_Hoy"] > 2.50]
            if not anomalias.empty:
                st.warning(
                    f"⚠️ {len(anomalias)} filas con VMP > 2.50 m/s detectadas. "
                    "Revisa que los datos correspondan a VMP del CMJ."
                )

            st.success(
                f"Archivo válido: {len(df_imp)} filas · "
                f"{df_imp['Nombre'].nunique()} atletas detectados."
            )
            st.dataframe(df_imp.head(10), use_container_width=True)

            if st.button("⬆️ Importar a Base de Datos", type="primary"):
                with st.spinner("Importando..."):
                    ins, omi, errs = db.importar_dataframe(df_imp)
                st.success(f"✅ Insertados: {ins} · Omitidos: {omi}")
                if errs:
                    st.warning("Errores:\n" + "\n".join(errs))
                st.cache_data.clear()

        except Exception as exc:
            log.error("tab_importacion error: %s", exc)
            st.error(f"Error al leer el archivo: {exc}")


# =============================================================================
#  MAIN
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

    vars_tuple, simulador = construir_motor_fuzzy_cached()

    if "tab_activa" not in st.session_state:
        st.session_state.tab_activa = 0

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
        "Base científica: Jukic 2022 · González-Badillo 2022 · Moura 2023 · Weakley 2019"
    )


if __name__ == "__main__":
    main()
