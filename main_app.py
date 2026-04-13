import streamlit as st
import pandas as pd
from fuzzy.fuzzy_processor import ejecutar_calculo
from fuzzy.fuzzy_utils import normalizar_wellness
from data.supabase_client import obtener_clavadistas, guardar_fatiga, conectar_db

# Importamos TUS gráficas originales
from visualization.charts import fig_semaforo_historico

st.set_page_config(page_title="Dashboard Clavados v4.2", layout="wide")
st.title("🤸 Sistema de Monitoreo: Clavados v4.2")

# --- RECUPERACIÓN DE DATOS BASE ---
# Traemos a los atletas
lista_atleta = obtener_clavadistas()
df_atletas = pd.DataFrame(lista_atleta) if lista_atleta else pd.DataFrame()

# --- ESTRUCTURA DE PESTAÑAS (TABS) ---
tab1, tab2, tab3 = st.tabs(["📊 Dashboard Histórico", "📝 Carga de Datos", "⚙️ Configuración"])

# ---------------------------------------------------------
# PESTAÑA 1: DASHBOARD (AQUÍ USAMOS TU CHARTS.PY ORIGINAL)
# ---------------------------------------------------------
# --- DENTRO DE LA PESTAÑA 1: DASHBOARD ---
with tab1:
    st.header("📊 Historial de Rendimiento (VMP)")
    
    supabase = conectar_db()
    if supabase:
        # 1. Traemos los datos de la tabla sesiones_vmp
        res = supabase.table("sesiones_vmp").select("*").execute()
        df_vmp = pd.DataFrame(res.data)
        
        if not df_vmp.empty:
            # 2. TRADUCCIÓN DE COLUMNAS PARA TU CHARTS.PY
            # Convertimos vmp_hoy a numérico por si viene como texto
            df_vmp["vmp_hoy"] = pd.to_numeric(df_vmp["vmp_hoy"])
            
            # Renombramos para que tu charts.py encuentre lo que busca
            df_hist = df_vmp.rename(columns={
                "fecha": "fecha",
                "vmp_hoy": "score",  # Usamos la velocidad como el puntaje a graficar
                "nombre": "nombre"
            })
            
            # 3. CREACIÓN DE ESTADOS (Basado en la velocidad vmp_hoy)
            def clasificar_vmp(vmp):
                if vmp >= 1.50:
                    return "🟢 ÓPTIMO"
                elif vmp >= 1.30:
                    return "🟡 ALERTA"
                else:
                    return "🔴 CRÍTICO"
                
            df_hist["estado"] = df_hist["score"].apply(clasificar_vmp)

            # 4. LLAMADA A TUS GRÁFICAS ORIGINALES
            # Mostramos un selector para filtrar por deportista si quieres ver uno a la vez
            atleta_sel = st.selectbox("Seleccionar Deportista", df_hist["nombre"].unique())
            df_filtrado = df_hist[df_hist["nombre"] == atleta_sel]
            
            fig = fig_semaforo_historico(df_filtrado, titulo=f"Evolución VMP: {atleta_sel}")
            st.plotly_chart(fig, use_container_width=True)
            
            # Tabla opcional para ver los datos crudos
            with st.expander("Ver tabla de datos completa"):
                st.write(df_filtrado)
        else:
            st.info("No hay datos en 'sesiones_vmp'.")
# ---------------------------------------------------------
# PESTAÑA 2: CARGA DE DATOS
# ---------------------------------------------------------
with tab2:
    st.header("Cargar Nuevo Registro")
    col_a, col_b = st.columns([1, 1])
    
    with col_a:
        if not df_atletas.empty:
            # Selector de atleta
            atleta_form = st.selectbox("Atleta", df_atletas['nombre'], key="form_atleta")
            # Obtenemos el ID del atleta seleccionado
            id_form = df_atletas[df_atletas['nombre'] == atleta_form]['id'].values[0]
            
            wellness = st.slider("Wellness (1-5)", 1.0, 5.0, 3.0)
            vmp = st.number_input("Carga VMP", 0, 200, 100)
            
            # Cálculo del motor fuzzy
            w_norm = normalizar_wellness(wellness)
            resultado_fatiga = ejecutar_calculo(w_norm, vmp)
            
            st.metric("Resultado Calculado", f"{resultado_fatiga:.2f}%")
            
            if st.button("🚀 Guardar en la Nube"):
                # Guarda en supabase usando tu función
                guardar_fatiga(id_form, wellness, vmp, resultado_fatiga)
                st.success("¡Datos enviados correctamente!")
                st.rerun() # Recarga para que aparezca en el Dashboard
        else:
            st.error("Registra un atleta en Configuración primero.")

# ---------------------------------------------------------
# PESTAÑA 3: CONFIGURACIÓN
# ---------------------------------------------------------
with tab3:
    st.header("Gestión del Equipo")
    nuevo_nombre = st.text_input("Nombre del nuevo clavadista")
    if st.button("➕ Registrar Atleta") and nuevo_nombre:
        supabase = conectar_db()
        # Asegúrate de usar la tabla correcta de la imagen que enviaste antes (atletas o clavadistas)
        supabase.table("clavadistas").insert({"nombre": nuevo_nombre}).execute()
        st.success(f"{nuevo_nombre} añadido al equipo.")
        st.rerun()
        