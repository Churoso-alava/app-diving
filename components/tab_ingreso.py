"""
components/tab_ingreso.py — NMF-Optimizer v4.4
Sub-pestañas de ingreso de datos: VMP · Wellness · Carga Grupal.
Sin lógica de negocio: solo UI + llamadas a data.db.
"""
from __future__ import annotations

from datetime import date

import pandas as pd
import streamlit as st

import data.db as db


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Estandariza columnas a snake_case para importación masiva."""
    df = df.copy()
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    return df


def render_tab_ingreso(atletas_lista: list[str]) -> None:
    """
    Renderiza la pestaña de ingreso de datos con 3 sub-tabs:
      🏃 Velocidad (VMP) — individual + importación CSV
      💤 Wellness        — individual (sliders) + masivo (data_editor)
      🏋️ Carga Grupal   — tabla de ejercicios → todos los atletas
    """
    sub_vel, sub_well, sub_carga = st.tabs([
        "🏃 Velocidad (VMP)",
        "💤 Wellness",
        "🏋️ Carga Grupal",
    ])

    # =========================================================================
    # VMP — Individual + CSV
    # =========================================================================
    with sub_vel:
        st.markdown("### ➕ Registrar Sesión VMP")
        st.caption("Velocidad media propulsiva ante carga submaximal fija (40–60% 1RM).")

        with st.expander("📂 Importación masiva CSV"):
            file_imp = st.file_uploader(
                "Subir CSV (nombre, fecha, vmp_hoy)", type=["csv"], key="imp_vmp_file"
            )
            if file_imp is not None:
                df_imp = _normalize_columns(pd.read_csv(file_imp))

                if len(df_imp) > db.MAX_IMPORT_ROWS:
                    st.error(
                        f"🚫 **{len(df_imp)} filas** superan el límite "
                        f"**{db.MAX_IMPORT_ROWS}**. Divide el CSV en lotes."
                    )
                else:
                    if "vmp_hoy" in df_imp.columns:
                        anomalias = df_imp[df_imp["vmp_hoy"] > 2.50]
                        if not anomalias.empty:
                            st.warning(f"⚠️ {len(anomalias)} filas con VMP > 2.50 m/s.")
                    st.info(f"Vista previa: {len(df_imp)} filas")
                    st.dataframe(df_imp.head(5), use_container_width=True, hide_index=True)

                    if st.button("📥 Importar VMP", key="btn_imp_vmp"):
                        ins, omi, errs = db.importar_dataframe(df_imp)
                        if errs:
                            st.error("\n".join(errs))
                        else:
                            st.success(f"✅ {ins} insertados, {omi} omitidos.")
                            st.cache_data.clear()
                            st.rerun()

        st.markdown("---")
        col_a, col_b = st.columns(2)
        with col_a:
            atleta_sel = st.selectbox("Atleta", atletas_lista, key="vmp_atleta")
        with col_b:
            fecha_vmp = st.date_input(
                "Fecha", value=date.today(), max_value=date.today(), key="vmp_fecha"
            )
        vmp_hoy_val = st.number_input(
            "VMP hoy (m/s)", min_value=0.10, max_value=2.50,
            value=0.80, step=0.01, format="%.3f", key="vmp_hoy_input"
        )
        notas_vmp = st.text_input("Notas (opcional)", key="vmp_notas")

        if st.button("💾 Guardar VMP", type="primary", key="btn_vmp"):
            ok, msg = db.insertar_sesion(atleta_sel, fecha_vmp, vmp_hoy_val, notas=notas_vmp)
            if ok:
                st.success(msg)
                st.cache_data.clear()
                st.rerun()
            else:
                st.error(msg)

    # =========================================================================
    # WELLNESS — Individual + Masivo
    # =========================================================================
    with sub_well:
        modo_well = st.radio(
            "Modalidad de registro",
            ["👤 Individual (sliders)", "👥 Masivo (tabla)"],
            horizontal=True,
            key="well_modo",
        )

        if modo_well == "👤 Individual (sliders)":
            st.markdown("### 💤 Cuestionario Wellness (Hooper Modificado)")
            st.caption(
                "Escala Likert 1–7 · Inversa: **Sueño / Fatiga / Estrés / Dolor** (1 = óptimo) · "
                "Directa: **Humor** (7 = óptimo)."
            )
            col_w0, col_w_fecha = st.columns(2)
            with col_w0:
                atleta_well = st.selectbox("Atleta", atletas_lista, key="well_atleta")
            with col_w_fecha:
                fecha_well = st.date_input(
                    "Fecha", value=date.today(), max_value=date.today(), key="well_fecha"
                )
            col_w1, col_w2, col_w3 = st.columns(3)
            with col_w1:
                w_sueno  = st.slider("😴 Sueño",  1, 7, 4, key="well_sueno")
                w_fatiga = st.slider("😓 Fatiga", 1, 7, 4, key="well_fatiga")
            with col_w2:
                w_estres = st.slider("😰 Estrés", 1, 7, 4, key="well_estres")
                w_dolor  = st.slider("🦵 Dolor",  1, 7, 4, key="well_dolor")
            with col_w3:
                w_humor = st.slider("😊 Humor",  1, 7, 4, key="well_humor")

            # Preview W_norm
            w_norm_preview = (
                (7 - w_sueno) + (7 - w_fatiga) + (7 - w_estres) +
                (7 - w_dolor) + (w_humor - 1)
            ) / (5 * 6)
            color_wn = (
                "#00C49A" if w_norm_preview >= 0.65
                else "#E67E22" if w_norm_preview >= 0.35
                else "#E74C3C"
            )
            st.markdown(
                f'<span style="font-size:13px;color:#8B949E;">W_norm preview: </span>'
                f'<span style="font-size:20px;font-weight:700;color:{color_wn};">'
                f'{w_norm_preview:.2f}</span>',
                unsafe_allow_html=True,
            )

            notas_well = st.text_input("Notas (opcional)", key="well_notas")

            if st.button("💾 Guardar Wellness", type="primary", key="btn_guardar_well"):
                ok, msg = db.insertar_wellness(
                    nombre=atleta_well, fecha=fecha_well,
                    sueno=w_sueno, fatiga_hooper=w_fatiga,
                    estres=w_estres, dolor=w_dolor, humor=w_humor, notas=notas_well,
                )
                if ok:
                    st.success(msg)
                    st.cache_data.clear()
                    st.rerun()
                else:
                    st.error(msg)

        else:  # Masivo
            st.markdown("### 👥 Registro Masivo de Wellness")

            with st.expander("📂 Importación masiva CSV Wellness"):
                file_well = st.file_uploader(
                    "Subir CSV Wellness", type=["csv"], key="imp_well_file"
                )
                if file_well is not None:
                    df_well_imp = pd.read_csv(file_well)
                    if len(df_well_imp) > db.MAX_IMPORT_ROWS:
                        st.error(
                            f"🚫 **{len(df_well_imp)} filas** superan el límite "
                            f"**{db.MAX_IMPORT_ROWS}**."
                        )
                    else:
                        st.info(f"Vista previa: {len(df_well_imp)} filas")
                        st.dataframe(df_well_imp.head(5), use_container_width=True)
                        if st.button("📥 Importar Wellness CSV", key="btn_imp_well"):
                            ins, omi, errs = db.importar_wellness_dataframe(df_well_imp)
                            if errs:
                                st.error("\n".join(errs))
                            else:
                                st.success(f"✅ {ins} insertados, {omi} omitidos.")
                                st.cache_data.clear()
                                st.rerun()

            fecha_masiva = st.date_input(
                "Fecha de la sesión", value=date.today(),
                max_value=date.today(), key="well_masiva_fecha",
            )
            df_editor = db.wellness_masivo_template(atletas_lista)
            df_editado = st.data_editor(
                df_editor,
                use_container_width=True,
                hide_index=True,
                num_rows="fixed",
                column_config={
                    "Nombre": st.column_config.TextColumn("Atleta", disabled=True),
                    "Sueño":  st.column_config.NumberColumn("😴 Sueño",  min_value=1, max_value=7, step=1),
                    "Estrés": st.column_config.NumberColumn("😰 Estrés", min_value=1, max_value=7, step=1),
                    "Fatiga": st.column_config.NumberColumn("😓 Fatiga", min_value=1, max_value=7, step=1),
                    "Humor":  st.column_config.NumberColumn("😊 Humor",  min_value=1, max_value=7, step=1),
                    "Dolor":  st.column_config.NumberColumn("🦵 Dolor",  min_value=1, max_value=7, step=1),
                },
                key="well_masiva_editor",
            )

            if st.button("💾 Guardar Wellness Masivo", type="primary", key="btn_well_masivo"):
                errores_w, insertados_w = [], 0
                for _, row in df_editado.iterrows():
                    ok, msg = db.insertar_wellness(
                        nombre=row["Nombre"], fecha=fecha_masiva,
                        sueno=int(row["Sueño"]), fatiga_hooper=int(row["Fatiga"]),
                        estres=int(row["Estrés"]), dolor=int(row["Dolor"]),
                        humor=int(row["Humor"]), notas="",
                    )
                    if ok:
                        insertados_w += 1
                    else:
                        errores_w.append(f"{row['Nombre']}: {msg}")
                if errores_w:
                    st.warning(f"⚠️ {len(errores_w)} errores:\n" + "\n".join(errores_w))
                else:
                    st.success(f"✅ Wellness guardado para {insertados_w} atletas.")
                    st.cache_data.clear()
                    st.rerun()

    # =========================================================================
    # CARGA GRUPAL
    # =========================================================================
    with sub_carga:
        st.markdown("### 🏋️ Carga Grupal de Entrenamiento")
        st.caption(
            "Define los ejercicios de la sesión. Se asocia a "
            "**todos los atletas seleccionados** para esa fecha."
        )

        col_c_fecha, col_c_notas = st.columns([2, 4])
        with col_c_fecha:
            fecha_carga = st.date_input(
                "Fecha", value=date.today(), max_value=date.today(), key="carga_fecha"
            )
        with col_c_notas:
            notas_carga = st.text_input(
                "Notas del entrenador", key="carga_notas",
                placeholder="Contexto de la sesión..."
            )

        st.markdown("#### Ejercicios de la sesión")
        df_ejercicios_base = pd.DataFrame({
            "tipo_plataforma": pd.Series([], dtype="str"),
            "altura_salto":    pd.Series([], dtype="float"),
            "n_saltos":        pd.Series([], dtype="int"),
            "tipo_caida":      pd.Series([], dtype="str"),
        })
        df_ejercicios = st.data_editor(
            df_ejercicios_base,
            use_container_width=True,
            hide_index=True,
            num_rows="dynamic",
            column_config={
                "tipo_plataforma": st.column_config.SelectboxColumn(
                    "Plataforma", options=["trampolín", "plataforma"], required=True,
                ),
                "altura_salto": st.column_config.NumberColumn(
                    "Altura (m)", min_value=0.5, max_value=15.0, step=0.5, required=True,
                ),
                "n_saltos": st.column_config.NumberColumn(
                    "N° Saltos", min_value=1, max_value=100, step=1, required=True,
                ),
                "tipo_caida": st.column_config.SelectboxColumn(
                    "Caída", options=["pie", "mano"], required=True,
                ),
            },
            key="carga_ejercicios_editor",
        )

        st.markdown("#### Atletas participantes")
        atletas_participantes = st.multiselect(
            "Selecciona los atletas:",
            options=atletas_lista,
            default=atletas_lista,
            key="carga_atletas_sel",
        )

        if not df_ejercicios.empty and atletas_participantes:
            total_saltos = int(df_ejercicios["n_saltos"].sum())
            st.metric("Total de saltos en la sesión", total_saltos)

            if st.button("💾 Guardar Carga Grupal", type="primary", key="btn_carga_grupal"):
                ok, errors = db.insertar_carga_grupal_batch(
                    fecha=str(fecha_carga),
                    df_ejercicios=df_ejercicios,
                    atletas=atletas_participantes,
                    notas=notas_carga,
                )
                if ok:
                    st.success(
                        f"✅ Carga grupal guardada para {len(atletas_participantes)} atletas "
                        f"({total_saltos} saltos)."
                    )
                    st.cache_data.clear()
                else:
                    st.error("❌ Errores:\n" + "\n".join(errors))
        else:
            st.info("Agrega al menos un ejercicio y selecciona atletas para guardar.")
