# Training Load Chart Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement a new time-series chart in the "Análisis Individual" tab showing daily training load as bars with an overlaid trend curve.

**Architecture:**
1.  Implement a new chart function `fig_carga_entrenamiento` in `ui/charts_redesign.py` that utilizes `plotly` to render bar charts for load and a smoothed trend line using the existing `_apply_savgol` utility.
2.  Update `app.py` (Tab 2: Análisis Individual) to calculate daily load (RPE * Duration) for the selected athlete from `df_raw`, and pass this data to the new chart function.

**Tech Stack:**
- Streamlit
- Plotly
- Pandas

---

### Task 1: Implement `fig_carga_entrenamiento`

**Files:**
- Modify: `ui/charts_redesign.py`

- [ ] **Step 1: Implement `fig_carga_entrenamiento` in `ui/charts_redesign.py`**

```python
def fig_carga_entrenamiento(df: pd.DataFrame, nombre_atleta: str) -> go.Figure:
    """
    Gráfico de barras para carga diaria (RPE * Duración) con curva de tendencia suavizada.
    """
    df = df.copy()
    df["fecha"] = pd.to_datetime(df["fecha"])
    # Asegurar que la carga está calculada
    if "carga_interna" not in df.columns:
        df["carga_interna"] = df["carga_subjetiva"] * df["duracion_min"]
    
    df = df.sort_values("fecha")
    df["carga_smooth"] = _apply_savgol(df, "carga_interna")

    fig = go.Figure()
    
    # Barras de carga diaria
    fig.add_trace(go.Bar(
        x=df["fecha"], 
        y=df["carga_interna"], 
        name="Carga Total (UA)",
        marker_color=COLORS["primary_brand"],
        opacity=0.6
    ))
    
    # Curva de tendencia
    fig.add_trace(go.Scatter(
        x=df["fecha"], 
        y=df["carga_smooth"], 
        mode="lines", 
        name="Tendencia (Suavizado)",
        line=dict(color="#ffffff", width=2)
    ))

    fig.update_layout(**_DARK_LAYOUT, title=f"Carga de Entrenamiento — {nombre_atleta}")
    return fig
```

- [ ] **Step 2: Commit**

```bash
git add ui/charts_redesign.py
git commit -m "feat: add fig_carga_entrenamiento chart function"
```

### Task 2: Integrate chart in `app.py`

**Files:**
- Modify: `app.py`

- [ ] **Step 1: Update `app.py` to prepare data and call chart**

Find the section in "Análisis Individual" tab (Tab 2) and add the chart integration:

```python
# ... (existing chart code)
                    # Gráficos (Rediseñados - Apilados)
                    df_atleta_history = get_vmp_history(df_raw, atleta_sel)
                    df_hist = calcular_historial_fatiga(df_raw, atleta_sel, simulador)
                    df_wellness = get_wellness_history(atleta_sel)

                    # --- Nueva Gráfica de Carga ---
                    df_atleta_raw = df_raw[df_raw["nombre"] == atleta_sel].copy()
                    df_atleta_raw["carga_interna"] = df_atleta_raw["carga_subjetiva"] * df_atleta_raw["duracion_min"]
                    st.plotly_chart(fig_carga_entrenamiento(df_atleta_raw, atleta_sel), use_container_width=True)
                    # -----------------------------
                    
                    # Pasar los umbrales de seguridad desde 'res'
                    st.plotly_chart(fig_vmp_tendencia_redesign(df_atleta_history, atleta_sel, res["mmc28"]), use_container_width=True)
# ...
```

- [ ] **Step 2: Commit**

```bash
git add app.py
git commit -m "feat: integrate training load chart in tab 2"
```

### Task 3: Verification

- [ ] **Step 1: Manual verification**

Run the app and navigate to "Análisis Individual". Select an athlete and verify the new "Carga de Entrenamiento" chart appears with bars and a trend line.

```bash
streamlit run app.py
```
