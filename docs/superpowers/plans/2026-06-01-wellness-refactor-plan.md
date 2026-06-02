# Wellness Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restore Emoji scale (Legend) and implement Group Data Entry for Wellness in AppDivingCodex.

**Architecture:** Update `components/tab_wellness_registro.py` to handle mode switching (Individual/Group) and rendering of the legend, using `st.data_editor` for group input.

**Tech Stack:** Python, Streamlit, Pandas.

---

### Task 1: Add Wellness Legend to UI

**Files:**
- Modify: `components/tab_wellness_registro.py`

- [ ] **Step 1: Import wellness_legend and add to renderer**

```python
import streamlit as st
from datetime import date
from data.db import cargar_atletas, insertar_wellness
from components.wellness_legend import render_wellness_legend # Add this

def render_wellness_registro():
    st.subheader("Registro Wellness Hooper")
    render_wellness_legend() # Add this
    
    # ... rest of function ...
```

- [ ] **Step 2: Commit**

```bash
git add components/tab_wellness_registro.py
git commit -m "feat: show wellness legend in registro"
```

### Task 2: Implement Group Wellness Registration UI

**Files:**
- Modify: `components/tab_wellness_registro.py`

- [ ] **Step 1: Add Mode Switch and Logic**

```python
def render_wellness_registro():
    st.subheader("Registro Wellness Hooper")
    render_wellness_legend()
    
    mode = st.radio("Modo de Registro", ["Individual", "Grupal"], horizontal=True, key="mode_well")
    
    atletas = cargar_atletas()
    if not atletas:
        st.warning("No hay atletas activos para registrar.")
        return

    if mode == "Individual":
        # ... keep existing form logic here (you might need to indent/refactor) ...
        pass
    else:
        # Task 2 Implementation
        st.subheader("Registro Grupal")
        fecha_grupal = st.date_input("Fecha Grupal", date.today())
        
        df_grupal = pd.DataFrame({
            "Atleta": atletas,
            "Sueño": [4] * len(atletas),
            "Fatiga": [4] * len(atletas),
            "Estrés": [4] * len(atletas),
            "Dolor": [4] * len(atletas),
            "Humor": [4] * len(atletas)
        })
        
        edited_df = st.data_editor(df_grupal, use_container_width=True)
        # Save logic for Task 3
```

- [ ] **Step 2: Commit**

```bash
git add components/tab_wellness_registro.py
git commit -m "feat: add group mode switch for wellness"
```

### Task 3: Implement Group Wellness Save Logic

**Files:**
- Modify: `components/tab_wellness_registro.py`

- [ ] **Step 1: Add Save Logic**

```python
        # ... inside else of Task 2 ...
        if st.button("Guardar Sesiones Grupales"):
            for _, row in edited_df.iterrows():
                success, msg = insertar_wellness(
                    nombre=row["Atleta"],
                    fecha=fecha_grupal,
                    sueno=int(row["Sueño"]),
                    fatiga_hooper=int(row["Fatiga"]),
                    estres=int(row["Estrés"]),
                    dolor=int(row["Dolor"]),
                    humor=int(row["Humor"]),
                    notas=""
                )
                if not success:
                    st.error(f"Error guardando {row['Atleta']}: {msg}")
            st.success("Proceso de guardado grupal finalizado.")
            st.rerun()
```

- [ ] **Step 2: Commit**

```bash
git add components/tab_wellness_registro.py
git commit -m "feat: implement group save logic for wellness"
```

### Task 4: Verification

**Files:**
- Test: `tests/test_wellness_insertion.py` (Verify it exists or create)

- [ ] **Step 1: Run application and verify changes**

Run: `streamlit run app.py`
Expected:
1. Wellness tab shows legend.
2. Mode switch "Individual"/"Grupal" is present.
3. Group mode shows data editor with all athletes.
4. Saving works.
