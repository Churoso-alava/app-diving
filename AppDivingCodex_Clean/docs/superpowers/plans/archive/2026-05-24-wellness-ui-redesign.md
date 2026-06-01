# Wellness UI Redesign Design Specification

## Overview
Redesign the "Wellness (Hooper)" registration tab to be more intuitive for children while maintaining data integrity for the staff.

## Functional Design

### 1. Global Legend (Reference)
- **Placement:** Always visible at the top of the Wellness tab.
- **Content:** Mapping of 1-7 numeric scale to emoji states (1=Óptimo, 7=Crítico).
  - 1: 😃 (Óptimo)
  - 4: 😐 (Neutral)
  - 7: 😵 (Crítico)

### 2. Individual Registration Mode
- **Component:** `st.select_slider` for each variable (Sueño, Fatiga, Estrés, Dolor, Humor).
- **Functionality:** Allows visual selection using the emoji scale while clearly showing the numerical value (1-7).

### 3. Group Registration Mode
- **Component:** `st.data_editor` (Wide table).
- **Functionality:** Numeric input (1-7) per variable per athlete.
- **Design:** Simple numeric input to ensure staff efficiency; rely on the Global Legend for athlete guidance.

## Implementation Plan
1. Refactor `components/tab_wellness_registro.py` to add the Global Legend panel.
2. Update the Individual mode to use `st.select_slider`.
3. Update the Group mode to maintain the numeric table input, ensuring the global legend is clearly positioned above it.
