# Registration Redesign Design Specification

## Overview
Enable both Individual and Group registration modes for "Registro Sesión" (VMP) and "Wellness (Hooper)" tabs.

## Functional Design

### 1. Toggle Mode
- Each registration tab will start with a radio button or toggle: **Individual** / **Group**.
- Default mode: Individual (current behavior).

### 2. Group Registration (VMP)
- **Auto-populate:** When "Group" is selected, the page loads a table containing all active athletes.
- **Table Structure:**
  - Column 1: Atleta (Name)
  - Column 2: VMP Hoy (Number Input)
  - Column 3: VMP Referencia (Number Input - Optional)
  - Column 4: Notas (Text Input)
- **Submission:** Single "Guardar Sesión Grupal" button at the bottom.

### 3. Group Registration (Wellness)
- **Auto-populate:** When "Group" is selected, the page loads a table containing all active athletes.
- **Table Structure (Wide):**
  - Column 1: Atleta (Name)
  - Columns 2-6: Sueño, Fatiga, Estrés, Dolor, Humor (Number Inputs 1-7)
  - Column 7: Notas (Text Input)
- **Submission:** Single "Guardar Wellness Grupal" button at the bottom.

## Implementation Plan
1. Refactor registration components to handle mode switching (Individual/Group).
2. Create new table-based entry components for group VMP and Wellness.
3. Update database submission logic to handle bulk inserts.
4. Ensure existing individual registration remains functional.
