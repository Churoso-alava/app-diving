# Individual Analysis Tab Redesign Specification

## Overview
Redesign the "Individual Analysis" tab to improve readability and visual interpretation of athlete data.

## Visual Design

### 1. Updated Trend Charts (VMP, Fatigue, Wellness)
- **Common Styling:** Smoothed lines (3-session SMA) with markers.
- **VMP Chart:** VMP trend line + safety thresholds. MMA7/MMC28 series removed.
- **Fatigue Chart:** Shift from bars to trend line. Mapped to current status thresholds (Óptimo, Alerta, Crítico).
- **Wellness Chart:** New trend line chart for wellness metrics + safety thresholds.

### 2. Biomechanical Variables Table
- **Layout:** Static table (no expanders or tabs).
- **Columns:** Variable, Value, Conceptual Formula, Brief Explanation.
- **Context:** Scientific context text rendered as fixed, always-visible content below the table.

## Implementation Plan
1. Update `ui/charts.py` to add/refactor chart functions for smoothed lines.
2. Update `app.py` to integrate the new charts and replace the biomechanical metrics section with the new static table structure.
3. Update `core/services.py` if necessary to provide wellness metrics in a format suitable for the new chart.
