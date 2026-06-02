# Dashboard Redesign Design Specification

## Overview
Redesign the "Dashboard Grupal" to improve visual clarity and data accessibility for monitoring athlete fatigue.

## Visual Design

### 1. Athlete Fatigue Indicators (Radial Gauges)
- **Container:** Grid layout, 5 columns by 2 rows.
- **Component:** Radial progress gauge per athlete.
  - **Center:** Fatigue percentage value.
  - **Ring:** Color-coded based on predefined fatigue thresholds.
  - **Label:** Athlete name centered below the gauge.
- **Logic:** Use existing predefined status and fatigue percentage values.

### 2. Detailed Data Table
- **Columns:**
  - Name
  - Date
  - Fatigue Status
  - SWRC (Smallest Worthwhile Change)
  - DQI (Data Quality Index)
  - Feedback (Predefined text)
- **Behavior:**
  - Row expands vertically to accommodate predefined feedback text.

## Implementation Plan
1. Update UI components for radial gauges.
2. Refactor Dashboard layout to use CSS Grid/Flexbox for 5x2 arrangement.
3. Update table rendering logic to allow row expansion for feedback content.
4. Integrate existing data services to populate these new UI elements.
