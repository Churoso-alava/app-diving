#!/bin/bash

################################################################################
# COORDINATED IMPLEMENTATION ORCHESTRATOR - PHASES 3-5
# 
# Executes the app-diving stabilization plan using Gemini CLI with:
#  - Swarm coordination for parallel task execution
#  - Memory persistence across phases
#  - Progressive validation gates
#  - Atomic commits per task
#  - Risk mitigation patterns
#
# Usage: ./execute-phases-3-5.sh [--repo PATH] [--dry-run] [--verbose]
################################################################################

set -euo pipefail

# ============================================================================
# CONFIGURATION & GLOBALS
# ============================================================================

REPO_URL="https://github.com/Churoso-alava/app-diving.git"
REPO_PATH="${1:-.}"
DRY_RUN="${DRY_RUN:-false}"
VERBOSE="${VERBOSE:-false}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/logs"
MEMORY_FILE="${SCRIPT_DIR}/PHASE_EXECUTION.md"
STATE_FILE="${SCRIPT_DIR}/.execution-state.json"

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Counters & tracking
TASKS_COMPLETED=0
TASKS_FAILED=0
PHASE_START_TIME=0

mkdir -p "${LOG_DIR}"

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

log() {
    local level=$1; shift
    local msg="$*"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo -e "${timestamp} [${level}] ${msg}" | tee -a "${LOG_DIR}/execution.log"
}

log_section() {
    echo -e "\n${BLUE}═══════════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}$*${NC}"
    echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}\n"
    log "INFO" "$*"
}

log_success() {
    echo -e "${GREEN}✓ $*${NC}"
    log "SUCCESS" "$*"
    ((TASKS_COMPLETED++))
}

log_error() {
    echo -e "${RED}✗ $*${NC}"
    log "ERROR" "$*"
    ((TASKS_FAILED++))
}

log_warn() {
    echo -e "${YELLOW}⚠ $*${NC}"
    log "WARN" "$*"
}

log_info() {
    echo -e "${BLUE}ℹ $*${NC}"
    log "INFO" "$*"
}

# Save state to JSON for recovery
save_state() {
    local phase=$1 task=$2 status=$3
    cat > "${STATE_FILE}" <<EOF
{
  "phase": ${phase},
  "last_task": "${task}",
  "status": "${status}",
  "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "tasks_completed": ${TASKS_COMPLETED},
  "tasks_failed": ${TASKS_FAILED}
}
EOF
}

# Execute with dry-run support
run_cmd() {
    local cmd=$1
    if [[ "${VERBOSE}" == "true" ]]; then
        log_info "Executing: ${cmd}"
    fi
    if [[ "${DRY_RUN}" == "true" ]]; then
        log_info "[DRY RUN] ${cmd}"
        return 0
    fi
    eval "${cmd}"
}

# ============================================================================
# VALIDATION GATES (from coordination-plan.json)
# ============================================================================

validate_gate_t3() {
    log_info "Validating T3 gate: test_security_hardening.py::TestDependencyPinning"
    run_cmd "cd '${REPO_PATH}' && python -m pytest tests/test_security_hardening.py::TestDependencyPinning -v"
    if [ $? -eq 0 ]; then
        log_success "T3 gate: 2 tests PASSED"
        return 0
    else
        log_error "T3 gate FAILED"
        return 1
    fi
}

validate_gate_t4() {
    log_info "Validating T4 gate: test_services.py"
    run_cmd "cd '${REPO_PATH}' && python -m pytest tests/test_services.py -v"
    if [ $? -eq 0 ]; then
        log_success "T4 gate: test_services.py all classes PASSED"
        return 0
    else
        log_error "T4 gate FAILED"
        return 1
    fi
}

validate_gate_t6() {
    log_info "Validating T6 gate: btn_carga_grupal must not exist in tab_ingreso.py"
    run_cmd "cd '${REPO_PATH}' && grep -q btn_carga_grupal components/tab_ingreso.py"
    if [ $? -ne 0 ]; then
        log_success "T6 gate: btn_carga_grupal fully removed"
        return 0
    else
        log_error "T6 gate FAILED: btn_carga_grupal still present"
        return 1
    fi
}

validate_gate_final() {
    log_info "Validating final gate: full test suite"
    run_cmd "cd '${REPO_PATH}' && python -m pytest tests/ -v --tb=short 2>&1 | tee '${LOG_DIR}/final_test_results.log'"
    if [ $? -eq 0 ]; then
        log_success "Final gate: >=60 tests PASSED, 0 FAILED"
        return 0
    else
        log_error "Final gate FAILED"
        cat "${LOG_DIR}/final_test_results.log" | tail -20
        return 1
    fi
}

# ============================================================================
# PHASE 3: TEST CORRECTIONS (with swarm coordination)
# ============================================================================

phase_3_orchestrate() {
    log_section "PHASE 3: TEST CORRECTIONS & COMPONENT FIXES"
    PHASE_START_TIME=$(date +%s)
    save_state 3 "orchestration_start" "in_progress"

    # Initialize Gemini memory for phase tracking
    log_info "Initializing Gemini CLI memory for phase 3..."
    gemini /memory add "PHASE 3 CONTEXT: Fixing test imports, column names, and button key. Three parallel subtasks (T3a, T3b, T3c). Validation gate: test_services.py + test_diving_load.py + btn_carga_grupal grep."

    # Prepare swarm context
    cat > "${LOG_DIR}/phase3-context.md" <<'CONTEXT'
# Phase 3: Test Corrections & Component Fixes

## Architecture Context
- Stable modules (data/db.py, logic/services.py, fuzzy/*): READ-ONLY
- Snake_case contract: DataFrames use [nombre, fecha, vmp_hoy, ...]
- Requirements pinned to test expectations: streamlit 1.38.0, pandas 2.1.4, etc.

## Parallel Tasks (Execute simultaneously with swarm)

### T3a: Fix test_services.py
**What:** Rewrite imports and column names to match snake_case contract
**Status:** BLOCKED on T3 setup completion
- Change `from data.db import ...` → `from logic.services import ...`
- Update all DataFrame column references: `df['Nombre']` → `df['nombre']`
- Ensure test fixtures create DataFrames with snake_case keys
- Validation: `pytest tests/test_services.py -v` → all classes PASSED

### T3b: Fix test_diving_load.py  
**What:** Correct import path to logic.biomechanics
**Status:** BLOCKED on T3 setup completion
- Change `from biomechanics import ...` → `from logic.biomechanics import ...`
- Update test fixtures to use snake_case column contract
- Validation: `pytest tests/test_diving_load.py -v` → 24 PASSED

### T3c: Rename button key in tab_ingreso.py
**What:** btn_carga_grupal → btn_guardar_carga
**Status:** BLOCKED on T3 setup completion
- Single string replacement in components/tab_ingreso.py
- Ensure state key names match
- Validation: `grep btn_carga_grupal components/tab_ingreso.py` → no output (exit 1)

## Gate Validation
- After all three tasks complete in parallel:
  1. `pytest tests/test_services.py -v` (20+ tests)
  2. `pytest tests/test_diving_load.py -v` (24 tests)
  3. `grep -q btn_carga_grupal components/tab_ingreso.py` (must return 1)
- All three MUST PASS to proceed to phase 4

## Regression Guard
- Before and after each subtask: `pytest tests/test_chart_logic.py tests/test_dataframe_utils.py`
- These must remain green (0 modifications allowed)
CONTEXT

    # Swarm dispatch: parallel execution of T3 subtasks
    log_info "Dispatching Phase 3 tasks to swarm coordination..."
    
    # Task T3a: Fix test_services.py
    (
        log_info "→ T3a: Fixing test_services.py imports and columns"
        save_state 3 "T3a" "in_progress"
        gemini /swarm-advanced <<'T3A'
/agents enable subagent-driven-development
/swarm-advanced coordinate --mode parallel --priority high

@tests/test_services.py

Task: Fix test_services.py for snake_case column contract
1. Change import from "from data.db import" to "from logic.services import SessionInput, calcular_metricas"
2. Locate all DataFrame creation in test fixtures
3. Replace PascalCase columns [Nombre, Fecha, VMP_Hoy] → snake_case [nombre, fecha, vmp_hoy]
4. Update assertions: df['Nombre'] → df['nombre']
5. Run: pytest tests/test_services.py -v before commit
6. Commit atomically: git add tests/test_services.py && git commit -m "T3a: Fix test_services imports and column contract"

Constraints:
- MUST use exact column names from data.db.py contract
- Do NOT modify data/db.py, logic/services.py, or logic/biomechanics.py
- Column contract: [nombre, fecha, vmp_hoy, vmp_ref, notas, created_at]
T3A
        if [ $? -eq 0 ]; then
            log_success "T3a: test_services.py fixed"
            save_state 3 "T3a" "completed"
        else
            log_error "T3a: FAILED"
            save_state 3 "T3a" "failed"
        fi
    ) &
    P_T3A=$!

    # Task T3b: Fix test_diving_load.py
    (
        log_info "→ T3b: Fixing test_diving_load.py imports"
        save_state 3 "T3b" "in_progress"
        gemini /swarm-advanced <<'T3B'
/agents enable subagent-driven-development
/swarm-advanced coordinate --mode parallel --priority high

@tests/test_diving_load.py

Task: Fix test_diving_load.py import path
1. Change import from "from biomechanics import" to "from logic.biomechanics import"
2. Verify all test fixtures use snake_case columns [nombre, fecha, vmp_hoy, vmp_ref, notas]
3. Check that SessionInput dataclass matches expected input contract
4. Run: pytest tests/test_diving_load.py -v before commit
5. Expected result: 24 PASSED
6. Commit atomically: git add tests/test_diving_load.py && git commit -m "T3b: Fix test_diving_load import path to logic.biomechanics"

Constraints:
- Do NOT modify logic/biomechanics.py itself
- Column names MUST match logic/services.py expectations
T3B
        if [ $? -eq 0 ]; then
            log_success "T3b: test_diving_load.py fixed"
            save_state 3 "T3b" "completed"
        else
            log_error "T3b: FAILED"
            save_state 3 "T3b" "failed"
        fi
    ) &
    P_T3B=$!

    # Task T3c: Fix button key in tab_ingreso.py
    (
        log_info "→ T3c: Fixing btn_carga_grupal key in tab_ingreso.py"
        save_state 3 "T3c" "in_progress"
        gemini /swarm-advanced <<'T3C'
/swarm-advanced coordinate --mode parallel --priority high

@components/tab_ingreso.py

Task: Rename button key btn_carga_grupal → btn_guardar_carga
1. Find all occurrences of "btn_carga_grupal" in file
2. Replace with "btn_guardar_carga" (exact match, case-sensitive)
3. Check for state key references that depend on old name
4. Verify button still calls correct callback function
5. Run: grep btn_carga_grupal components/tab_ingreso.py && echo "FOUND" || echo "SUCCESS"
6. Expected: grep returns exit 1 (not found)
7. Commit atomically: git add components/tab_ingreso.py && git commit -m "T3c: Rename button key btn_carga_grupal → btn_guardar_carga"

Constraints:
- Single string replacement - keep all functionality intact
- Do NOT modify other components or main_app.py
- Ensure state session keys align with new button name
T3C
        if [ $? -eq 0 ]; then
            log_success "T3c: tab_ingreso.py button key fixed"
            save_state 3 "T3c" "completed"
        else
            log_error "T3c: FAILED"
            save_state 3 "T3c" "failed"
        fi
    ) &
    P_T3C=$!

    # Wait for all parallel tasks
    log_info "Waiting for swarm tasks to complete..."
    wait $P_T3A $P_T3B $P_T3C
    SWARM_EXIT=$?

    if [ $SWARM_EXIT -eq 0 ]; then
        log_success "All Phase 3 swarm tasks completed"
    else
        log_error "One or more Phase 3 tasks failed"
        return 1
    fi

    # Regression check before validation gates
    log_info "Running regression guard: test_chart_logic.py + test_dataframe_utils.py"
    if ! run_cmd "cd '${REPO_PATH}' && python -m pytest tests/test_chart_logic.py tests/test_dataframe_utils.py -v"; then
        log_error "Regression tests failed - Phase 3 compromised"
        return 1
    fi

    # Validation gates
    log_section "PHASE 3: VALIDATION GATES"
    
    local gate_passed=true
    if ! validate_gate_t4; then gate_passed=false; fi
    if ! validate_gate_t6; then gate_passed=false; fi
    
    if [ "${gate_passed}" = false ]; then
        log_error "Phase 3 validation gates FAILED"
        return 1
    fi

    local phase_duration=$(($(date +%s) - PHASE_START_TIME))
    log_success "PHASE 3 COMPLETE: ${phase_duration}s elapsed, ${TASKS_COMPLETED} tasks done"
    save_state 3 "complete" "success"
}

# ============================================================================
# PHASE 4: APP.PY CREATION (sequential, high complexity)
# ============================================================================

phase_4_create_app() {
    log_section "PHASE 4: CREATE APP.PY WITH FULL FEATURE SET"
    PHASE_START_TIME=$(date +%s)
    save_state 4 "orchestration_start" "in_progress"

    # Plan mode for complex app.py structure
    log_info "Using Plan Mode to design app.py structure..."
    gemini /plan <<'PLAN_PHASE4'
# app.py Architecture & Implementation Strategy

## Imports & Module Loading (with risk mitigation)
- From stable modules: carga_bruta_sesion (logic.biomechanics), conjunto_dominante_ci (fuzzy.diving_rules), fig_membership_fuzzy (visualization.charts)
- Re-export: db.MAX_IMPORT_ROWS (line-level validation in test_audit_fixes.py)
- Conditional decorator for @st.cache_data (test environment fallback)

## Function: calcular_membresias_atleta()
- Input: atleta_id (str), df_sesiones (DataFrame with snake_case columns)
- Process: Filter df for atleta_id, calculate membership via fuzzy engine
- Output: dict with membership scores and dominant set
- Caching: @st.cache_data(ttl=30) with fallback wrapper

## Function: calcular_historial_batch_cached()
- Input: atletas (tuple of str), fecha_inicio (date), fecha_fin (date)
- Process: For each atleta, compute monthly trend + fatigue history
- Output: List[dict] with historical metrics per athlete
- Caching: @st.cache_data(ttl=30) with hashable args
- Risk mitigation: tuple() conversion at all call sites

## UI: Buttons & State Management
- btn_guardar_well: Clear cache, rerun after wellness insert
- btn_guardar_carga: Clear cache, rerun after load insert
- Both must call: st.cache_data.clear() → logic → st.rerun()

## RBAC Guard: Within 6 Lines of "Funciones de Pertenencia" Expander
```
if rol_usuario in ["analitico", "admin"]:  # ← test_audit_fixes looks for this
    with st.expander("Funciones de Pertenencia"):
        # fuzzy membership rendering
```

## Validation Checklist (test_audit_fixes.py requirements)
- ✓ Imports appear as top-level ImportFrom (not inside functions)
- ✓ carga_bruta_sesion imported from logic.biomechanics
- ✓ conjunto_dominante_ci imported from fuzzy.diving_rules
- ✓ fig_membership_fuzzy imported from visualization.charts
- ✓ db.MAX_IMPORT_ROWS re-exported and used (not hardcoded 500)
- ✓ RBAC guard present within 6 lines before expander
- ✓ No matplotlib imports
- ✓ calcular_membresias_atleta defined
- ✓ calcular_historial_batch_cached defined with tuple arg

## Implementation Order
1. Imports & module setup (with fallback decorators)
2. Helper functions (RBAC check, cache wrapper)
3. Core functions (calcular_*)
4. Button callbacks
5. Main layout (tabs, expanders, RBAC gate)
6. Test: pytest tests/test_audit_fixes.py -v

PLAN_PHASE4

    # Approve plan and execute with single focused agent
    log_info "Creating app.py with verified architecture..."
    gemini /subagent-driven-development <<'T4_EXECUTE'
/agents enable receiving-code-review
@REPO_PATH

Task: Implement app.py per approved plan
1. Create app.py at repository root
2. Imports section (top-level, no conditionals):
   - from logic.biomechanics import carga_bruta_sesion
   - from fuzzy.diving_rules import conjunto_dominante_ci
   - from visualization.charts import fig_membership_fuzzy
   - import db (for db.MAX_IMPORT_ROWS)
   - Conditional @st.cache_data: try/except wrapper for non-Streamlit context

3. Helper function: _cache_data_ttl(func, ttl=30)
   - Wraps functions for caching in both test and Streamlit contexts

4. Function: calcular_membresias_atleta(atleta_id: str, df_sesiones: pd.DataFrame)
   - Filter by atleta_id
   - Call fuzzy engine on latest session
   - Return dict with membership scores

5. Function: calcular_historial_batch_cached(atletas: tuple, fecha_inicio, fecha_fin)
   - @st.cache_data(ttl=30) decorator with fallback
   - For each atleta in tuple: compute metrics + fatigue trend
   - Return List[dict] with monthly summaries

6. Callbacks: btn_guardar_well(), btn_guardar_carga()
   - st.cache_data.clear()
   - Insert to Supabase via db module
   - st.rerun()

7. Main layout with RBAC guard:
   ```python
   if rol_usuario in ["analitico", "admin"]:  # ← MUST be within 6 lines before expander
       with st.expander("Funciones de Pertenencia"):
           # render fig_membership_fuzzy
   ```

8. Validation tests before commit:
   - pytest tests/test_audit_fixes.py::TestAppStructure -v
   - pytest tests/test_security_hardening.py -v
   - python -c 'import app; assert hasattr(app, "calcular_membresias_atleta")'

9. Atomic commit: git add app.py && git commit -m "T4: Create app.py with RBAC, caching, and fuzzy integration"

Risk Mitigations:
- @st.cache_data inside try/except (pytest context doesn't have active session)
- All atletas params converted to tuple() at call sites
- RBAC check exactly matches test_audit_fixes.py AST scan pattern
- No matplotlib imports (would fail if Streamlit not in session)
- db.MAX_IMPORT_ROWS referenced exactly as 'db.MAX_IMPORT_ROWS' (not hardcoded 500)

T4_EXECUTE

    if [ $? -eq 0 ]; then
        log_success "app.py created and validated"
        save_state 4 "app_created" "success"
    else
        log_error "app.py creation failed"
        save_state 4 "app_created" "failed"
        return 1
    fi

    # Atomic verification gates
    log_section "PHASE 4: VALIDATION GATES"
    
    if ! run_cmd "cd '${REPO_PATH}' && python -m pytest tests/test_audit_fixes.py::TestAppStructure -v"; then
        log_error "test_audit_fixes.py::TestAppStructure FAILED"
        return 1
    fi
    log_success "TestAppStructure validation PASSED"

    if ! validate_gate_final; then
        log_error "Full test suite validation FAILED"
        return 1
    fi

    local phase_duration=$(($(date +%s) - PHASE_START_TIME))
    log_success "PHASE 4 COMPLETE: ${phase_duration}s elapsed"
    save_state 4 "complete" "success"
}

# ============================================================================
# PHASE 5: FINAL VALIDATION & TAGGING
# ============================================================================

phase_5_finalize() {
    log_section "PHASE 5: FINAL VALIDATION & STABLE TAG"
    PHASE_START_TIME=$(date +%s)
    save_state 5 "orchestration_start" "in_progress"

    log_info "Running comprehensive test suite validation..."
    
    if ! validate_gate_final; then
        log_error "Full test suite validation FAILED - cannot proceed to tagging"
        return 1
    fi

    # Detailed test report
    log_section "TEST RESULTS SUMMARY"
    log_info "Generating detailed test report..."
    run_cmd "cd '${REPO_PATH}' && python -m pytest tests/ -v --tb=line | tee '${LOG_DIR}/test-report-final.txt'"

    # Count passing tests
    local pass_count=$(grep -c "PASSED" "${LOG_DIR}/test-report-final.txt" || echo "0")
    local fail_count=$(grep -c "FAILED" "${LOG_DIR}/test-report-final.txt" || echo "0")
    
    log_info "Results: ${pass_count} PASSED, ${fail_count} FAILED"
    
    if [ "${fail_count}" -gt 0 ]; then
        log_error "Cannot tag - ${fail_count} tests still failing"
        return 1
    fi

    if [ "${pass_count}" -lt 60 ]; then
        log_warn "Only ${pass_count} tests passed (expected ≥60)"
        return 1
    fi

    log_success "All validation gates PASSED (${pass_count} tests)"

    # Create stable tag
    log_info "Creating v4.4-stable tag..."
    run_cmd "cd '${REPO_PATH}' && git tag -a v4.4-stable -m 'Phase 3-5 completion: test suite stabilized, 60+ tests passing, app.py integrated'"
    
    if [ $? -eq 0 ]; then
        log_success "Tag v4.4-stable created"
    else
        log_error "Failed to create tag"
        return 1
    fi

    # Push changes (optional, can be done manually)
    log_info "To push to remote, run: cd '${REPO_PATH}' && git push origin v4.4-stable"

    local phase_duration=$(($(date +%s) - PHASE_START_TIME))
    log_success "PHASE 5 COMPLETE: ${phase_duration}s elapsed"
    save_state 5 "complete" "success"
}

# ============================================================================
# EXECUTION ORCHESTRATION
# ============================================================================

main() {
    log_section "APP-DIVING PHASES 3-5 ORCHESTRATION"
    log_info "Repository: ${REPO_PATH}"
    log_info "Dry run: ${DRY_RUN}"
    log_info "Verbose: ${VERBOSE}"
    log_info "Logs: ${LOG_DIR}"

    # Check repository exists
    if [ ! -d "${REPO_PATH}/.git" ]; then
        log_error "Not a git repository: ${REPO_PATH}"
        exit 1
    fi

    # Check Gemini CLI is available
    if ! command -v gemini &> /dev/null; then
        log_error "Gemini CLI not found. Install via: npm install -g @anthropic-ai/gemini-cli"
        exit 1
    fi

    log_info "Gemini CLI version: $(gemini /about 2>/dev/null | head -1)"

    # Initialize execution state
    echo "[]" > "${STATE_FILE}"

    # Execute phases sequentially with error handling
    if ! phase_3_orchestrate; then
        log_error "Phase 3 failed - aborting"
        exit 1
    fi

    if ! phase_4_create_app; then
        log_error "Phase 4 failed - aborting"
        exit 1
    fi

    if ! phase_5_finalize; then
        log_error "Phase 5 failed - review logs"
        exit 1
    fi

    # Summary
    log_section "EXECUTION COMPLETE"
    log_info "Tasks completed: ${TASKS_COMPLETED}"
    log_info "Tasks failed: ${TASKS_FAILED}"
    log_info "Full logs: ${LOG_DIR}/execution.log"
    log_info "Test report: ${LOG_DIR}/test-report-final.txt"
    
    if [ "${TASKS_FAILED}" -eq 0 ]; then
        log_success "ALL PHASES 3-5 SUCCESSFULLY COMPLETED"
        exit 0
    else
        log_error "${TASKS_FAILED} tasks failed during execution"
        exit 1
    fi
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --repo) REPO_PATH="$2"; shift 2 ;;
        --dry-run) DRY_RUN=true; shift ;;
        --verbose) VERBOSE=true; shift ;;
        --help)
            echo "Usage: $0 [--repo PATH] [--dry-run] [--verbose]"
            exit 0
            ;;
        *) REPO_PATH="$1"; shift ;;
    esac
done

main "$@"
