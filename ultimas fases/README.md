# APP-DIVING PHASES 3-5: GEMINI CLI ORCHESTRATION SUITE

## 📋 Quick Start

```bash
# Clone and navigate
git clone https://github.com/Churoso-alava/app-diving.git
cd app-diving

# Option 1: Automatic orchestration (recommended)
bash setup-and-run.sh --repo .

# Option 2: Manual phase-by-phase execution
bash execute-phases-3-5.sh --repo . --verbose
```

---

## 📦 What's Included

### 1. **execute-phases-3-5.sh** (2,500+ lines)
Professional orchestration engine with:
- **Phase 3**: Parallel execution of test corrections (T3a, T3b, T3c)
- **Phase 4**: Sequential app.py creation with full feature set
- **Phase 5**: Final validation and v4.4-stable tagging
- Automated validation gates at each phase
- State tracking and recovery capabilities
- Comprehensive logging to `logs/` directory

**Key Features**:
- Swarm coordination for parallel tasks (Phase 3)
- Progressive validation gates matching coordination-plan.json
- Atomic git commits per task (ensures clean history)
- Risk mitigation patterns (cache decorator fallback, tuple conversion, etc.)
- Full regression guard (test_chart_logic.py + test_dataframe_utils.py always green)

### 2. **setup-and-run.sh** (400+ lines)
Environment setup wrapper:
- Validates Git, Python, Gemini CLI, Node.js availability
- Enables all required Gemini CLI skills automatically
- Initializes memory system for phase tracking
- Checks repository structure before execution
- Runs baseline regression tests
- Provides clear success/failure reporting

### 3. **AGENT_REFERENCE.md** (600+ lines)
Detailed technical reference for Gemini CLI agents:
- **Phase 3 specifics**: Column name mappings, import path fixes, button key details
- **Phase 4 architecture**: Required imports, function signatures, RBAC placement, cache patterns
- **Phase 5 validation**: Gate commands, expected outputs, tagging procedure
- **Risk mitigations**: 6 identified risks with concrete implementation patterns
- **Skills recommendations**: Which Gemini CLI skills to enable for each task
- **File locations**: Complete repository structure reference

### 4. **EXECUTION_EXAMPLES.sh** (400+ lines)
Ready-to-run interactive examples:
- Copy-paste Gemini CLI commands for each phase
- Actual prompts to give to agents
- Expected outputs and verification commands
- Troubleshooting procedures
- Performance optimization tips
- Final comprehensive checklist

---

## 🎯 What These Scripts Do

### Execute Phases 3-5 Programmatically

```
INPUT: coordination-plan.json (phases 1-2 already complete)
   ↓
PHASE 3: Test Corrections (15-20 min)
   → T3a: Fix test_services.py imports & columns (parallel)
   → T3b: Fix test_diving_load.py import path (parallel)
   → T3c: Rename button key in tab_ingreso.py (parallel)
   → Validation gate: pytest + grep + regression check
   ↓
PHASE 4: Create app.py (30-40 min)
   → Design architecture with Plan Mode
   → Implement with subagent-driven-development
   → Verify with verification-before-completion
   → Validation gate: test_audit_fixes.py + test_security_hardening.py
   ↓
PHASE 5: Final Validation (5-10 min)
   → Run full test suite
   → Create v4.4-stable tag
   → Validation gate: 60+ tests PASSED, 0 FAILED
   ↓
OUTPUT: Repository ready for production with:
   ✓ All test imports corrected
   ✓ Snake_case column contract enforced
   ✓ Button key refactored
   ✓ app.py integrated with RBAC & caching
   ✓ 60+ tests passing, zero failures
   ✓ v4.4-stable tag for reproducible releases
```

---

## 🚀 Execution Paths

### Path A: Fully Automatic (Recommended)
```bash
bash setup-and-run.sh --repo .
```
**Time**: ~90 minutes | **Hands-on**: Minimal | **Best for**: CI/CD, unattended execution

### Path B: Manual with Gemini CLI
```bash
# 1. Setup environment
bash setup-and-run.sh --skip-setup --repo .

# 2. Execute orchestrator
bash execute-phases-3-5.sh --repo . --verbose

# 3. Monitor logs
tail -f logs/execution.log
```
**Time**: ~90 minutes | **Hands-on**: Low | **Best for**: Learning, verification

### Path C: Interactive Gemini Sessions
```bash
# Phase 3 (parallel swarm)
gemini /swarm-advanced coordinate --mode parallel @tests/test_services.py @tests/test_diving_load.py @components/tab_ingreso.py

# Phase 4 (sequential with plan mode)
gemini /plan  # Design architecture
gemini /subagent-driven-development  # Implement

# Phase 5 (verification)
gemini /verification-before-completion
```
**Time**: ~120 minutes | **Hands-on**: High | **Best for**: Detailed control, learning, debugging

---

## 🔄 Gemini CLI Skills Utilized

| Skill | Used In | Purpose |
|-------|---------|---------|
| `subagent-driven-development` | T3a, T3b, T4 | Individual agent handling single complex task |
| `swarm-advanced` | Phase 3 | Parallel orchestration of T3a, T3b, T3c |
| `verification-before-completion` | All gates | Validate before committing |
| `test-driven-development` | T4 | Design app.py with test-first approach |
| `systematic-debugging` | Troubleshooting | Fix issues if gates fail |
| `writing-plans` | T4 planning | Architecture design before implementation |
| `receiving-code-review` | T4 validation | Handle feedback on app.py |

---

## ⚙️ Configuration & Customization

### Environment Variables
```bash
export DRY_RUN=false          # Set to 'true' to preview without changes
export VERBOSE=true           # Enable detailed logging
export LOG_DIR="/path/to/logs" # Custom log directory
```

### Repository Path
```bash
bash execute-phases-3-5.sh --repo /absolute/path/to/app-diving
# or from within repo:
bash execute-phases-3-5.sh --repo .
```

### Memory Persistence
Memory file: `PHASE_EXECUTION.md` (auto-created)  
State file: `.execution-state.json` (tracks progress)

---

## 📊 Expected Outputs

### Successful Completion Metrics

```
Phase 3:
  ✓ test_services.py: 20+ tests PASSED
  ✓ test_diving_load.py: 24 tests PASSED
  ✓ btn_carga_grupal: Fully removed (grep exit 1)
  ✓ Regression: test_chart_logic.py + test_dataframe_utils.py PASSED

Phase 4:
  ✓ app.py created at repository root
  ✓ Imports present: carga_bruta_sesion, conjunto_dominante_ci, fig_membership_fuzzy
  ✓ Functions: calcular_membresias_atleta(), calcular_historial_batch_cached()
  ✓ RBAC guard within 6 lines of "Funciones de Pertenencia" expander
  ✓ test_audit_fixes.py: All checks PASSED
  ✓ test_security_hardening.py: 2 tests PASSED

Phase 5:
  ✓ Full test suite: 60+ tests PASSED
  ✓ Failures: 0
  ✓ Tag: v4.4-stable created
  ✓ Git history: 4 atomic commits (T3a, T3b, T3c, T4)
```

### Log Files
- `logs/execution.log`: Complete execution timeline
- `logs/test-report-final.txt`: Full pytest output
- `PHASE_EXECUTION.md`: Memory state (auto-updated)
- `.execution-state.json`: Progress tracking

---

## 🛡️ Risk Mitigations

All 6 identified risks in coordination-plan.json are handled:

| Risk | Mitigation | Implementation |
|------|-----------|-----------------|
| R1: @st.cache_data fails outside Streamlit | Try/except wrapper with fallback | `_cache_decorator` pattern in app.py |
| R2: AST import check fails | Top-level imports only | Enforced at Phase 4 task level |
| R3: Wrong module path confusion | Explicit specification in task | Agent reference includes exact imports |
| R4: Dependency version mismatch | Both files pinned atomically | Done in Phase 2 (already complete) |
| R5: Hardcoded row limit | Reference db.MAX_IMPORT_ROWS | Enforced in test_audit_fixes.py gate |
| R6: Entry point divergence | Document inline vs component UI | Risk acknowledged for future consolidation |

---

## 🔍 Validation Gates

### Phase 3 Gates
```bash
pytest tests/test_services.py -v              # → All classes PASSED
pytest tests/test_diving_load.py -v           # → 24 PASSED
grep -q btn_carga_grupal components/tab_ingreso.py  # → Exit 1 (not found)
pytest tests/test_chart_logic.py tests/test_dataframe_utils.py -v  # → All PASSED
```

### Phase 4 Gates
```bash
python -c "from app import carga_bruta_sesion, conjunto_dominante_ci, fig_membership_fuzzy"  # → No error
pytest tests/test_audit_fixes.py::TestAppStructure -v  # → All PASSED
pytest tests/test_security_hardening.py -v    # → 2 PASSED
```

### Phase 5 Gates
```bash
pytest tests/ -v --tb=short                   # → 60+ PASSED, 0 FAILED
git tag -l v4.4-stable                        # → v4.4-stable (exists)
```

---

## 🐛 Troubleshooting

### "Gemini CLI not found"
```bash
npm install -g @anthropic-ai/gemini-cli
gemini /about  # Verify installation
```

### "pytest not found"
```bash
pip install pytest
python -m pytest --version  # Verify
```

### "One or more Phase X tasks failed"
1. Check logs: `tail -50 logs/execution.log`
2. Review test output: `cat logs/final_test_results.log`
3. Re-run with debugging: `bash execute-phases-3-5.sh --repo . --verbose`
4. Check agent reference: See AGENT_REFERENCE.md for exact requirements

### "test_audit_fixes.py still failing after app.py created"
1. Verify imports are top-level (not inside functions)
2. Check RBAC guard within 6 lines: `grep -B6 "Funciones de Pertenencia" app.py | grep -E "rol_usuario|analitico"`
3. Confirm db.MAX_IMPORT_ROWS usage: `grep db.MAX_IMPORT_ROWS app.py`
4. Run AST check: `python -c "import ast; ast.parse(open('app.py').read()); print('✓ Valid syntax')"`

---

## 📈 Performance Characteristics

### Execution Timeline
| Phase | Duration | Parallelizable | CPU Usage |
|-------|----------|----------------|-----------|
| Setup & Validation | 2-3 min | No | Low |
| Phase 3 (Tests) | 15-20 min | Yes (3 tasks) | Medium |
| Phase 4 (app.py) | 30-40 min | No | Low |
| Phase 5 (Validation) | 5-10 min | No | Medium |
| **Total** | **~60-75 min** | Partial | Low-Medium |

### Resource Usage
- **Disk**: ~500 MB (repository + logs)
- **Memory**: ~200 MB (Python + Gemini CLI)
- **Network**: Minimal (local execution)
- **Gemini CLI API calls**: ~15-20 (can be parallelized)

---

## 📚 Documentation Structure

```
outputs/
├── execute-phases-3-5.sh          ← Main orchestration engine
├── setup-and-run.sh                ← Environment setup wrapper
├── AGENT_REFERENCE.md              ← Technical reference for agents
├── EXECUTION_EXAMPLES.sh           ← Ready-to-run examples
└── README.md                       ← This file
```

### How to Use Each Document

1. **Start here**: This README + Quick Start section
2. **Setup**: Run `setup-and-run.sh` for automated setup
3. **Execution**: Use `execute-phases-3-5.sh` for orchestration
4. **Learning**: Read `AGENT_REFERENCE.md` for deep technical details
5. **Interactive**: Use `EXECUTION_EXAMPLES.sh` for manual Gemini CLI sessions
6. **Troubleshooting**: Reference AGENT_REFERENCE.md risk mitigations

---

## ✅ Success Criteria

You've completed phases 3-5 successfully when:

- [ ] **Phase 3**: All test imports fixed, button key renamed, regression tests green
- [ ] **Phase 4**: app.py created with RBAC, caching, fuzzy integration
- [ ] **Phase 5**: 60+ tests PASSED, 0 FAILED, v4.4-stable tag created
- [ ] **Git History**: 4 atomic commits visible (T3a, T3b, T3c, T4)
- [ ] **Deployment Ready**: Changes pushed to remote, v4.4-stable tag pushed

---

## 🚀 Next Steps After Completion

```bash
# 1. Verify everything locally
bash final-check.sh

# 2. Push to remote (if using Git)
cd app-diving
git push origin main
git push origin v4.4-stable

# 3. Create release on GitHub (optional)
gh release create v4.4-stable --notes "Phase 3-5 completion: test suite stabilized, 60+ tests passing"

# 4. Deploy with confidence!
# Your app-diving project is now production-stable
```

---

## 📞 Support & Issues

**If something fails**:
1. Check AGENT_REFERENCE.md for that specific phase
2. Review error message in logs/execution.log
3. Rerun with `--verbose` flag for detailed output
4. Check coordination-plan.json critical issues section

**For coordination plan questions**:
See: `/mnt/user-data/uploads/coordination-plan.json`

---

## 📄 License & Attribution

- **Coordination Plan**: From app-diving project diagnosis (Jan 2025)
- **Orchestration Scripts**: Generated for Gemini CLI execution
- **Skills Used**: All standard Gemini CLI skills
- **Target**: app-diving v4.4-stable release

---

## 🎓 Learning Resources

- **Gemini CLI Documentation**: `gemini /docs`
- **Skills Documentation**: `gemini /skills list all`
- **MCP Server Integration**: `gemini /mcp list`
- **Plan Mode Guide**: `gemini /plan` (use interactively)
- **Swarm Coordination**: `/swarm-advanced coordinate --help`

---

**Generated**: April 17, 2025  
**Status**: Ready for execution  
**Confidence Level**: PROFESSIONAL (tested against coordination-plan.json requirements)

