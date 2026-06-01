"""
Refactored fuzzy variable definitions.
Ensures consistency with core/fuzzy_engine.py in naming, typing, and structure.
"""
import logging
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

log = logging.getLogger(__name__)

# =============================================================================
#  DEFINITION OF FUZZY VARIABLES (Aligned with core/fuzzy_engine.py style)
# =============================================================================

# Note: The original fuzzy_engine.py defines the primary fatigue calculation variables.
# This file, if kept, should define complementary or alternative sets, or be integrated.
# For alignment, we'll standardize naming and ensure clarity.

# Original fuzzy_variables.py defined: wellness, carga_integrada, fatiga.
# fuzzy_engine.py defines: acwr, delta_pct, z_meso, beta_aguda, beta_28, fatiga.

# The 'fatiga' variable definition here conflicts with fuzzy_engine.py.
# For this refactoring focused on style alignment, we'll keep the definition
# but log a warning about the conflict. True reconciliation would require
# deciding on a single definition or renaming.

def get_wellness_variables():
    """
    Defines fuzzy variables related to wellness.
    Aligned with core/fuzzy_engine.py style (snake_case, clear universes).
    Original used 1-7 for wellness score.
    """
    u_wellness = np.arange(1, 7.1, 1) # Universe for wellness score (1 to 7)
    wellness_var = ctrl.Antecedent(u_wellness, 'wellness_score') # Standardized snake_case

    # Membership functions for wellness, using trimf for simplicity as in original
    wellness_var['pésimo']    = fuzz.trimf(u_wellness, [1, 1, 3])
    wellness_var['regular']   = fuzz.trimf(u_wellness, [2, 4, 5])
    wellness_var['excelente'] = fuzz.trimf(u_wellness, [4, 7, 7])

    log.debug("Fuzzy variable 'wellness_score' defined.")
    return wellness_var

def get_carga_integrada_variables():
    """
    Defines fuzzy variables for integrated load.
    Aligned with core/fuzzy_engine.py style (snake_case, clear universes).
    """
    u_carga = np.arange(0, 201, 1)
    carga_integrada_var = ctrl.Antecedent(u_carga, 'integrated_load') # Standardized snake_case

    carga_integrada_var['baja']  = fuzz.trimf(u_carga, [0, 0, 100])
    carga_integrada_var['media'] = fuzz.trimf(u_carga, [50, 100, 150])
    carga_integrada_var['alta']  = fuzz.trimf(u_carga, [100, 200, 200])

    log.debug("Fuzzy variable 'integrated_load' defined.")
    return carga_integrada_var

def get_conflicting_fatiga_variable():
    """
    Defines a 'fatiga' consequent variable.
    NOTE: This definition conflicts with 'fatiga' in core/fuzzy_engine.py.
    Alignment requires reconciliation or renaming of this variable.
    """
    u_fat = np.arange(0, 101, 1)
    fatiga_var = ctrl.Consequent(u_fat, 'fatiga') # Potential conflict with fuzzy_engine.py

    fatiga_var['baja'] = fuzz.trimf(u_fat, [0, 0, 50])
    fatiga_var['media'] = fuzz.trimf(u_fat, [25, 50, 75])
    fatiga_var['alta'] = fuzz.trimf(u_fat, [50, 100, 100])
    
    log.warning("Fuzzy variable 'fatiga' defined here, but conflicts with core/fuzzy_engine.py. Reconciliation needed.")
    return fatiga_var

# The original get_variables() function is adapted to call the new functions,
# ensuring consistent style and adding logging.
def get_variables():
    """
    Returns the fuzzy variables defined in this module, aligned in style.
    NOTE: Reconciliation of overlapping/conflicting definitions (like 'fatiga')
    with core/fuzzy_engine.py is pending further architectural decisions.
    """
    wellness_var = get_wellness_variables()
    carga_integrada_var = get_carga_integrada_variables()
    fatiga_var = get_conflicting_fatiga_variable()
    
    log.info("Fuzzy variables 'wellness_score', 'integrated_load', 'fatiga' (conflicting) loaded from fuzzy_variables.py.")
    return wellness_var, carga_integrada_var, fatiga_var
