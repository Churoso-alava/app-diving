import pandas as pd
import numpy as np
from ui.charts_redesign import _apply_savgol

def test_apply_savgol_handles_nan_inf():
    # Setup dataframe with problematic data
    df = pd.DataFrame({"test_col": [1.0, np.nan, 3.0, float('inf'), 5.0]})
    
    # This should not raise a ValueError and should handle the NaN/inf
    result = _apply_savgol(df, "test_col")
    
    # Assertions
    assert not result.isna().any(), "Result contains NaNs"
    assert not result.isin([float('inf'), float('-inf')]).any(), "Result contains infs"
    assert len(result) == 5, "Result length mismatch"
