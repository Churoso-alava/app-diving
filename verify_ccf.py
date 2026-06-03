import numpy as np
from statsmodels.tsa.stattools import ccf as _ccf

x = np.random.normal(0, 1, 50)
y = np.random.normal(0, 1, 50)
max_lag = 7

try:
    raw = _ccf(x, y, nlags=max_lag, adjusted=True)
    print(f"Length of raw with nlags={max_lag}: {len(raw)}")
    for lag in range(max_lag + 1):
        _ = raw[lag]
    print("Success with range(max_lag + 1)")
except Exception as e:
    print(f"Error with nlags={max_lag}: {e}")

try:
    raw = _ccf(x, y, nlags=max_lag + 1, adjusted=True)
    print(f"Length of raw with nlags={max_lag+1}: {len(raw)}")
    for lag in range(max_lag + 1):
        _ = raw[lag]
    print("Success with range(max_lag + 1) and nlags=max_lag+1")
except Exception as e:
    print(f"Error with nlags={max_lag+1}: {e}")
