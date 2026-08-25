import pandas as pd
import warnings
warnings.simplefilter('always', DeprecationWarning)

print(pd.__version__)
try:
    import numpy as np
    print(np.__version__)
except ImportError:
    pass

with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    pd.Timedelta(days=2)
    print("pd.Timedelta(days=2)", len(w))

with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    t = pd.Timestamp("2020-01-01")
    t - pd.Timedelta(days=2)
    print("t - pd.Timedelta(days=2)", len(w))

with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    pd.Timedelta(2, "D")
    print("pd.Timedelta(2, 'D')", len(w))
