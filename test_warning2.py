import pandas as pd
import warnings
import numpy as np

warnings.simplefilter('always', DeprecationWarning)

try:
    np.timedelta64(1)
except Exception as e:
    print("error1", e)

try:
    pd.Timedelta(days=2)
except Exception as e:
    print("error2", e)

try:
    pd.to_timedelta(2, unit="d")
except Exception as e:
    print("error3", e)

print("done")
