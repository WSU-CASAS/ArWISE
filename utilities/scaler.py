"""
scaler.py <data.csv>

   Creates standard scaler and trains on given dataset. The standard scalar scales
   each column independently by removing the mean and scaling to unit variance.
   Saves scaler to 'scaler.pkl' for input to all programs.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import sys

df = pd.read_csv(sys.argv[1])
df = df.drop(columns=["stamp", "stamp_start", "stamp_end", "activity_label"], errors='ignore')
df = df.astype(np.float32).fillna(0.0)
scaler = StandardScaler()
scaler.fit(df)
joblib.dump(scaler, "scaler.pkl")
