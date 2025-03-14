"""
   label_encoder.py <activity_mapping.csv>

   Create label encoder for activities mentioned in given activity map.
   Save into file 'label_encoder.pkl' for input to all programs.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import joblib
import sys

# Load activity mapping from CSV
def load_activity_mapping(file_path="activity_mapping.csv"):
    return pd.read_csv(file_path).set_index("original")["mapped"].to_dict()

act_map_file = sys.argv[1]
activity_mapping = load_activity_mapping(act_map_file)
label_encoder = LabelEncoder()
activities = np.unique(list(activity_mapping.values()))
label_encoder.fit(activities)
joblib.dump(label_encoder, "label_encoder.pkl")
