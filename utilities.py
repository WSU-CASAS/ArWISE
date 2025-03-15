"""
utilities.py

Various utilities used by other programs.
   
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib

def load_activity_mapping(file_path="activity_mapping.csv"):
    """Load a file containing mappings from the original activity labels to the activity categories being modeled."""
    return pd.read_csv(file_path).set_index("original")["mapped"].to_dict()

def load_unwanted_activities(file_path="unwanted_activities.csv"):
    """Load a file containing names of activity categories to ignore (drop)."""
    return set(pd.read_csv(file_path)["activity_label"])

def create_scaler_from_data(df, output_file=None):
   """
   Creates standard scaler and trains on given ArWISE dataframe dataset. The standard scalar scales
   each column independently by removing the mean and scaling to unit variance.
   If output_file given, saves scaler to file for input to all programs.
   """
   df = df.drop(columns=["stamp", "stamp_start", "stamp_end", "activity_label"], errors='ignore')
   df = df.astype(np.float32).fillna(0.0)
   scaler = StandardScaler()
   scaler.fit(df)
   if output_file:
      joblib.dump(scaler, output_file)
   return scaler

def create_encoder_from_data(df, output_file=None):
   """
   Creates label encoder for activities mentioned in given ArWISE dataframe dataset.
   If output_file given, save encoder to file for input to all programs.
   """
   label_encoder = LabelEncoder()
   label_encoder.fit(df['activity_label'])
   if output_file:
      joblib.dump(label_encoder, output_file)
   return label_encoder

def create_encoder_from_mapping(act_map_file, output_file=None):
   """
   Creates label encoder for activities mentioned in given activity map file.
   If output_file given, save encoder to file for input to all programs.
   """
   activity_mapping = pd.read_csv(act_map_file).set_index("original")["mapped"].to_dict()
   label_encoder = LabelEncoder()
   activities = np.unique(list(activity_mapping.values()))
   label_encoder.fit(activities)
   if output_file:
      joblib.dump(label_encoder, output_file)
   return label_encoder
