"""
create_encoder.py [--datafile <data_file.csv>] [--mapping <activity_mapping_file.csv>]
                  [--unwanted <unwanted_activities.csv] --encoderfile <encoder_file.pkl>

Create label encoder from either <data_file.csv> and/or <activity_mapping_file.csv> and/or
<unwanted_activities.csv>, and write to <encoder_file.pkl> in PKL format.
"""

import argparse
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import joblib

from utilities import create_encoder_from_data, create_encoder_from_mapping
from utilities import load_activity_mapping, load_unwanted_activities, map_and_filter_data

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--encoderfile', dest='encoder_file', type=str, required=True)
    parser.add_argument('--datafile', dest='data_file', type=str)
    parser.add_argument('--mapping', dest='activity_mapping_file', type=str)
    parser.add_argument('--unwanted', dest='unwanted_activities_file', type=str)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
   args = parse_arguments()
   if args.data_file:
      df = pd.read_csv(args.data_dir)
   activity_mapping = None
   if args.activity_mapping_file:
      activity_mapping = load_activity_mapping(args.activity_mapping_file)
   unwanted_activities = None
   if args.unwanted_activities_file:
      unwanted_activities = load_unwanted_activities(args.unwanted_activities_file)
   if args.data_file:
      df = map_and_filter_data(df, activity_mapping, unwanted_activities)
      label_encoder = create_encoder_from_data(df)
   else:
      label_encoder = create_encoder_from_mapping(activity_mapping, unwanted_activities)
   joblib.dump(label_encoder, args.encoder_file)
