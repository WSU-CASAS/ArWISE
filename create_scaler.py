"""
create_scaler.py --datafile <data_file.csv> --scalerfile <scaler_file.pkl>

Create scaler from <data_file.csv> and write to <scaler_file.pkl> in PKL format.
"""

import argparse
import pandas as pd
import joblib

from utilities import create_scaler_from_data

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scalerfile', dest='scaler_file', type=str, required=True)
    parser.add_argument('--datafile', dest='data_file', type=str, required=True)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
   args = parse_arguments()
   df = pd.read_csv(args.data_file)
   scaler = create_scaler_from_data(df)
   joblib.dump(scaler, args.scaler_file)
