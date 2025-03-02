# generate-dataset.py --inputfile <input_file> [--testsize 0.2] [--windowsize 300] [--seed N]
#
# The input file is expected to be in the format that is output by the above generate-features.py
# script, using the --timewindow option on that script; namely, each line begins with a stamp_begin
# and stamp_end timestamp, followed by the feature values, followed by the activity_label (str or None).
# The script generates two files: <input_file>-train.csv and <input_file>-test.csv, in the same format
# as the input, except only one timestamp 'stamp' is given, which is the end of the window. The test
# file contains labeled examples that cover approximately `--testsize` (default=0.2) of the time span
# of the input file, and the train file contains the rest of the examples that do not overlap with the
# test examples. The script selects examples for the test set by randomly selecting an unused example
# and collecting all the nearby unused examples within a time window of size `--windowsize` seconds
# (default=300). The random selection is controlled by the `--seed` if given; otherwise, the seed is
# randomly chosen.

import pandas as pd
import numpy as np # Use NumPy random generator
import argparse
import os

def read_csv_file(file_path):
    """
    Reads a CSV file into a Pandas DataFrame. Assumes each line starts with a
    stamp_start and stamp_end timestamps, followed by some number of float features,
    followed by an activity_label string (or None).
    """
    # Read the CSV file and remove rows where activity_label is None
    df = pd.read_csv(file_path, low_memory=False)
    df['stamp_start'] = pd.to_datetime(df['stamp_start'], format='mixed')
    df['stamp_end'] = pd.to_datetime(df['stamp_end'], format='mixed')
    df = df.dropna(subset=['activity_label'])
    # Set index as the interval from start_time to end_time
    df['time_range'] = pd.IntervalIndex.from_arrays(df['stamp_start'], df['stamp_end'], closed='both')
    df.set_index('time_range', inplace=True)
    return df

def generate_train_test(df, test_size, window_size):
    """
    Generate train and test sets from examples in df, where the test set is
    approximately test_size of the time spanned by df, and examples are randomly
    selected from a window_size amount of time for each selection.
    """
    train_df = pd.DataFrame
    test_df = pd.DataFrame
    df['used'] = False
    merged_intervals = df.index.union(df.index)
    total_seconds = sum((interval.right - interval.left).total_seconds() for interval in merged_intervals)
    test_seconds_target = test_size * total_seconds
    test_seconds = 0
    # Collect test set
    while (test_seconds < test_seconds_target):
        # Add examples to test set according to randomly selected window
        unused_rows_df = df[df['used'] == False]
        if not unused_rows_df.empty:
            random_row_interval_index = unused_rows_df.sample(n=1).index[0]  # Randomly pick one row
            nearby_rows = find_nearby_rows(df, random_row_interval_index, window_size)
            df.loc[nearby_rows.index, 'used'] = True
            # Recompute seconds covered by test set
            used_df = df[df['used'] == True]
            used_intervals = used_df.index.union(used_df.index)
            test_seconds = sum((interval.right - interval.left).total_seconds() for interval in used_intervals)
        else:
            print(f'Unable to generate {test_seconds} seconds of examples for Test set.')
            break
    test_df = df[df['used'] == True]
    # Collect train set
    used_intervals = test_df.index
    for used_interval in used_intervals:
        overlapping_rows = df.index.overlaps(used_interval)  # Boolean mask
        df.loc[overlapping_rows, 'used'] = True  # Set 'used' to True for overlapping rows
    train_df = df[df['used'] == False]
    # Compute seconds covered by train set
    train_intervals = train_df.index.union(train_df.index)
    train_seconds = sum((interval.right - interval.left).total_seconds() for interval in train_intervals)
    if (not train_df.empty):
        print(f'Generated Train set with {len(train_df)} examples covering {round(train_seconds)} seconds')
        train_df = train_df.drop(columns=['stamp_start', 'used'])
        train_df = train_df.rename(columns={'stamp_end': 'stamp'})
    else:
        print('Unable to generate Train set.')
    if (not test_df.empty):
        print(f'Generated Test set with {len(test_df)} examples covering {round(test_seconds)} seconds')
        test_df = test_df.drop(columns=['stamp_start', 'used'])
        test_df = test_df.rename(columns={'stamp_end': 'stamp'})
    else:
        print('Unable to generate Test set.')
    return train_df, test_df

def find_nearby_rows(df, row_interval_index, window_size):
    """Return dataframe composed of unused rows near given row such that the dataframe spans
    no more than window_size seconds and encompasses no previously used rows."""
    window_size = pd.Timedelta(seconds=window_size)
    # Initialize search window
    total_window_start = row_interval_index.left
    total_window_end = row_interval_index.right
    # Sort intervals for efficient forward/backward search
    sorted_intervals = df.index.sort_values()
    # Get index of the randomly selected interval
    idx = np.where(sorted_intervals == row_interval_index)[0][0]
    # Search forward in time
    for i in range(idx + 1, len(sorted_intervals)):
        if df.loc[sorted_intervals[i], 'used']:  # Stop if row is used
            break
        proposed_end = sorted_intervals[i].right
        if (proposed_end - total_window_start) > window_size:  # Stop if exceeding window_size
            break
        total_window_end = proposed_end  # Expand window
    # Search backward in time if window_size is not exhausted
    for i in range(idx - 1, -1, -1):
        if df.loc[sorted_intervals[i], 'used']:  # Stop if row is used
            break
        proposed_start = sorted_intervals[i].left
        if (total_window_end - proposed_start) > window_size:  # Stop if exceeding window_size
            break
        total_window_start = proposed_start  # Expand window
    # Find all intervals within the expanded search window and unused
    search_window = pd.Interval(total_window_start, total_window_end, closed="both")
    matching_rows = df[(df.index.overlaps(search_window)) & (df["used"] == False)]
    return matching_rows

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputfile', dest='input_file', type=str, required=True)
    parser.add_argument('--testsize', dest='test_size', type=float, default=0.2)
    parser.add_argument('--windowsize', dest='window_size', type=int, default=300)
    parser.add_argument('--seed', dest='seed', type=int)
    args = parser.parse_args()
    return args  

if __name__ == "__main__":
    args = parse_arguments()
    # Read input data
    df = read_csv_file(args.input_file)
    # Set random seed if given (for repeatability)
    if args.seed is not None:
        np.random.seed(args.seed)
    # Generate train and test sets
    train_df, test_df = generate_train_test(df, args.test_size, args.window_size)
    # Write train and test sets to CSV files
    if (not train_df.empty) and (not test_df.empty):
        base_file = os.path.splitext(args.input_file)[0]
        train_file = base_file + '-train.csv'
        test_file = base_file + '-test.csv'
        train_df.to_csv(train_file, index=False, date_format='%Y-%m-%d %H:%M:%S.%f')
        test_df.to_csv(test_file, index=False, date_format='%Y-%m-%d %H:%M:%S.%f')
    else:
        print('Unable to generate dataset.')
