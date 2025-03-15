"""
   cnn.py --datadir <datadir> --modelfile <model_file>
          [--encoder <label_encoder.pkl>] [--scaler <scaler.pkl>]
          [--mapping <activity_mapping.csv>] [--unwanted <unwanted_activities.csv>]
          
   Train and test a 1D CNN human activity recognition model. The program reads in all CSV (*.csv)
   files from the given <data_dir> and saves the trained model in Keras format to <model_file>.

   If encoder given, then use to encode activity labels; otherwise, compute encoder from data.
   If scaler given, then use to scale data; otherwise, compute scaler from data.
   If unwanted activities given, then remove examples classified with these activities.
   If activity mapping given, then use to map activities in data.

   This model processes time series data points. Each row of the data files, after the header row,
   starts with a timestamp field and ends with an activity label field. The fields in between
   represent all of the features over the window of consecutive time points. E.g., if each row
   represents 8 features for each of 100 consecutive, then there will be 802 features in the row,
   including the timestamp and activity label. These values are provided as constants in the code
   below: NUM_FEATURES, WINDOW_SIZE.
"""

import numpy as np
import pandas as pd
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.saving import save_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import f1_score, matthews_corrcoef
import joblib
import argparse

from utilities import create_encoder_from_data, create_scaler_from_data
from utilities import load_activity_mapping, load_unwanted_activities

NUM_FEATURES = 8  # Number of features per time point
WINDOW_SIZE = 100 # Consecutive time points concatenated to form features
TEST_SIZE = 0.2   # Fractional of data used for testing
EPOCHS = 20       # Number of training epochs for CNN
BATCH_SIZE = 32   # Batch size for CNN training

# Load data from files in data dir
def load_data(data_dir):
    df = pd.DataFrame()
    data_files = sorted(os.listdir(data_dir))
    for file in data_files:
        file_path = os.path.join(data_dir, file)
        if file.endswith(".csv"):
            new_df = pd.read_csv(file_path)
            df = pd.concat([df, new_df], ignore_index=True)
    return df

def map_and_filter_data(df, activity_mapping, unwanted_activities):
    """Remove rows with no activity label, or a label in unwanted activities.
    Then, map activity labels according to the given mapping."""
    df = df.dropna(subset=["activity_label"])
    if unwanted_activities:
        df = df[~df["activity_label"].isin(unwanted_activities)]
    if activity_mapping:
        df["activity_label"] = df["activity_label"].replace(activity_mapping)
    return df

# Build train and test data.
def process_data(df, label_encoder, scaler):
    y = df["activity_label"].values
    y_encoded = label_encoder.transform(y)  # Transform activity labels to integers
    df = df.drop(columns=["stamp", "stamp_start", "stamp_end", "activity_label"], errors='ignore')
    df = df.astype(np.float32).fillna(0.0)
    df = df.fillna(0.0)
    X = scaler.fit_transform(df)
    try:
        X = X.reshape(-1, WINDOW_SIZE, NUM_FEATURES)
    except ValueError as e:
        print(f"Reshape error: {e}")
        print(f"X.shape before reshape: {X.shape}, num_timepoints: {WINDOW_SIZE}, num_features: {NUM_FEATURES}")
        exit(1)
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=TEST_SIZE)
    return X_train, X_test, y_train, y_test

# Define the 1D CNN model structure
def build_model(input_shape, num_classes):
    model = keras.Sequential([
        layers.Conv1D(filters=64, kernel_size=5, activation='relu', input_shape=input_shape, dtype=tf.float32),
        layers.BatchNormalization(),
        layers.Conv1D(filters=128, kernel_size=5, activation='relu', dtype=tf.float32),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=2, dtype=tf.float32),
        layers.Conv1D(filters=256, kernel_size=5, activation='relu', dtype=tf.float32),
        layers.BatchNormalization(dtype=tf.float32),
        layers.GlobalAveragePooling1D(dtype=tf.float32),
        layers.Dense(128, activation='relu', dtype=tf.float32),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax', dtype=tf.float32)
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Calculate top-k accuracy based on a vectors of ground truth and predicted activity values.
def top_k_accuracy(y_true, y_pred, k=3):
    metric = tf.keras.metrics.sparse_top_k_categorical_accuracy(y_true, y_pred, k=k)
    return float(tf.reduce_mean(metric).numpy())

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datadir', dest='data_dir', type=str, required=True)
    parser.add_argument('--modelfile', dest='model_file', type=str, required=True)
    parser.add_argument('--encoder', dest='label_encoder_file', type=str)
    parser.add_argument('--scaler', dest='scaler_file', type=str)
    parser.add_argument('--mapping', dest='activity_mapping_file', type=str)
    parser.add_argument('--unwanted', dest='unwanted_activities_file', type=str)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_arguments()
    # Get data
    data_df = load_data(args.data_dir)
    # Build label_encoder, activity mapping, and unwanted activities (if provided)
    label_encoder = None
    if args.label_encoder_file:
        label_encoder = joblib.load(args.label_encoder_file)
    scaler = None
    if args.scaler_file:
        scaler = joblib.load(args.scaler_file)
    activity_mapping = None
    if args.activity_mapping_file:
        activity_mapping = load_activity_mapping(args.activity_mapping_file)
    unwanted_activities = None
    if args.unwanted_activities_file:
        unwanted_activities = load_unwanted_activities(args.unwanted_activities_file)
    data_df = map_and_filter_data(data_df, activity_mapping, unwanted_activities)
    if not label_encoder:
        label_encoder = create_encoder_from_data(data_df)
    if not scaler:
        scaler = create_scaler_from_data(data_df)
    # Build train and test sets
    num_classes = len(label_encoder.classes_)
    X_train, X_test, y_train, y_test = process_data(data_df, label_encoder, scaler)
    # Build, train and save model
    input_shape = (X_train.shape[1], X_train.shape[2])
    model = build_model(input_shape, num_classes)
    model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(X_test, y_test))
    save_model(model, args.model_file)
    # Evaluate model
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Accuracy: {test_acc:.4f}")
    y_pred_probs = model.predict(X_test.astype(np.float32))
    y_pred = np.argmax(y_pred_probs, axis=1)
    f1 = f1_score(y_test, y_pred, average="weighted")
    mcc = matthews_corrcoef(y_test, y_pred)
    top_3_acc = top_k_accuracy(y_test, y_pred_probs, k=3)
    print(f"Metrics:")
    print(f"- F1 Score = {f1:.4f}")
    print(f"- MCC = {mcc:.4f}")
    print(f"- Top-3 Accuracy = {top_3_acc:.4f}")
