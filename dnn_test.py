"""
dnn_test.py --datafile <data_file> --modelfile <model_file>
            [--encoder <label_encoder.pkl>] [--scaler <scaler.pkl>]
            [--mapping <activity_mapping.csv>] [--unwanted <unwanted_activities.csv>]

Activity recognition with DNN and data augmentation to address class imbalance.
This code evaluates the given model in <model_file> on the given dataset in <data_file>.

If encoder given, then use to encode activity labels; otherwise, compute encoder from data.
If scaler given, then use to scale data; otherwise, compute scaler from data.
If unwanted activities given, then remove examples classified with these activities.
If activity mapping given, then use to map activities in data.

The same encoder and scaler should be used here that were used to train the model.
"""

import argparse
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from sklearn.metrics import confusion_matrix, f1_score, matthews_corrcoef
from tensorflow.keras.models import load_model

from utilities import create_encoder_from_data, create_scaler_from_data
from utilities import load_activity_mapping, load_unwanted_activities, map_and_filter_data

def process_data(df, label_encoder, scaler):
    X = df.drop(columns=["stamp", "activity_label"], errors="ignore")
    y = df["activity_label"].values
    y_encoded = label_encoder.transform(y)
    X = X.fillna(0.0)
    X_scaled = scaler.transform(X).astype(np.float32)
    return X_scaled, y_encoded.astype(np.int32)

def top_k_accuracy(y_true, y_pred, k=1):
    y_true_tensor = tf.convert_to_tensor(y_true, dtype=tf.int32)
    metric = tf.keras.metrics.sparse_top_k_categorical_accuracy(y_true_tensor, y_pred, k=k)
    return float(tf.reduce_mean(metric).numpy())

def evaluate_model(classification_model, X, y, encoder):
    test_loss, test_accuracy = classification_model.evaluate(X, y, verbose=0)
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

    y_pred_probs = classification_model.predict(X.astype(np.float32))
    y_pred = np.argmax(y_pred_probs, axis=1)

    f1 = f1_score(y, y_pred, average="weighted")
    mcc = matthews_corrcoef(y, y_pred)
    top_3_acc = top_k_accuracy(y, y_pred_probs, k=3)
    print(f"Metrics:")
    print(f"- F1 Score = {f1:.4f}")
    print(f"- MCC = {mcc:.4f}")
    print(f"- Top-3 Accuracy = {top_3_acc:.4f}")

    #y_pred_labels = label_encoder.inverse_transform(y_pred)
    #y_labels = label_encoder.inverse_transform(y)
    #cm = confusion_matrix(y_labels, y_pred_labels, labels=label_encoder.classes_)
    #print("\nConfusion Matrix:")
    #cm_df = pd.DataFrame(cm, index=label_encoder.classes_, columns=label_encoder.classes_)
    #print(cm_df)

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datafile', dest='data_file', type=str, required=True)
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
    df = pd.read_csv(args.data_file)
    # Build label_encoder, scalar, activity mapping, and unwanted activities (if provided)
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
    df = map_and_filter_data(df, activity_mapping, unwanted_activities)
    if not label_encoder:
        label_encoder = create_encoder_from_data(df)
    if not scaler:
        scaler = create_scaler_from_data(df)
    # Evaluate model on data
    X, y = process_data(df, label_encoder, scaler)
    classification_model = load_model(args.model_file, compile=True)
    evaluate_model(classification_model, X, y, label_encoder)
