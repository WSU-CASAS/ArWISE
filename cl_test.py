"""
cl_test.py --datafile <data_file> --modelfile <model_file>
          [--encoder <label_encoder.pkl>] [--scaler <scaler.pkl>]
          [--mapping <activity_mapping.csv>] [--unwanted <unwanted_activities.csv>]

Supervised contrastive pretraining activity recognition. This code tests a deep network
activity recognizer based on a pretrained model. This is the third of three steps that
are: 1) pretrain (cl_pretrain.py), 2) train activity classifier (cl_train.py), 
and 3) evaluate the classifier (cl_test.py).

The program reads data from the given <data_file> and reads the pretrained model from the
<model_file>. The code reports performance in terms of accuracy, f1 score, mcc, and top-3
accuracy.

If encoder given, then use to encode activity labels; otherwise, compute encoder from data.
If scaler given, then use to scale data; otherwise, compute scaler from data.
If unwanted activities given, then remove examples classified with these activities.
If activity mapping given, then use to map activities in data.
"""

import numpy as np
import pandas as pd
import joblib
import argparse
import tensorflow as tf
from sklearn.metrics import f1_score, matthews_corrcoef
from keras.layers import Layer
from keras.models import Model, load_model
from keras.saving import register_keras_serializable

from utilities import create_encoder_from_data, create_scaler_from_data
from utilities import load_activity_mapping, load_unwanted_activities, map_and_filter_data

def process_data(df, label_encoder, scaler):
    X = df.drop(columns=["stamp", "activity_label"], errors="ignore").fillna(0.0).astype(np.float32)
    y = df["activity_label"].values
    y_encoded = label_encoder.transform(y)
    X = X.fillna(0.0)
    X = scaler.transform(X).astype(np.float32)
    return X.astype(np.float32), y_encoded.astype(np.int32)

@register_keras_serializable()
class L2Normalization(Layer):
    def call(self, inputs):
        return tf.math.l2_normalize(inputs, axis=1)

@register_keras_serializable()
def supervised_contrastive_loss(labels, embeddings, temperature=0.1):
    """ Computes supervised contrastive loss with improved stability. """
    embeddings = tf.math.l2_normalize(embeddings, axis=1)  # Normalize embeddings
    logits = tf.matmul(embeddings, embeddings, transpose_b=True) / temperature  # Cosine similarity
    labels = tf.reshape(labels, [-1, 1])  # Reshape for broadcasting
    mask = tf.equal(labels, tf.transpose(labels))  # Mask for positive pairs
    mask = tf.cast(mask, tf.float32)

    # Remove self-contrast pairs (diagonal should be zero)
    mask = tf.linalg.set_diag(mask, tf.zeros_like(tf.linalg.diag_part(mask)))

    # Compute loss with numerical stability fixes
    exp_logits = tf.exp(logits)
    sum_exp_logits = tf.reduce_sum(exp_logits, axis=1, keepdims=True)
    sum_exp_logits = tf.clip_by_value(sum_exp_logits, 1e-9, tf.reduce_max(sum_exp_logits))  # Prevent log(0)

    log_prob = logits - tf.math.log(sum_exp_logits)
    positive_pairs = tf.reduce_sum(mask, axis=1)

    # Prevent division by zero
    loss = -tf.reduce_mean(tf.reduce_sum(mask * log_prob, axis=1) / tf.maximum(positive_pairs, 1e-6))

    return loss

def top_k_accuracy(y_true, y_pred, k=1):
    metric = tf.keras.metrics.sparse_top_k_categorical_accuracy(y_true, y_pred, k=k)
    return float(tf.reduce_mean(metric).numpy())

def evaluate_model(classification_model, X, y, num_classes):
    classification_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    test_loss, test_accuracy = classification_model.evaluate(X, y, verbose=0)
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

    y_pred_probs = classification_model.predict(X.astype(np.float32))
    y_pred = np.argmax(y_pred_probs, axis=1)

    f1 = f1_score(y, y_pred, average="weighted")
    mcc = matthews_corrcoef(y, y_pred)
    top_3_accuracy = top_k_accuracy(y, y_pred_probs, k=3)
    print(f"Metrics:")
    print(f"- F1 Score = {f1:.4f}")
    print(f"- MCC = {mcc:.4f}")
    print("- Top-3 Accuracy =", top_3_accuracy)

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
    num_classes=len(label_encoder.classes_)
    classification_model = load_model(args.model_file, compile=False)
    evaluate_model(classification_model, X, y, num_classes)
