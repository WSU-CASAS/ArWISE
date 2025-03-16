"""
cl_pretrain.py --datafile <data_file> --modelfile <model_file>
            [--encoder <label_encoder.pkl>] [--scaler <scaler.pkl>]
            [--mapping <activity_mapping.csv>] [--unwanted <unwanted_activities.csv>]

Supervised contrastive pretraining. This code builds a supervised contrastive pretrained model.
This is the first of three steps that are: 1) pretrain (cl_pretrain.py), 2) train activity
classifier (cl_train.py), and 3) evaluate the classifier (cl_test.py).

The program reads data from the given <data_file> and saves the trained model in
Keras format to <model_file>.

If encoder given, then use to encode activity labels; otherwise, compute encoder from data.
If scaler given, then use to scale data; otherwise, compute scaler from data.
If unwanted activities given, then remove examples classified with these activities.
If activity mapping given, then use to map activities in data.
"""

import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
import argparse
from keras.models import Model
from keras.layers import Dense, Dropout, Input, Layer
from keras.optimizers import Adam
from keras.saving import save_model, register_keras_serializable

from utilities import create_encoder_from_data, create_scaler_from_data
from utilities import load_activity_mapping, load_unwanted_activities, map_and_filter_data

EPOCHS = 20     # Number of training epochs
BATCH_SIZE = 64 # Batch size for training

def process_data(df, label_encoder, scaler):
    y = df["activity_label"].values
    y_encoded = label_encoder.transform(y)
    X = df.drop(columns=["stamp", "activity_label"], errors="ignore").fillna(0.0).astype(np.float32)
    X_scaled = scaler.transform(X).astype(np.float32)
    return X_scaled, y_encoded.astype(np.int32)

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

    # Prevent log(0) by ensuring min value
    sum_exp_logits = tf.clip_by_value(sum_exp_logits, 1e-9, tf.reduce_max(sum_exp_logits))

    log_prob = logits - tf.math.log(sum_exp_logits)
    positive_pairs = tf.reduce_sum(mask, axis=1)

    # Prevent division by zero for empty masks
    loss = -tf.reduce_mean(tf.reduce_sum(mask * log_prob, axis=1) / tf.maximum(positive_pairs, 1e-6))

    return loss

@register_keras_serializable()
class L2Normalization(Layer):
    def call(self, inputs):
        return tf.math.l2_normalize(inputs, axis=1)

# Contrastive Pretraining Model
def build_contrastive_model(input_shape, embedding_dim=64):
    inputs = Input(shape=input_shape, dtype=tf.float32)
    x = Dense(128, activation="relu", dtype=tf.float32)(inputs)
    x = Dropout(0.3)(x)
    x = Dense(64, activation="relu", dtype=tf.float32)(x)
    x = Dropout(0.3)(x)
    raw_embeddings = Dense(embedding_dim, activation=None, dtype=tf.float32)(x)
    outputs = L2Normalization()(raw_embeddings)
    model = Model(inputs, outputs)
    return model

# Pretrain Model with Contrastive Learning
def pretrain_contrastive(X_train, y_train):
    input_shape = X_train.shape[1:]
    contrastive_model = build_contrastive_model(input_shape)
    contrastive_model.compile(optimizer=Adam(learning_rate=0.0003, clipnorm=1.0), loss=supervised_contrastive_loss)
    contrastive_model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1, shuffle=True)
    return contrastive_model

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
    # Train and save model
    X, y = process_data(df, label_encoder, scaler)
    contrastive_model = pretrain_contrastive(X, y)
    save_model(contrastive_model, args.model_file)
