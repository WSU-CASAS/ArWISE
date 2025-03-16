"""
mae_pretrain.py --datafile <data_file> --modelfile <model_file> [--scaler <scaler.pkl>]

Unsupervised masked autoencoder (MAE) pretraining. This code builds a MAE pretrained model.
This is the first of three steps that are: 1) pretrain (mae_pretrain.py), 2) train activity
classifier (mae_train.py), and 3) evaluate the classifier (mae_test.py).

The program reads data from the given <data_file> and saves the pretrained model in
Keras format to <model_file>.

If scaler given, then use to scale data; otherwise, compute scaler from data.
"""

import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
import argparse
from keras.models import Model
from keras.layers import Dense, Dropout, Input, Masking
from keras.optimizers import Adam
from keras.saving import save_model

from utilities import create_scaler_from_data

# Suppress TensorFlow logs
#os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

EPOCHS = 10 # Number of training epochs
BATCH_SIZE = 32 # Batch size for training
VALIDATION_SPLIT = 0.2 # Fraction of data to use for validation during training

# Function to mask input features randomly
def mask_features(X, mask_ratio=0.3):
    mask = np.random.rand(*X.shape) > mask_ratio
    while np.any(mask.sum(axis=1) == 0):  # Ensure no all-zero rows
        mask = np.random.rand(*X.shape) > mask_ratio
    X_masked = X * mask.astype(np.float32)
    return X_masked

# Build an autoencoder for pretraining
def build_autoencoder(input_dim):
    inputs = Input(shape=(input_dim,), dtype=tf.float32)
    x = Dense(128, activation="relu", dtype=tf.float32)(inputs)
    x = Dropout(0.3)(x)
    x = Dense(64, activation="relu", dtype=tf.float32)(x)
    x = Dropout(0.3)(x)
    outputs = Dense(input_dim, activation="linear", dtype=tf.float32)(x)  # Reconstruct input
    autoencoder = Model(inputs, outputs)
    autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss="mse")
    return autoencoder

# Load data and train the autoencoder
def pretrain_autoencoder(df, scaler):
    df = df.drop(columns=["stamp_start", "stamp_end", "stamp", "activity_label"], errors='ignore')
    df = df.fillna(0.0)
    X = scaler.transform(df).astype(np.float32)
    del df
    X_masked = mask_features(X, mask_ratio=0.3)
    autoencoder = build_autoencoder(input_dim=X.shape[1])
    autoencoder.fit(X_masked, X, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_split=VALIDATION_SPLIT, verbose=1)
    return autoencoder

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datafile', dest='data_file', type=str, required=True)
    parser.add_argument('--modelfile', dest='model_file', type=str, required=True)
    parser.add_argument('--scaler', dest='scaler_file', type=str)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_arguments()
    # Get data
    df = pd.read_csv(args.data_file)
    # Build scalar
    scaler = None
    if args.scaler_file:
        scaler = joblib.load(args.scaler_file)
    if not scaler:
        scaler = create_scaler_from_data(df)
    # Train autoencoder
    autoencoder = pretrain_autoencoder(df, scaler)
    save_model(autoencoder, args.model_file)
    