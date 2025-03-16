"""
mae_train.py --datafile <data_file> --modelfile <model_file> --pretrain <pretrain_model_file>
             [--encoder <label_encoder.pkl>] [--scaler <scaler.pkl>]
             [--mapping <activity_mapping.csv>] [--unwanted <unwanted_activities.csv>]

Unsupervised MAE pretrained activity recognition. This code trains a deep network activity recognizer
based on a pretrained model. This is the second of three steps that are: 1) pretrain (mae_pretrain.py),
2) train activity classifier (mae_train.py), and 3) evaluate the classifier (mae_test.py).

The program reads data from the given <data_file>, reads the pretrained model from the
<pretrain_model_file> and saves the trained AR model in Keras format to <model_file>.

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
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.saving import load_model, save_model

from utilities import create_encoder_from_data, create_scaler_from_data
from utilities import load_activity_mapping, load_unwanted_activities, map_and_filter_data

EPOCHS = 20 # Number of training epochs
BATCH_SIZE = 64 # Batch size for training
VALIDATION_SPLIT = 0.2 # Fraction of data to use for validation during training

def process_data(df, label_encoder, scaler):
    X = df.drop(columns=["stamp", "activity_label"], errors="ignore")
    y = df["activity_label"].values
    y_encoded = label_encoder.transform(y)
    X = X.fillna(0.0)
    X_scaled = scaler.transform(X).astype(np.float32)
    return X_scaled, y_encoded.astype(np.int32)

def build_classification_model(mae_model, num_classes):
    """ Attaches a classification head to the pretrained MAE model, then fine tunes model for classification. """
    inputs = mae_model.input
    x = mae_model(inputs)
    x = Dense(64, activation="relu", dtype=tf.float32)(x)
    x = Dropout(0.3)(x)
    outputs = Dense(num_classes, activation="softmax", dtype=tf.float32)(x)
    model = Model(inputs, outputs)
    model.compile(optimizer=Adam(learning_rate=0.0005), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model

def train_activity_recognition(X, y, mae_model, label_encoder):
    classification_model = build_classification_model(mae_model, num_classes=len(label_encoder.classes_))
    early_stopping = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True, verbose=1)
    classification_model.fit(X, y, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_split=VALIDATION_SPLIT, verbose=1, shuffle=True, callbacks=[early_stopping])
    return classification_model

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datafile', dest='data_file', type=str, required=True)
    parser.add_argument('--modelfile', dest='model_file', type=str, required=True)
    parser.add_argument('--pretrain', dest='pretrain_model_file', type=str, required=True)
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
    mae_model = load_model(args.pretrain_model_file)
    classification_model = train_activity_recognition(X, y, mae_model, label_encoder)
    save_model(classification_model, args.model_file)
