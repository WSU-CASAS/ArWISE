"""
dnn_train.py --datafile <data_file> --modelfile <model_file>
        [--encoder <label_encoder.pkl>] [--scaler <scaler.pkl>]
        [--mapping <activity_mapping.csv>] [--unwanted <unwanted_activities.csv>]
        [--augment]

Activity recognition with DNN and data augmentation to address class imbalance.
The program reads data from the given <data_file> and saves the trained model in
Keras format to <model_file>. The script dnn_test.py can then be used to evaluate
the model on test data.

If encoder given, then use to encode activity labels; otherwise, compute encoder from data.
If scaler given, then use to scale data; otherwise, compute scaler from data.
If unwanted activities given, then remove examples classified with these activities.
If activity mapping given, then use to map activities in data.
If --augment given, then the the data is augmented to improve model robustness.
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
import argparse
from keras.models import Model
from keras.layers import Dense, Dropout, Input
from keras.optimizers import Adam
from keras.saving import save_model
from keras.callbacks import EarlyStopping

from utilities import create_encoder_from_data, create_scaler_from_data
from utilities import load_activity_mapping, load_unwanted_activities, map_and_filter_data

# These features, if present in the input data, will be used to augment the data by permuting
# these values. Motion features are good candidates for permutation-based data augmentation.
PERMUTE_FEATURES = [
    'yaw_mean', 'pitch_mean', 'roll_mean',
    'rotation_rate_x_mean', 'rotation_rate_y_mean', 'rotation_rate_z_mean',
    'user_acceleration_x_mean', 'user_acceleration_y_mean', 'user_acceleration_z_mean']

EPOCHS = 20       # Number of training epochs
BATCH_SIZE = 32   # Batch size for training
VALIDATION_SPLIT = 0.2 # Fraction of data to use for validation during training

# Uncomment these for debugging
#os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
#os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

def process_data(df, label_encoder, scaler, augment):
    X = df.drop(columns=["stamp", "activity_label"], errors="ignore")
    y = df["activity_label"].values
    y_encoded = label_encoder.transform(y)
    X = X.fillna(0.0)
    X_scaled = scaler.transform(X).astype(np.float32)
    if augment:
        permute_feature_indices = [X.columns.get_loc(col) for col in PERMUTE_FEATURES if col in X.columns]
        print(permute_feature_indices)
        X_aug, y_aug = augment_data(X_scaled, y_encoded, permute_feature_indices)
        X_scaled = np.vstack((X_scaled, X_aug))
        y_encoded = np.hstack((y_encoded, y_aug))
    return X_scaled, y_encoded.astype(np.int32)

def augment_data(X, y, permute_feature_indices):
    """
    Generate approximately 5 synthetic points for every real point, inversely
    proportional to relative class size.
    Jitter: add small Gaussian noise to each feature.
    Scaling: multiply each feature by random factor between 0.9 and 1.1.
    Shuffling: randomly order points to avoid bias due to data point order.
    Permutation: within PERMUTE_FEATURES feature names.
    """
    augmented_X = []
    augmented_y = []
    class_counts = np.bincount(y)
    total_samples = len(y)
    class_weights = np.where(class_counts > 0, total_samples / (len(class_counts) * np.maximum(class_counts, 1)), 0.0).astype(np.float32)

    for class_idx, weight in enumerate(class_weights):
        num_to_generate = min(int(weight * 5), 100)  # Inversely proportional to class distribution
        class_indices = np.where(y == class_idx)[0]
        class_samples = X[class_indices]

        for _ in range(num_to_generate):
            sample_idx = np.random.choice(len(class_samples))
            sample = class_samples[sample_idx]

            if not np.isnan(sample).any():
                # Jittering (Gaussian noise)
                jitter_sample = sample + np.random.normal(scale=0.01, size=class_samples.shape[1]).astype(np.float32)
                augmented_X.append(jitter_sample)
                augmented_y.append(class_idx)

                # Feature-wise permutation within the same class
                if len(permute_feature_indices) > 0:
                    permuted_sample = sample.copy()
                    permuted_indices = np.random.permutation(permute_feature_indices)
                    permuted_sample[permute_feature_indices] = sample[permuted_indices]
                    augmented_X.append(permuted_sample)
                    augmented_y.append(class_idx)

    augmented_X = np.array(augmented_X, dtype=np.float32)
    augmented_y = np.array(augmented_y, dtype=np.int32)

    # Shuffle the augmented data
    indices = np.arange(len(augmented_X))
    np.random.shuffle(indices)
    augmented_X = augmented_X[indices]
    augmented_y = augmented_y[indices]

    return augmented_X.astype(np.float32), augmented_y.astype(np.int32)

def build_classification_model(input_shape, num_classes):
    inputs = Input(shape=input_shape, dtype=tf.float32)
    x = Dense(128, activation="relu", dtype=tf.float32)(inputs)
    x = Dropout(0.3)(x)
    x = Dense(64, activation="relu", dtype=tf.float32)(x)
    x = Dropout(0.3)(x)
    outputs = Dense(num_classes, activation="softmax", dtype=tf.float32)(x)
    model = Model(inputs, outputs)
    model.compile(optimizer=Adam(learning_rate=0.0001), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model

def ar(df, model_file, label_encoder, scaler, augment):
    X, y = process_data(df, label_encoder, scaler, augment)
    model = build_classification_model(X.shape[1:], len(label_encoder.classes_))
    early_stopping = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True, verbose=1)
    model.fit(X, y, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_split=VALIDATION_SPLIT, verbose=1, shuffle=True, callbacks=[early_stopping])
    save_model(model, model_file)

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datafile', dest='data_file', type=str, required=True)
    parser.add_argument('--modelfile', dest='model_file', type=str, required=True)
    parser.add_argument('--encoder', dest='label_encoder_file', type=str)
    parser.add_argument('--scaler', dest='scaler_file', type=str)
    parser.add_argument('--mapping', dest='activity_mapping_file', type=str)
    parser.add_argument('--unwanted', dest='unwanted_activities_file', type=str)
    parser.add_argument('--augment', dest='augment', action='store_true')
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
    # Train model
    ar(df, args.model_file, label_encoder, scaler, args.augment)
