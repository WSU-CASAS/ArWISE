"""
ft_train.py --datafile <data_file> --modelfile <model_file> [--scaler <scaler.pkl>]
            [--mapping <activity_mapping.csv>] [--unwanted <unwanted_activities.csv>]

Activity recognition with a FT-Transformer augmented network. This is step one of three steps: 1) train a
FT-Transformer model (ft_train.py), 2) create embeddings for training and test data (ft_embed.py), and
3) train and test a random forest using feature vectors augmented with FT-Transformer embeddings (ft_rf.py).

The program reads data from the given <data_file> and saves the trained FT model in Keras format to <model_file>.

If scaler given, then use to scale data; otherwise, compute scaler from data.
If unwanted activities given, then remove examples classified with these activities.
If activity mapping given, then use to map activities in data.
"""

import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
import argparse
import keras_tuner as kt
from keras.models import Model
from keras.layers import Dense, Dropout, Input, LayerNormalization, MultiHeadAttention, Reshape
from keras.saving import save_model

# Local imports
from utilities import create_scaler_from_data
from utilities import load_activity_mapping, load_unwanted_activities, map_and_filter_data

EPOCHS = 10 # Number of training epochs
BATCH_SIZE = 128 # Batch size for training
VALIDATION_SPLIT = 0.2 # Fraction of data to use for validation during training

def process_data(df, scaler):
    X = df.drop(columns=['stamp', 'activity_label'])
    X.fillna(0.0, inplace=True)
    X = scaler.transform(X).astype(np.float32)
    return X

def build_ft_transformer(hp, input_dim):
    """ Builds FT-Transformer model with tunable hyperparameters. """
    embed_dim = hp.Choice('embed_dim', [64])
    num_heads = hp.Choice('num_heads', [2])
    ff_dim = hp.Choice('ff_dim', [64])
    num_layers = hp.Choice('num_layers', [4])
    dropout = hp.Choice('dropout', [0.3])
    learning_rate = hp.Choice('learning_rate', [1e-4])

    inputs = Input(shape=(input_dim,), dtype=tf.float32)
    x = Dense(embed_dim, activation="relu")(inputs)
    x = Reshape((1, embed_dim))(x)

    for _ in range(num_layers):
        attn_output = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim, dtype=tf.float32)(x, x)
        attn_output = Dropout(dropout, dtype=tf.float32)(attn_output)
        attn_output = Dense(embed_dim, activation='linear', dtype=tf.float32)(attn_output)  # Ensure shape consistency
        x = LayerNormalization(epsilon=1e-6, dtype=tf.float32)(x + attn_output)  # Residual connection

        ff_output = Dense(ff_dim, activation='relu', dtype=tf.float32)(x)
        ff_output = Dropout(dropout, dtype=tf.float32)(ff_output)
        ff_output = Dense(embed_dim, activation='linear', dtype=tf.float32)(ff_output)  # Shape consistency
        x = LayerNormalization(epsilon=1e-6, dtype=tf.float32)(x + ff_output)

    x = Reshape((embed_dim,))(x)
    outputs = Dense(input_dim, activation='linear', name='feature_embeddings', dtype=tf.float32)(x)
    model = Model(inputs, outputs, name="FT_Transformer")
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss="mse")
    return model

def ft(X):
    input_dim = X.shape[1]
    tuner = kt.RandomSearch(lambda hp: build_ft_transformer(hp, input_dim), objective='val_loss', max_trials=1,
                            executions_per_trial=1, directory='ft_tuning', project_name='ft_transformer')
    tuner.search(X, X, epochs=1, batch_size=8, validation_split=0.2, verbose=1,
                 callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)])
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    best_model = tuner.hypermodel.build(best_hps)
    best_model.fit(X, X, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_split=VALIDATION_SPLIT, verbose=1,
                   callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)])
    return best_model

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datafile', dest='data_file', type=str, required=True)
    parser.add_argument('--modelfile', dest='model_file', type=str, required=True)
    parser.add_argument('--scaler', dest='scaler_file', type=str)
    parser.add_argument('--mapping', dest='activity_mapping_file', type=str)
    parser.add_argument('--unwanted', dest='unwanted_activities_file', type=str)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_arguments()
    # Get data
    df = pd.read_csv(args.data_file)
    # Build scalar, activity mapping, and unwanted activities (if provided)
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
    if not scaler:
        scaler = create_scaler_from_data(df)
    # Configure GPU(s) to use only needed memory
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    # Train model
    X = process_data(df, scaler)
    model = ft(X)
    save_model(model, args.model_file)
