"""
ft_embed.py --train <train.csv> --test <test.csv> --pretrain <pretrain_model_file>
            [--scaler <scaler.pkl>] [--mapping <activity_mapping.csv>] [--unwanted <unwanted_activities.csv>]

Activity recognition with a FT-Transformer augmented network. This is step two of three steps: 1) train a
FT-Transformer model (ft_train.py), 2) create embeddings for training and test data (ft_embed.py), and
3) train and test a random forest using feature vectors augmented with FT-Transformer embeddings (ft_rf.py).

The program reads in training <train.csv> and testing <test.csv> data, and a pretrained embedding model
built using ft_train.py. The data is rewritten to <train_ft.csv> and <test_ft.csv> with the embedding
features added.

If scaler given, then use to scale data; otherwise, compute scaler from data.
If unwanted activities given, then remove examples classified with these activities.
If activity mapping given, then use to map activities in data.
"""

import os
import argparse
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from keras.saving import load_model

# Local imports
from utilities import create_scaler_from_data, map_and_filter_data
from utilities import load_activity_mapping, load_unwanted_activities

def process_data(train_data, test_data, scaler):
    train_data.drop(columns=["stamp"], inplace=True)
    test_data.drop(columns=["stamp"], inplace=True)

    y_train = train_data["activity_label"]
    y_test = test_data["activity_label"]
    X_train = train_data.drop(columns=["activity_label"])
    X_test = test_data.drop(columns=["activity_label"])
    X_train.fillna(0.0, inplace=True)
    X_test.fillna(0.0, inplace=True)

    X_train = scaler.transform(X_train).astype(np.float32)
    X_test = scaler.transform(X_test).astype(np.float32)
    return X_train, X_test, y_train, y_test

def extract_embeddings(model, X):
    """ Extract learned feature embeddings from the FT-Transformer. """
    return model.predict(X, batch_size=32).astype(np.float32)

def augment_features(X_train, X_test, y_train, y_test, ft_transformer):
    X_train_emb = extract_embeddings(ft_transformer, X_train)
    X_test_emb = extract_embeddings(ft_transformer, X_test)
    X_train_combined = np.hstack((X_train, X_train_emb))
    X_test_combined = np.hstack((X_test, X_test_emb))
    ft_train = pd.DataFrame(X_train_combined)
    ft_test = pd.DataFrame(X_test_combined)
    ft_train["activity_label"] = y_train.values
    ft_test["activity_label"] = y_test.values
    return ft_train, ft_test

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', dest='train_file', type=str, required=True)
    parser.add_argument('--test', dest='test_file', type=str, required=True)
    parser.add_argument('--pretrain', dest='pretrain_model_file', type=str, required=True)
    parser.add_argument('--scaler', dest='scaler_file', type=str)
    parser.add_argument('--mapping', dest='activity_mapping_file', type=str)
    parser.add_argument('--unwanted', dest='unwanted_activities_file', type=str)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_arguments()
    # Get data
    train_df = pd.read_csv(args.train_file)
    test_df = pd.read_csv(args.test_file)
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
    train_df = map_and_filter_data(train_df, activity_mapping, unwanted_activities)
    test_df = map_and_filter_data(test_df, activity_mapping, unwanted_activities)
    if not scaler:
        scaler = create_scaler_from_data(train_df)
    # Configure GPU(s) to use only needed memory
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    # Process data, load model, augment datasets with FT embedding features, and save
    X_train, X_test, y_train, y_test = process_data(train_df, test_df, scaler)
    ft_transformer = load_model(args.pretrain_model_file)
    ft_train, ft_test = augment_features(X_train, X_test, y_train, y_test, ft_transformer)
    ft_train_file = os.path.splitext(args.train_file)[0] + '_ft.csv'
    ft_test_file = os.path.splitext(args.test_file)[0] + '_ft.csv'
    ft_train.to_csv(ft_train_file, index=False)
    ft_test.to_csv(ft_test_file, index=False)
    