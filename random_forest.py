"""
random_forest.py --train <train.csv> --test <test.csv> --modelfile <model_file.pkl>

This code trains and tests a Random Forest activity recognition model. The model is
saved in <model_file.pkl>.
"""

import argparse
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, matthews_corrcoef

NUM_TREES = 100 # Number of estimators (trees) to use in the RF classifier

np.set_printoptions(threshold=np.inf, suppress=True)

def load_data(train_data, test_data):
    y_train = train_data["activity_label"]
    y_test = test_data["activity_label"]
    X_train = train_data.drop(columns=["stamp", "activity_label"])
    X_test = test_data.drop(columns=["stamp", "activity_label"])
    return X_train, y_train, X_test, y_test

def learn_rf(X_train, y_train, X_test, y_test):
    clf = RandomForestClassifier(n_estimators=NUM_TREES, random_state=42, class_weight='balanced')
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)
    class_labels = clf.classes_

    print('Accuracy:', accuracy_score(y_test, y_pred))
    print('F1 Score:', f1_score(y_test, y_pred, average='weighted'))
    print('Precision', precision_score(y_test, y_pred, average='weighted'))
    print('Recall:', recall_score(y_test, y_pred, average='weighted'))
    print('MCC:', matthews_corrcoef(y_test, y_pred))

    y_test = np.array(y_test)
    top_3_indices = np.argpartition(y_proba, -3, axis=1)[:, -3:]  # Select top-3 without sorting
    top_3_sorted_indices = np.argsort(y_proba[np.arange(y_proba.shape[0])[:, None], top_3_indices], axis=1)[:, ::-1]
    top_3_indices = top_3_indices[np.arange(y_proba.shape[0])[:, None], top_3_sorted_indices]  # Reorder indices properly
    top_3_classes = class_labels[top_3_indices]
    is_correct = np.any(top_3_classes == y_test[:, None], axis=1)
    top_3_result = np.sum(is_correct)
    top_3_accuracy = top_3_result / len(y_test)
    print('Top-3 Correct:', top_3_result, ', Total:', len(y_test), ', Top-3 Accuracy:', top_3_accuracy)
    return clf

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', dest='train_file', type=str, required=True)
    parser.add_argument('--test', dest='test_file', type=str, required=True)
    parser.add_argument('--modelfile', dest='model_file', type=str, required=True)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_arguments()
    train_data = pd.read_csv(args.train_file)
    test_data = pd.read_csv(args.test_file)
    X_train, y_train, X_test, y_test = load_data(train_data, test_data)
    model = learn_rf(X_train, y_train, X_test, y_test)
    joblib.dump(model, args.model_file)
    