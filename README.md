# ArWISE

Activity recognition from in-the-WIld SmartwatchEs

This repository contains the code used to process the associated ArWISE dataset available at [https://doi.org/10.5061/dryad.jdfn2z3nm](https://doi.org/10.5061/dryad.jdfn2z3nm).  The `generate-dataset.py` script can be used to generate training and testing sets from the Dryad datasets suitable for machine learning methods. The `generate-features.py` script was used to generate the datasets and is included here to understand the details of how the datasets were generated.

For visualizing data and results, the `visualize.py` script plots a pca and umap of a dataset stored in data.csv. The `confusion_wheel.py` script creates a png file visualizing a confusion wheel based on values provided in confusion_matrix.csv.

To predict activity labels, a set of alternative models are provided. These include a 1D CNN that processes time series data, a DNN, a contrastive pretrainer, a masked autoencoder pretrainer, a random forest, and a ft-transformer-augmented random forest. These are created using scripts `cnn.py`, `dnn_train.py` and `dnn_test.py`, `cl_pre.py`, `cl_train.py`, and `cl_test.py`, `mae_pre.py`, `mae_train.py`, and `mae_test.py`, `rf.py`, and `ft.py`.

Library used to build machine learning models for detecting general human activity classes from smart watch data.

## Generate Dataset

The `generate-data.py` script generates training and testing sets suitable for machine learning methods.

```
python generate-dataset.py --inputfile <input_file.csv> [--testsize 0.2] [--windowsize 300] [--seed N]
```

The input file is expected to be in the format that is output by the below `generate-features.py` script with the `--timewindow` option given; namely, each line begins with a `stamp_begin` and `stamp_end` timestamp, followed by the feature values, followed by the `activity_label` (str or None). The script generates two files: `<input_file>-train.csv` and `<input_file>-test.csv`, in the same format as the input, except only one timestamp `stamp` is given, which is the end of the window. The test file contains labeled examples that cover approximately `--testsize` (default=0.2) of the time span of the input file, and the train file contains the rest of the examples that do not overlap with the test examples. The script selects examples for the test set by randomly selecting an unused example and collecting all the nearby unused examples within a time window of size `--windowsize` seconds (default=300). The random selection is controlled by the `--seed` if given; otherwise, the seed is randomly chosen.

## Generate Features

The `generate-features.py` script generates feature-based datasets given timestamped sensor data from smart devices.

```
python generate-features.py --inputfile <input_file> --outputfile <output_file> [--backfill 0] [--downsample None] [--raw] [--windowsize 300] [--stepsize 300] [--mindata 1] [--timewindow]
```

The input file is expected to be a CSV file in the CASAS format. For details on this format, see [https://github.com/WSU-CASAS/smartwatch-data](https://github.com/WSU-CASAS/smartwatch-data). Scroll to the bottom of the README for pointers to actual data.

Generated features and the activity label are written to the given output file. To choose the activity label, the script first checks for a non-Null user_activity_label` value at the end of the window. If found, then uses that value. Otherwise, the code checks for a non-Null `activity_label` value at the end of the window. If found, then uses that value. If neither yields a non-Null value, then the activity label is set to None.

The `--backfill` option specifies the number of seconds into the past to copy a `user_activity_label`. The default of 0 means no backfill.

If a `--downsample` rate is given (in samples/second Hz), then the data is first resampled at this rate. The downsample rate is assumed to be less than the original sampling rate of the data.

By default, features are generated as aggregated values over the window. But if the `--raw` option is given, then all the values over all the rows in the window are used as features, not the aggregated values. See the `raw_features` list in `compute_features_raw` for the raw features that are extracted.

Features are generated by passing a window of size `--windowsize` (in seconds, default=300) over the data, advancing the window by `--stepsize` (in seconds, default=300). Each window must have at least `--mindata` (default 1) entries from the downsampled input data file to be a viable window.

By default, each output row has a `stamp` timestamp feature, which is the latest time in the window of data used to generate the feature values. If the `--timewindow` option is given, then both the start and end times are included in the features using feature names `stamp_start` and `stamp_end`. If you intend to run the `generate_dataset.py` script below on the output features, then the `--timewindow` option is necessary here.

You can add new features in the `compute_features` method. See the existing features there for examples.

## Visualize Data

The `visualize.py` script creates a 2D PCA plot and 2D UMAP plot of the data contains in the file `data.csv.` The resulting plots are stored in files `pca.png` and `umap.png.` The first field in data.csv is assumped to be a time stamp (yyyy-mm-dd hh:mm:ss.ffffff) and the last field is a string activity label. The remaining fields represent the feature vector that is visualized. The colors in the plots represent the value of the activity for the corresponding data point.

```
python visualize.py
```

The `confusion_wheel.py` script generates a confusion wheel plot to visualize a confusion matrix set of classification results, stored in file `confusion_matrix.csv.` The whell is stored in file `cw.png.`

```
python confusion_wheel.py
```

## Models

## 1D CNN

The `cnn.py` script trains and tests a 1D CNN human activity recognizer. This model processes time series data. The code assumes each data point is described by 8 features at 100 consecutive time points, the data have been separated into train and test files, and the files are stored in directory `cnndata`.

```
python cnn.py
```

## DNN

The `dnn_train.py` script trains deep neural network human activity recognizer. This model processes tabular features describing the data points. The code assumes the training data are available in file `data/train.csv`. The `--augment` option specifies that synthetic data points be added to the training data. The number of synthetic points for each class type is inversely proportional to the relative class size. This option maybe to used to address potential class imbalance.

```
python dnn_train.py [--augment]
```

The `dnn_test.py` script tests the deep neural network human activity recognizer. This model processes tabular features describing the data points. The code assumes the model has been trained and is available in `models/dnn_model.keras`. The code also assumes test data are available in file `data/test.csv`. This script processes the data using the trained model and reports performance in terms of accuracy, f1 score, mcc, and top-3 accuracy.

```
python dnn_test.py
```

## DNN with Contrastive Pretraining

The scripts `cl_pretrain.py`, `cl_train.py`, and `cl_test.py` train and evaluate a deep network for activity recognition
that is boosted by contrastive pretraining.

The `cl_pretrain.py` script pretrains an embedding model. The code assumes the tabular training data are available in file `data/train.csv`.  The model is trained to keep points from the same class close together and points from different classes farther apart.

```
python cl_pretrain.py
```

The `cl_train.py` script trains an activity classification model using the pretrained contrastive embedder. The code assumes the tabular training data are available in file `data/train.csv` and the pretrained model is available in `models/contrastive_pretrained.keras`.  The trained model is stored in `models/contrastive_ar_model.keras`.

```
python cl_train.py
```

The `cl_test.py` script tests an activity classification model that was trained using a deep neural network and a contrastive pretrained model. The code assumes the tabular test data are available in file `data/test.csv` and the trained model is available in `models/contrastive_ar_model.keras`. This script processes the data using the trained model and reports performance in terms of accuracy, f1 score, mcc, and top-3 accuracy.

```
python cl_test.py
```

## DNN with Masked Autoencoder Pretraining

The scripts `mae_pretrain.py`, `mae_train.py`, and `mae_test.py` train and evaluate a deep network for activity recognition
that is boosted by contrastive pretraining.

The `mae_pretrain.py` script pretrains an embedding model. The code assumes the tabular training data are available in file `data/data.csv`.  The model is trained to reconstruct points using a masked autoencoder and learning a representation for the data.

```
python mae_pretrain.py
```

The `mae_train.py` script trains an activity classification model using the MAE pretrained model. The code assumes the tabular training data are available in file `data/train.csv` and the pretrained model is available in `models/mae_pretrained.keras`.  The trained model is stored in `models/mae_ar_model.keras`.

```
python mae_train.py
```

The `mae_test.py` script tests an activity classification model that was trained using a deep neural network and a MAE pretrained model. The code assumes the tabular test data are available in file `data/test.csv` and the trained model is available in `models/mae_ar_model.keras`. This script processes the data using the trained model and reports performance in terms of accuracy, f1 score, mcc, and top-3 accuracy.

```
python mae_test.py
```

## Random Forest

The `random_forest.py` script trains and evaluates a random forest activity recognition model. The code assumes the tabular training data are available in `data/train.csv` and test data are available in `data/test.csv`. This script processes the data using the trained model and reports performance in terms of accuracy, f1 score, mcc, and top-3 accuracy.

## FT-Transformer Augmented Random Forest

The scripts `ft_train.py`, `ft_embed.py`, and `rf_ft.py` train and evaluate an activity recognition model based on
random forest with features that are augmented using FT-Transformer embeddings.

The `ft_train.py` script trains an embedding model. The code assumes tabular data are available in file `data/data.csv`.

```
python ft_train.py
```

The `ft_embed.py` uses the embedding model to generate augmented features for training and test data.  The code assumes the tabular training and test data are available in file `data/train.csv` and `data/test.csv`. The pretrained model should be available in `models/ft_embedding_model.keras`. The code stores the augmented features in `data/ft_train.csv` and `data/ft_test.csv`.

```
python ft_embed.py
```

The `rf_ft.py` script script trains and evaluates a random forest activity recognition model using FT-Transformer augmented feature vectors. The code assumes the tabular training data are available in `data/ft_train.csv` and test data are available in `data/ft_test.csv`. This script processes the data using the trained model and reports performance in terms of accuracy, f1 score, mcc, and top-3 accuracy.

```
python rf_ft.py
```

## Utilities

Several utilities are provided. These include a `scaler.py` script that trains a normalization scaler on all available data (this is used by most of the models). These also includes `activity_mapping.csv`, that contains a list mapping labels found in the original data to the activity categories learned by the models, `unwanted_activities.csv` that filters out unwanted activity categories, `scaler.pkl` that is a normalization model trained on the ArWISE data, and `label_encoder.pkl` that is a model for encoding ArWISE string activity labels to integer representations.

## Datasets

Datasets that can be used as input to the `generate-dataset.py` script are available at the Dryad repository [https://doi.org/10.5061/dryad.jdfn2z3nm](https://doi.org/10.5061/dryad.jdfn2z3nm). These datasets were generated using the following command:

```
python generate-features.py --inputfile input.csv --outputfile output.csv --backfill 300 --windowsize 60 --stepsize 1 --timewindow
```

Datasets that can be used as input to the `generate-features.py` script are available at [https://github.com/WSU-CASAS/smartwatch-data](https://github.com/WSU-CASAS/smartwatch-data). Scroll to the bottom of the README for pointers to actual data.

Smaller datasets, including `train.csv`, `test.csv`, and `data.csv`, are included with this distribution to facilitate testing the scripts.

## Acknowledgements

The code was designed by Dr. Diane Cook (djcook@wsu.edu) and Dr. Lawrence Holder (holder@wsu.edu) with help from the WSU CASAS Lab [https://casas.wsu.edu](https://casas.wsu.edu). This work was supported by NIH/NIA grants R035AG071451 and R01AG065218, and NSF/IIS grant 1954372.
