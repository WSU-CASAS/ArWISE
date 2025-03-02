# ArWISE

Activity recognition from in-the-WIld SmartwatchEs

This repository contains the code used to process the associated ArWISE dataset available at [https://doi.org/10.5061/dryad.jdfn2z3nm](https://doi.org/10.5061/dryad.jdfn2z3nm). The `generate-dataset.py` script can be used to generate training and testing sets from the Dryad datasets suitable for machine learning methods. The `generate-features.py` script was used to generate the datasets and is included here to understand the details of how the datasets were generated. 

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

## Datasets

Datasets that can be used as input to the `generate-dataset.py` script are available at the Dryad repository [https://doi.org/10.5061/dryad.jdfn2z3nm](https://doi.org/10.5061/dryad.jdfn2z3nm). These datasets were generated using the following command:

```
python generate-features.py --inputfile input.csv --outputfile output.csv --backfill 300 --windowsize 60 --stepsize 1 --timewindow
```

Datasets that can be used as input to the `generate-features.py` script are available at [https://github.com/WSU-CASAS/smartwatch-data](https://github.com/WSU-CASAS/smartwatch-data). Scroll to the bottom of the README for pointers to actual data.

## Acknowledgements

The code was designed by Dr. Diane Cook (djcook@wsu.edu) and Dr. Lawrence Holder (holder@wsu.edu) with help from the WSU CASAS Lab [https://casas.wsu.edu](https://casas.wsu.edu). This work was supported by NIH/NIA grants R035AG071451 and R01AG065218, and NSF/IIS grant 1954372.
