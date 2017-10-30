# README
Authors: Kirill IVANOV, Matthias RAMIREZ, Nicolas TALABOT

Version used:
* `conda: 4.3.30`
* `python: 3.6.2`
* `numpy: 1.13.3`

## Launching `run.py`
In order to use `run.py`, follow these simple steps:
1. Put the *train.csv* and *test.csv* files in the same folder as `run.py`
2. Open a terminal and run `python run.py`
3. Wait for it to compute the model and create the submission file (it should take 3 to 5 min)
4. A new file *run_submission.csv* has been created, which contains the prediction for the Kaggle competition

## Code architecture
### `run.py`
The script to load the dataset, learn a model, predict the labels of the test set, and create a submission file.

3 parameters can be changed here:
* the seed for the random number generation (default = 3)
* the degree of the polynomial features (default = 11)
* the ratio of the train set actually going to training (default = 0.66)

### `utilites.py`
Functions used by `run.py` to split the train dataset, and for data pre-processing.

The main function for pre-processing `preprocess_data` takes as input the original dataset, and the degree of the polynomial features.
Additionally, a flag `compute_mean_std` indicates if the pre-processing should compute the mean and standard deviation of each feature,
or rather use the ones provided (useful for pre-processing the test set).

The pre-processing is composed of the following steps:
1. Build the polynomial features
2. Standardize the resulting features
3. Set the missing values to 0
4. Add binary features

### `implementations.py`
Implements the 6 ML method seen in the lectures. For `logistic_regression` (and `reg_logistic_regression`), the gradient descent
algorithm is used, rather than stochastic gradient descent.

### `proj1_helpers.py`
Provided functions to load the dataset, and create a submission file.
