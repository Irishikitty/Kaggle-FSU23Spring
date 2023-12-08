# Kaggle-FSU23Spring: Housing Prices Prediction

## Overview
- overview.ipynb: includes concise introduction of preprocessing data, performing feature engineering and modeling.
- eda_v1.ipynb: contains preprocessing details and first try.
- variables.txt: contains useful variables.

### Hyperparameters Tuning:
  - random_forest_tune.py
  - gbm_tune.py

Each of them contains 50 trials, and search spaces are included in the *overview.ipynb*

### Useful resources:

- Yggdrasil backend: https://dl.acm.org/doi/10.1145/3580305.3599933#sec-supp
- Model hyperparameters: 
  - https://www.tensorflow.org/decision_forests/api_docs/python/tfdf/keras/RandomForestModel
  - https://www.tensorflow.org/decision_forests/api_docs/python/tfdf/keras/GradientBoostedTreesModel
