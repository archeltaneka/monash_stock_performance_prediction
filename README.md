# FIT5149 Assignment 1: Monthly Stock Prediction

Assignment submission for FIT5149 assignment 1 about monthly stock prediction.


## Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Requirements](#requirements)
- [Project Structure](#project-structure)
---

## Project Overview

This project consists of exploratory data analysis (EDA) and machine learning skills by building predictive models for monthly stock relative performance. The task is to predict whether US stocks are likely to outperform or underperform the US Monash Index benchmark and the excess return value.

## Features

- Data loading flexibility (Build training data with or without optional macroeconomic features).
- Baseline (dummy) model comparison against five different models (linear/logistic regression, random forest, SVM, XGBoost, and Catboost).
- Hyperparameter tuning for all five models.
- Feature importance calculation.
- Forward feature selection.
- Ensemble model training.

## Installation

1. Create a new conda environment
```
conda create -n fit5149_assignment1 python=3.10 --y
conda activate fit5149_assignment1
```

2. Install dependencies
```
pip install -r requirements.txt
```
---

## Usage

Make sure that you are inside the project main directory.

### Prediction (for markers)

Running the prediction part can be done in two ways:

1. Run from script

```
python -m src.predict
```

2. Run from notebook

If running prediction from the script fails, open the `predict.ipynb` notebook under the main working directory and run all cells.

Then check under the `files/submission` directory. There should be `testing_targets.csv` and `kaggle_submission.csv` files

### Baseline

Replace the `task` from `config.yml` with either "classification" or "regression" to train different models separately.

```
python -m src.train_baseline
```

### Hyperparameter Tuning

Replace the `task` from `config.yml` with either "classification" or "regression" to tune different models separately.

```
python -m src.tune
```

### Feature Importance

Replace the `task` from `config.yml` with either "classification" or "regression".

```
python -m src.feature_importance
```

### Feature Selection
Replace the `task` from `config.yml` with either "classification" or "regression".

```
python -m src.feature_selection
```

### Train Ensemble
Replace the `task` from `config.yml` with either "classification" or "regression".

```
python -m src.train_ensemble
```

---

## Requirements

- Python 3.10++
- See `requirements.txt` for full list of dependencies.

## Project Structure
```
Group09_ass1_impl/
├── data/                       # Dataset files
├── files/
    ├── model_comparison_result # Model comparison metrics in csv files
    ├── models
        ├── classification
        ├── regression
    ├── plots                   # Feature importance and selection plot
    ├── submission              # Output submission CSV files
├── notebooks/                  # Jupyter notebooks
├── src/                        # Source code
    ├── __init__.py
    └── data/                   # Data builder modules
    └── feature/                # Feature importance and selector modules
        └── model/              # Model modules
        └── utils/              # Utility modules
    ├── feature_importance.py   # Find feature importance
    ├── feature_selection.py    # Forward feature selection
    └── predict.py              # final prediction
    └── train_baseline.py       # Train baseline model using default hyperparameters
    └── train_ensemble.py       # Train a final ensemble model from the tuned models
    └── tune.py                 # Perform hyperprameter tuning for all models
├── config.yml                  # Configuration file
├── prediction.ipynb            # Prediction notebook file if running using script is unsuccessful
├── requirements.txt            # Dependencies
├── README.md                   # Readme file
```
