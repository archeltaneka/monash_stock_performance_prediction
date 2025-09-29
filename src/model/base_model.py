from abc import ABC, abstractmethod
import pickle

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, f1_score

from src.feature.preprocessor import create_feature_preprocessor
from src.utils.cross_validation import time_series_cv_splits


class BaseModel(ABC):
    """
    Abstract base class for machine learning models with support for
    preprocessing, time-series cross-validation, training, prediction, 
    saving, and loading models.

    Attributes
    ----------
    config : dict
        Configuration dictionary for model hyperparameters and preprocessing.
    task : str
        Task type: "classification" or "regression".
    preprocessor : sklearn.preprocessing.Transformer
        Feature preprocessor returned from `create_feature_preprocessor`.
    model : object
        The machine learning model instance (e.g., CatBoost, LogisticRegression).

    Methods
    -------
    fit(X: pd.DataFrame, y: pd.Series, df: pd.DataFrame, n_splits: int = 30,
        task: str = "classification", window_type: str = "expanding", warmup: int = 12) -> list[float]
        Train the model using time-series cross-validation and return validation metrics.
    tune()
        Abstract method for hyperparameter tuning; must be implemented by subclasses.
    predict(X: pd.DataFrame) -> np.ndarray
        Predict target values for given features.
    save(model_fname: str) -> None
        Save the trained model to a file using pickle.
    load(model_fname: str) -> None
        Load a trained model from a pickle file.
    """

    def __init__(self, config: dict, task: str):
        self.config = config
        self.task = task
        self.preprocessor = create_feature_preprocessor(
            self.config["feature_preprocessor"]["categorical_features"], 
            self.config["feature_preprocessor"]["ordinal_features"], 
            self.config["feature_preprocessor"]["numerical_features"]
        )
        self.model = None

    def fit(
        self, 
        X: pd.DataFrame, 
        y: pd.Series, 
        df: pd.DataFrame, 
        n_splits: int = 30, 
        task: str = "classification",
        window_type: str = "expanding", 
        warmup: int = 12
    ) -> list[float]:
        """
        Fit the model using time-series cross-validation and return per-fold validation metrics.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix for training.
        y : pd.Series
            Target vector.
        df : pd.DataFrame
            Full dataset containing the 'month_id' column for time-series splitting.
        n_splits : int, default=30
            Number of time-series CV folds.
        task : str, default="classification"
            Task type: "classification" or "regression".
        window_type : str, default="expanding"
            Type of rolling window to use for cross-validation ("expanding" or "sliding").
        warmup : int, default=12
            Number of initial periods to skip when using rolling time-series CV.

        Returns
        -------
        metric_scores : list[float]
            List of validation metrics per fold. F1 scores for classification, RMSE for regression.
        """

        metric_scores = []

        for fold, (train_idx, val_idx) in enumerate(time_series_cv_splits(df, 
                                                                          warmup=warmup,
                                                                          n_splits=n_splits,
                                                                          window_type=window_type)):
            X_train_cv, X_val_cv = X.iloc[train_idx], X.iloc[val_idx]
            y_train_cv, y_val_cv = y.iloc[train_idx], y.iloc[val_idx]

            # Fit model
            self.model.fit(X_train_cv, y_train_cv)

            # Predictions
            if task == "classification":
                val_preds = self.model.predict(X_val_cv)
                metric = f1_score(y_val_cv, val_preds)
            elif task == "regression":
                val_preds = self.model.predict(X_val_cv)
                metric = np.sqrt(mean_squared_error(y_val_cv, val_preds))

            # Log metric for each fold
            metric_scores.append(metric)
            print(f"Fold {fold+1}: Validation Month={df.loc[val_idx,'month_id'].unique()[0]}, Metric={metric:.4f}")

        if task == "classification":
            print(f"Mean F1 Score across folds: {np.mean(metric_scores):.4f}")
        elif task == "regression":
            print(f"Mean RMSE across folds: {np.mean(metric_scores):.4f}")

        return metric_scores

    @abstractmethod
    def tune(self):
        """
        Abstract method for hyperparameter tuning. Subclasses must implement this.
        """
        
        raise NotImplementedError

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using the trained model.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix for which to make predictions.

        Returns
        -------
        np.ndarray
            Predicted values.
        """
        
        return self.model.predict(X)

    def save(self, model_fname: str) -> None:
        """
        Save the trained model to a pickle file.

        Parameters
        ----------
        model_fname : str
            Path to save the model.
        """

        with open(model_fname, 'wb') as file:
            pickle.dump(self.model, file)

    def load(self, model_fname: str) -> None:
        """
        Load a trained model from a pickle file.

        Parameters
        ----------
        model_fname : str
            Path to the saved model file.
        """

        with open(model_fname, 'rb') as file:
            self.model = pickle.load(file)