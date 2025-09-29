from typing import List, Tuple, Optional

from catboost import CatBoostClassifier, CatBoostRegressor
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import mean_squared_error, f1_score
from sklearn.pipeline import Pipeline

from src.model.catboost import CatboostModel


class TopKFeatureSelector(BaseEstimator, TransformerMixin):
    """
    Select the top-K features based on pre-computed importance scores.

    Parameters
    ----------
    feature_names : list[str]
        List of feature names corresponding to columns in the dataset.
    importances : np.ndarray
        Feature importance scores, used to rank features.
    k : int, default=20
        Number of top features to select.

    Attributes
    ----------
    selected_idx : Optional[np.ndarray]
        Indices of the selected top-K features.
    """

    def __init__(self, feature_names: List[str], importances: np.ndarray, k: int = 20):
        self.feature_names = feature_names
        self.importances = importances
        self.k = k
        self.selected_idx = None

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "TopKFeatureSelector":
        """
        Determine the indices of the top-K features based on importance scores.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix (unused but required for scikit-learn compatibility).
        y : np.ndarray, optional
            Target vector (unused).

        Returns
        -------
        self : TopKFeatureSelector
        """

        ranking = np.argsort(self.importances)[::-1]
        self.selected_idx = ranking[:self.k]
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Select only the top-K features from input array.

        Parameters
        ----------
        X : np.ndarray
            Input feature matrix.

        Returns
        -------
        X_transformed : np.ndarray
            Array with only the top-K features selected.
        """

        return X[:, self.selected_idx]

    def get_feature_names_out(self, input_features: Optional[List[str]] = None) -> List[str]:
        """
        Return the names of the selected top-K features.

        Parameters
        ----------
        input_features : list[str], optional
            Ignored, present for compatibility with sklearn API.

        Returns
        -------
        selected_features : list[str]
            Names of selected features.
        """
        
        return [self.feature_names[i] for i in self.selected_idx]


class ForwardSelector:
    """
    Perform forward feature selection using a base model and cross-validation.

    Iteratively evaluates top-k features, fitting a fresh model for each k and computing 
    performance metrics (F1 for classification, RMSE for regression).

    Parameters
    ----------
    config : dict
        Configuration dictionary containing model, feature importance, and plotting settings.
    task : str
        Task type, either "classification" or "regression".

    Attributes
    ----------
    saved_model : CatboostModel
        Pre-trained CatBoost model loaded from disk for initial feature importance.

    Methods
    -------
    fit(df: pd.DataFrame, X: pd.DataFrame, y: pd.Series, cv_splits: list[tuple[int, int]], step: int = 5) -> list[tuple[int, float]]
        Performs forward selection and returns a list of (number_of_features, mean_metric) tuples.
    plot_feature_selection_results(results: list[tuple[int, float]]) -> None
        Plots the performance metric versus number of features and saves the figure.
    """

    def __init__(self, config: dict, task: str):
        self.config = config
        self.task = task
        self.saved_model = CatboostModel(config=config, task=task)
        self.saved_model.load(f"{config['tuning']['model_save_dir']}/{task}/tuned_catboost_model.pkl")        

    def fit(
        self,
        df: pd.DataFrame,
        X: pd.DataFrame,
        y: pd.Series,
        cv_splits: list[tuple[int, int]],
        step: int = 5
    ) -> list[tuple[int, float]]:
        """
        Perform forward feature selection using cross-validation.

        Parameters
        ----------
        df : pd.DataFrame
            Full dataset, required for preprocessing within the pipeline.
        X : pd.DataFrame
            Feature matrix used for model training.
        y : pd.Series
            Target vector.
        cv_splits : list[tuple[int, int]]
            List of (train_idx, val_idx) tuples for cross-validation splits.
        step : int, default=5
            Step size for the number of features to evaluate at each iteration.

        Returns
        -------
        results : list[tuple[int, float]]
            Each tuple contains (number_of_features, mean_metric), where mean_metric
            is either mean F1 score (classification) or mean RMSE (regression).
        """

        results = []

        # Fit the model to obtain feature importances
        params = self.saved_model.model.named_steps["model"].get_params()
        base_model = CatboostModel(config=self.config, task=self.task)
        base_model.model.named_steps["model"].set_params(**params)
        _ = base_model.fit(
            X=X, y=y, df=df,
            task=self.config["task"],
            n_splits=self.config["feature_importance"]["n_splits"],
            window_type=self.config["feature_importance"]["window_type"],
            warmup=self.config["feature_importance"]["warmup"]
        )

        # Get feature importances
        importances = base_model.model.named_steps["model"].get_feature_importance()
        feature_names = base_model.model.named_steps["preprocessor"].get_feature_names_out()
        sorted_idx = np.argsort(importances)[::-1]

        for k in range(step, len(feature_names) + 1, step):
            print(f"Training with {k} number of features...")
            selected_idx = sorted_idx[:k]

            metric_scores = []
            for train_idx, val_idx in cv_splits:
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

                # Build fresh pipeline each time
                preprocessor = base_model.model.named_steps["preprocessor"]
                cat_params = base_model.model.named_steps["model"].get_params()

                if self.task == "classification":
                    fs_pipeline = Pipeline(steps=[
                        ("preprocessor", preprocessor),
                        ("selector", TopKFeatureSelector(feature_names, importances, k=k)),
                        ("model", CatBoostClassifier(**cat_params))
                    ])
                elif self.task == "regression":
                    fs_pipeline = Pipeline(steps=[
                        ("preprocessor", preprocessor),
                        ("selector", TopKFeatureSelector(feature_names, importances, k=k)),
                        ("model", CatBoostRegressor(**cat_params))
                    ])

                fs_pipeline.fit(X_train, y_train)
                y_pred = fs_pipeline.predict(X_val)
                if self.task == "classification":
                    metric_scores.append(f1_score(y_val, y_pred))
                elif self.task == "regression":
                    metric_scores.append(np.sqrt(mean_squared_error(y_val, y_pred)))

            results.append((k, np.mean(metric_scores)))

        return results

    def plot_feature_selection_results(self, results: list[tuple[int, float]]) -> None:
        """
        Plot the forward selection results and save the figure.

        Parameters
        ----------
        results : list[tuple[int, float]]
            Output of `fit()` method, containing (number_of_features, mean_metric) pairs.

        Returns
        -------
        None
            Saves a plot to the configured directory.
        """
        
        ks, f1s = zip(*results)
        plt.figure(figsize=(8,5))
        plt.plot(ks, f1s, marker="o")
        plt.xlabel("Number of features")
        if self.task == "classification":
            plt.ylabel("Mean F1 Score")
        elif self.task == "regression":
            plt.ylabel("Mean RMSE")
        plt.title("Feature Selection Evaluation")
        
        plt.savefig(f'{self.config["feature_selection"]["plot_save_dir"]}/{self.config["task"]}_{self.config["feature_selection"]["plot_save_name"]}', dpi=300)
        plt.close()