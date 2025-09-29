from typing import Tuple

import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.linear_model import ElasticNet, LogisticRegression
from sklearn.model_selection import GridSearchCV

from src.model.base_model import BaseModel
from src.utils.cross_validation import time_series_cv_splits


class LinearModel(BaseModel):
    """
    Linear model wrapper for classification and regression tasks.

    Parameters
    ----------
    config : dict
        Configuration dictionary containing model hyperparameters, tuning settings, and random seed.
    task : str
        Task type: "classification" or "regression".

    Attributes
    ----------
    model : Pipeline
        Pipeline containing preprocessing and linear model.

    Methods
    -------
    tune(df: pd.DataFrame, X: pd.DataFrame, y: pd.Series, warmup: int, n_splits: int, window_type: str) -> Tuple[Pipeline, dict, float]
        Performs grid search hyperparameter tuning for the linear model.
        Returns best estimator, best parameters, and best cross-validation score.
    """

    def __init__(self, config: dict, task: str):
        super().__init__(config, task)

        if task == "classification":
            self.model = Pipeline(steps=[
                ("preprocessor", self.preprocessor),
                ("model", LogisticRegression(
                    max_iter=config["initial_model"]["linear"]["max_iter"], 
                    random_state=config["random_seed"]
                ))
            ])
        elif task == "regression":
            self.model = Pipeline(steps=[
                ("preprocessor", self.preprocessor),
                ("model", ElasticNet())
            ])

    def tune(
        self,
        df: pd.DataFrame,
        X: pd.DataFrame,
        y: pd.Series,
        warmup: int,
        n_splits: int,
        window_type: str
    ) -> Tuple[Pipeline, dict, float]:
        """
        Perform grid search hyperparameter tuning using time-series cross-validation.

        Parameters
        ----------
        df : pd.DataFrame
            Full dataset used for generating CV folds.
        X : pd.DataFrame
            Feature matrix.
        y : pd.Series
            Target vector.
        warmup : int
            Number of warmup periods for expanding window cross-validation.
        n_splits : int
            Number of CV splits.
        window_type : str
            Type of rolling window, e.g., "expanding" or "sliding".

        Returns
        -------
        best_estimator : Pipeline
            Pipeline fitted with the best hyperparameters.
        best_params : dict
            Best hyperparameters found during tuning.
        best_score : float
            Best cross-validation score (F1 for classification, negative MSE for regression).
        """

        if self.task == "classification":
            # Build pipeline with logistic regression
            pipe = Pipeline(steps=[
                ("preprocessor", self.preprocessor),
                ("model", LogisticRegression(
                    max_iter=self.config["initial_model"]["linear"]["max_iter"], 
                    random_state=self.config["random_seed"]
                ))
            ])

            # Hyperparameter search space
            param_grid = {
                "model__penalty": self.config["tuning"]["linear_regression"]["classification"]["penalty"],
                "model__C": self.config["tuning"]["linear_regression"]["classification"]["C"],
                "model__solver": self.config["tuning"]["linear_regression"]["classification"]["solver"]
            }

            # Generate CV folds
            cv_splits = list(time_series_cv_splits(df, warmup=warmup, n_splits=n_splits, window_type=window_type))

            # Grid search
            grid_search = GridSearchCV(
                pipe,
                param_grid,
                cv=cv_splits,
                scoring="f1",
                n_jobs=-1,
                verbose=2
            )

            grid_search.fit(X, y)

            print("Best Params:", grid_search.best_params_)
            print("Best CV F1 Score:", grid_search.best_score_)
        
        elif self.task == "regression":
            # Build pipeline with Ridge regression
            pipe = Pipeline(steps=[
                ("preprocessor", self.preprocessor),
                ("model", ElasticNet(random_state=self.config["random_seed"]))
            ])

            # Hyperparameter search space
            param_grid = {
                "model__alpha": self.config["tuning"]["linear_regression"]["regression"]["alpha"],
                "model__l1_ratio": self.config["tuning"]["linear_regression"]["regression"]["l1_ratio"]
            }

            # CV folds
            cv_splits = list(time_series_cv_splits(df, warmup=warmup, n_splits=n_splits, window_type=window_type))

            # Grid search
            grid_search = GridSearchCV(
                pipe,
                param_grid,
                cv=cv_splits,
                scoring="neg_mean_squared_error",  # or "r2"
                n_jobs=-1,
                verbose=2
            )

            grid_search.fit(X, y)

            print("Best Params:", grid_search.best_params_)
            print("Best CV RMSE:", (-grid_search.best_score_)**0.5)

        return grid_search.best_estimator_, grid_search.best_params_, grid_search.best_score_