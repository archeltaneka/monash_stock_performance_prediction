from typing import Tuple

import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV

from scipy.stats import randint

from src.model.base_model import BaseModel
from src.utils.cross_validation import time_series_cv_splits


class RandomForestModel(BaseModel):
    """
    Random Forest model wrapper for classification and regression tasks.

    Parameters
    ----------
    config : dict
        Configuration dictionary containing model hyperparameters, tuning settings, and random seed.
    task : str
        Task type: "classification" or "regression".

    Attributes
    ----------
    model : Pipeline
        Pipeline containing preprocessing and Random Forest model.

    Methods
    -------
    tune(df: pd.DataFrame, X: pd.DataFrame, y: pd.Series, warmup: int, n_splits: int, window_type: str) -> Tuple[Pipeline, dict, float]
        Performs randomized hyperparameter tuning using time-series cross-validation.
        Returns best estimator, best parameters, and best cross-validation score.
    """

    def __init__(self, config: dict, task: str):
        super().__init__(config, task)

        if task == "classification":
            self.model = Pipeline(steps=[
                ("preprocessor", self.preprocessor),
                ("model", RandomForestClassifier(
                    n_jobs=-1, 
                    random_state=config["random_seed"])) 
            ])
        elif task == "regression":
            self.model = Pipeline(steps=[
                ("preprocessor", self.preprocessor),
                ("model", RandomForestRegressor(
                    n_jobs=-1, 
                    random_state=config["random_seed"])) 
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
        Perform randomized hyperparameter tuning using time-series cross-validation.

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
            Best cross-validation score (F1 for classification, RMSE for regression).
        """
        
        # Hyperparameter search space
        param_dist = {
            "model__n_estimators": randint(
                self.config["tuning"]["random_forest"]["n_estimators"][0], 
                self.config["tuning"]["random_forest"]["n_estimators"][1]),    
            "model__max_depth": self.config["tuning"]["random_forest"]["max_depth"],   
            "model__min_samples_split": randint(
                self.config["tuning"]["random_forest"]["min_samples_split"][0],
                self.config["tuning"]["random_forest"]["min_samples_split"][1]
            ),  
            "model__min_samples_leaf": randint(
                self.config["tuning"]["random_forest"]["min_samples_leaf"][0],
                self.config["tuning"]["random_forest"]["min_samples_split"][1]
            ),   
            "model__max_features": self.config["tuning"]["random_forest"]["max_features"]
        }

        # Generate CV folds the your custom splitter
        cv_splits = list(time_series_cv_splits(
            df, 
            warmup=warmup, 
            n_splits=n_splits, 
            window_type="expanding"
        ))

        # Randomized search
        scoring = "f1" if self.config["task"] == "classification" else "neg_mean_squared_error"
        search = RandomizedSearchCV(
            self.model,
            param_distributions=param_dist,
            n_iter=30,
            cv=cv_splits,
            scoring="f1" if self.config["task"] == "classification" else "neg_mean_squared_error",
            n_jobs=-1,
            verbose=2,
            random_state=self.config["random_seed"]
        )

        search.fit(X, y)

        print("Best Params:", search.best_params_)
        if self.config["task"] == "classification":
            print("Best CV F1 Score:", search.best_score_)
            return search.best_estimator_, search.best_params_, search.best_score_
        elif self.config["task"] == "regression":
            best_score = (-search.best_score_ )**0.5
            print("Best CV RMSE Score:", best_score)
            return search.best_estimator_, search.best_params_, best_score
