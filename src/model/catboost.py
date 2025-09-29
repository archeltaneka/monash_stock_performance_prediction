from typing import Tuple, Union

import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV
from catboost import CatBoostClassifier, CatBoostRegressor

from scipy.stats import randint, loguniform, uniform

from src.model.base_model import BaseModel
from src.utils.cross_validation import time_series_cv_splits


class CatboostModel(BaseModel):
    """
    CatBoost model wrapper for classification or regression tasks.

    This class uses CatBoostClassifier for classification tasks and 
    CatBoostRegressor for regression tasks. It supports hyperparameter 
    tuning using RandomizedSearchCV with time series cross-validation.

    Attributes
    ----------
    config : dict
        Configuration dictionary including random seed and tuning settings.
    model : sklearn.pipeline.Pipeline
        Pipeline containing a preprocessor and a CatBoost estimator.

    Methods
    -------
    tune(df, X, y, warmup, n_splits, window_type)
        Performs randomized hyperparameter search with time-series cross-validation.
    """

    def __init__(self, config: dict, task: str):
        """
        Initialize CatboostModel with preprocessing and CatBoost estimator.

        Parameters
        ----------
        config : dict
            Configuration dictionary. Must contain "random_seed".
        task : str
            Task type: "classification" or "regression".

        Returns
        -------
        None
        """

        super().__init__(config, task)

        if task == "classification":
            self.model = Pipeline(steps=[
                ("preprocessor", self.preprocessor),
                ("model", CatBoostClassifier(random_state=config["random_seed"], verbose=0)) 
            ])
        elif task == "regression":
            self.model = Pipeline(steps=[
                ("preprocessor", self.preprocessor),
                ("model", CatBoostRegressor(random_state=config["random_seed"], verbose=0)) 
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
        Tune hyperparameters using RandomizedSearchCV with time-series splits.

        Parameters
        ----------
        df : pd.DataFrame
            Full dataframe including features and target, used for CV split creation.
        X : pd.DataFrame
            Feature matrix for training.
        y : pd.Series
            Target values.
        warmup : int
            Number of initial periods to skip before CV starts.
        n_splits : int
            Number of CV splits.
        window_type : str
            Type of rolling window for CV ("expanding" or "sliding").

        Returns
        -------
        Tuple[Pipeline, dict, float]
            - best_estimator_: Pipeline with best hyperparameters.
            - best_params_: Dictionary of best hyperparameters found.
            - best_score_: Best CV metric: F1 score for classification, RMSE for regression.
        """
        
        param_dist = {
                "model__iterations": randint(200, 1000),
                "model__depth": randint(4, 10),
                "model__learning_rate": loguniform(0.01, 0.3),
                "model__l2_leaf_reg": loguniform(1, 10),
                "model__bagging_temperature": uniform(0, 5),
        }

        cv_splits = list(time_series_cv_splits(
            df, 
            warmup=warmup, 
            n_splits=n_splits, 
            window_type="expanding"
        ))

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

