from typing import Optional
import pickle

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, f1_score
from sklearn.ensemble import VotingClassifier, VotingRegressor

from src.model.base_model import BaseModel
from src.model.linear import LinearModel
from src.model.random_forest import RandomForestModel
from src.model.svm import SVMModel
from src.model.xgboost import XGBModel
from src.model.catboost import CatboostModel
from src.utils.cross_validation import time_series_cv_splits
from src.feature.selector import TopKFeatureSelector


class EnsembleModel(BaseModel):
    """
    Ensemble model combining multiple base learners with feature selection.

    The ensemble uses a pre-trained set of models (Linear, Random Forest, SVM, XGBoost, CatBoost)
    and combines them using a soft voting classifier or voting regressor depending on the task.

    Parameters
    ----------
    config : dict
        Configuration dictionary containing paths, feature importance, ensemble settings, etc.
    task : str
        Task type: either "classification" or "regression".

    Attributes
    ----------
    top_features : pd.DataFrame
        DataFrame containing features and their importance scores, used for top-K selection.
    feature_selector : TopKFeatureSelector
        Transformer selecting the top-K features based on importance scores.
    lr_model : BaseModel
        Pre-trained linear model.
    rf_model : BaseModel
        Pre-trained random forest model.
    svm_model : BaseModel
        Pre-trained SVM model.
    xgb_model : BaseModel
        Pre-trained XGBoost model.
    catboost_model : BaseModel
        Pre-trained CatBoost model.
    estimators : list of tuple[str, BaseEstimator]
        List of estimator name and model pairs for ensemble.
    ensemble_model : VotingClassifier or VotingRegressor
        The ensemble model combining all base learners.
    model : Pipeline
        Full pipeline including preprocessing, feature selection, and ensemble model.
    """

    def __init__(self, config: dict, task: str):
        super().__init__(config, task)

        self.config = config
        self.task = task

        self.top_features = pd.read_csv(f"{config['feature_importance']['importance_result_dir']}/{task}_{config['feature_importance']['importance_result_save_name']}")
        self.feature_selector = TopKFeatureSelector(
            feature_names=self.top_features["feature"].values, 
            importances=self.top_features["importance"].values, 
            k=self.config["ensemble"]["top_k_features"])

        self.lr_model = LinearModel(config=config, task=config["task"])
        self.rf_model = RandomForestModel(config=config, task=config["task"])
        self.svm_model = SVMModel(config=config, task=config["task"])
        self.xgb_model = XGBModel(config=config, task=config["task"])
        self.catboost_model = CatboostModel(config=config, task=config["task"])

        self.lr_model.load(model_fname=f"{config['ensemble']['model_dir']}/{task}/{config['ensemble']['linear_model']}")
        self.rf_model.load(model_fname=f"{config['ensemble']['model_dir']}/{task}/{config['ensemble']['random_forest_model']}")
        self.svm_model.load(model_fname=f"{config['ensemble']['model_dir']}/{task}/{config['ensemble']['svm_model']}")
        self.xgb_model.load(model_fname=f"{config['ensemble']['model_dir']}/{task}/{config['ensemble']['xgboost_model']}")
        self.catboost_model.load(model_fname=f"{config['ensemble']['model_dir']}/{task}/{config['ensemble']['catboost_model']}")

        self.estimators = [
            ("cat", self.catboost_model.model.named_steps["model"]),
            ("xgb", self.xgb_model.model.named_steps["model"]),
            ("rf", self.rf_model.model.named_steps["model"]),
            ("svm", self.svm_model.model.named_steps["model"]),
            ("lr", self.lr_model.model.named_steps["model"])
        ]

        if self.task == "classification":
            self.ensemble_model = VotingClassifier(
                estimators=self.estimators,
                voting="soft",
                n_jobs=-1
            )
        elif self.task == "regression":
            self.ensemble_model = VotingRegressor(
                estimators=self.estimators,
                n_jobs=-1
            )
        
        self.model = Pipeline(steps=[
            ("preprocessor", self.preprocessor),
            ("feature_selector", self.feature_selector),
            ("model", self.ensemble_model)
        ])

    def tune(self):
        pass

