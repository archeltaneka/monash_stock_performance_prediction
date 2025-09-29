import pandas as pd
import numpy as np

from src.utils.config import load_config
from src.data.data_builder import MonashIndexData
from src.data.data_splitter import TimeSeriesSplitter
from src.model.baseline import DummyModel
from src.model.linear import LinearModel
from src.model.random_forest import RandomForestModel
from src.model.svm import SVMModel
from src.model.xgboost import XGBModel
from src.model.catboost import CatboostModel


if __name__ == "__main__":
    config = load_config("config.yml")
    monash_data = MonashIndexData(config=config)
    splitter = TimeSeriesSplitter(config=config)
    
    # Load and split data
    df = monash_data.build()
    X_train, y_train, X_test = splitter.split_data(df=df)
    
    # Train a dummy model as a baseline and models with default hyperparameters
    dummy_model = DummyModel(config=config, task=config["task"])
    lr_model = LinearModel(config=config, task=config["task"])
    rf_model = RandomForestModel(config=config, task=config["task"])
    svm_model = SVMModel(config=config, task=config["task"])
    xgb_model = XGBModel(config=config, task=config["task"])
    catboost_model = CatboostModel(config=config, task=config["task"])
    initial_models = [("baseline", dummy_model), ("logistic_regression", lr_model), ("random_forest", rf_model),
                      ("svm", svm_model), ("xgboost", xgb_model), ("catboost", catboost_model)]

    models = []
    metric_scores = []
    for model_name, model in initial_models:
        metric_score = model.fit(X=X_train, y=y_train, df=df,
                        task=config["task"],
                        n_splits=config["initial_model"]["n_splits"],
                        window_type=config["initial_model"]["window_type"],
                        warmup=config["initial_model"]["warmup"])
        
        metric_scores.append(np.mean(metric_score))
        models.append(model_name)

    result_df = pd.DataFrame({"model": models, "validation_metric_score": metric_scores})
    if config["task"] == "classification":
        result_df = result_df.sort_values(by="validation_metric_score", ascending=False)
    elif config["task"] == "regression":
        result_df = result_df.sort_values(by="validation_metric_score", ascending=True)
        
    result_df.to_csv(f'{config["initial_model"]["comparison_save_dir"]}/{config["task"]}_{config["initial_model"]["comparison_result_save_name"]}', index=False)
    