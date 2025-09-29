import pandas as pd
import numpy as np

from src.utils.config import load_config
from src.data.data_builder import MonashIndexData
from src.data.data_splitter import TimeSeriesSplitter
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
    
    # Tune the hyperparameters for each model
    lr_model = LinearModel(config=config, task=config["task"])
    rf_model = RandomForestModel(config=config, task=config["task"])
    svm_model = SVMModel(config=config, task=config["task"])
    xgb_model = XGBModel(config=config, task=config["task"])
    catboost_model = CatboostModel(config=config, task=config["task"])
    initial_models = [
        ("linear_regression", lr_model), ("random_forest", rf_model), ("svm", svm_model),
        ("xgb", xgb_model), ("catboost", catboost_model)
    ]

    models = []
    metric_scores = []
    hyperparams = []
    for model_name, model in initial_models:
        best_model, best_hyperparams, best_score = model.tune(
            df=df, 
            X=X_train, 
            y=y_train, 
            warmup=config["cross_validation"]["warmup"], 
            n_splits=config["cross_validation"]["n_splits"], 
            window_type=config["cross_validation"]["window_type"]
        )
        
        models.append(model_name)
        hyperparams.append(best_hyperparams)
        metric_scores.append(best_score)

        save_fname = f"{config['tuning']['model_save_dir']}/{config['task']}/tuned_{model_name}_model.pkl"
        model.save(save_fname)

    result_df = pd.DataFrame({"model": models, "best_hyperparameters": hyperparams, "validation_metric_score": metric_scores})
    if config["task"] == "classification":
        result_df = result_df.sort_values(by="validation_metric_score", ascending=False)
    elif config["task"] == "regression":
        result_df = result_df.sort_values(by="validation_metric_score", ascending=True)
        
    result_df.to_csv(f'{config["tuning"]["comparison_save_dir"]}/{config["task"]}_{config["tuning"]["comparison_result_save_name"]}', index=False)
    
