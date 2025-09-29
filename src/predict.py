from src.feature.selector import TopKFeatureSelector
from src.model.ensemble import EnsembleModel
from src.utils.config import load_config
from src.data.data_builder import MonashIndexData
from src.data.data_splitter import TimeSeriesSplitter


if __name__ == "__main__":
    config = load_config("config.yml")
    monash_data = MonashIndexData(config=config)
    splitter = TimeSeriesSplitter(config=config)
    
    # Load and split data
    df = monash_data.build()
    _, _, X_test = splitter.split_data(df=df)

    for task in ["classification", "regression"]:
        ensemble_model = EnsembleModel(config=config, task=task)
        model_fname = f"{config['predict']['model_dir']}/{task}/{config['predict']['ensemble_model']}"
        ensemble_model.load(model_fname=model_fname)
        preds = ensemble_model.model.predict(X_test)
        if task == "classification":
            X_test["outperform_binary"] = preds
        else:
            X_test["excess_return"] = preds

    assignment_submission_df = X_test[["stock_id", "month_id", "outperform_binary", "excess_return"]]
    kaggle_submission_df = X_test[["stock_id", "excess_return"]]

    assignment_submission_df.to_csv(f"{config['predict']['submission_dir']}/testing_targets.csv", index=False)
    kaggle_submission_df.to_csv(f"{config['predict']['submission_dir']}/kaggle_submission.csv", index=False)
