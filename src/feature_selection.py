from src.data.data_builder import MonashIndexData
from src.data.data_splitter import TimeSeriesSplitter
from src.feature.importance import FeatureImportance
from src.feature.selector import ForwardSelector
from src.model.catboost import CatboostModel
from src.utils.config import load_config
from src.utils.cross_validation import time_series_cv_splits


if __name__ == "__main__":
    config = load_config("config.yml")
    monash_data = MonashIndexData(config=config)
    splitter = TimeSeriesSplitter(config=config)
    
    # Load and split data
    df = monash_data.build()
    X_train, y_train, X_test = splitter.split_data(df=df)

    # Perform forward feature selection
    selector = ForwardSelector(
        config=config, 
        task=config["task"]
    )
    cv_splits = list(time_series_cv_splits(
        df, 
        warmup=config["feature_selection"]["warmup"], 
        n_splits=config["feature_selection"]["n_splits"], 
        window_type=config["feature_selection"]["window_type"]
    ))
    results = selector.fit(
        df=df,
        X=X_train, 
        y=y_train, 
        cv_splits=cv_splits, 
        step=config["feature_selection"]["step"]
    )
    selector.plot_feature_selection_results(results=results)
    