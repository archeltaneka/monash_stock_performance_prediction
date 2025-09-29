from src.utils.config import load_config
from src.data.data_builder import MonashIndexData
from src.data.data_splitter import TimeSeriesSplitter
from src.feature.importance import FeatureImportance


if __name__ == "__main__":
    config = load_config("config.yml")
    monash_data = MonashIndexData(config=config)
    splitter = TimeSeriesSplitter(config=config)
    
    # Load and split data
    df = monash_data.build()
    X_train, y_train, X_test = splitter.split_data(df=df)

    feature_imp = FeatureImportance(config)
    feature_imp.find_importance(df=df, X=X_train, y=y_train)