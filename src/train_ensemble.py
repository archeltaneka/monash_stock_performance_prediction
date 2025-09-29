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
    X_train, y_train, X_test = splitter.split_data(df=df)

    # Fit an ensemble data from all 5 saved models
    ensemble_model = EnsembleModel(config=config, task=config["task"])
    ensemble_model.fit(
        X=X_train, y=y_train, df=df,
        task=config["task"],
        n_splits=config["ensemble"]["n_splits"],
        window_type=config["ensemble"]["window_type"],
        warmup=config["ensemble"]["warmup"]
    )

    ensemble_model.save(model_fname=f"{config['ensemble']['model_dir']}/{config['task']}/{config['ensemble']['ensemble_model']}")
