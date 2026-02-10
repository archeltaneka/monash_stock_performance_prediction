import pandas as pd
import numpy as np


class TimeSeriesSplitter:
    """
    A utility class for splitting time-series data into training and test sets.

    The splitter uses `month_id` to separate training and test data according 
    to configuration settings. It supports both classification and regression 
    tasks by selecting the appropriate target column.

    Parameters
    ----------
    config : dict
        Configuration dictionary.

    Attributes
    ----------
    config : dict
        Stores the configuration for splitting.
    task : str
        Task type, "classification" or "regression".

    Methods
    -------
    split_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]
        Splits the input DataFrame into training features, training labels, 
        and test features.
    """

    def __init__(self, config: dict):
        """
        Initialize the time-series splitter with a configuration.

        Parameters
        ----------
        config : dict
            Configuration dictionary containing task type and split parameters.
        """

        self.config = config
        self.task = config["task"]

    def split_data(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
        """
        Split a DataFrame into training and test sets based on month_id.

        Parameters
        ----------
        df : pd.DataFrame
            Input dataset containing at least the following columns:
            - "month_id" : str or int
                Time identifier used for splitting.
            - "outperform_binary" : int, optional
                Binary classification target, required if task="classification".
            - "excess_return" : float, optional
                Continuous regression target, required if task="regression".

        Returns
        -------
        tuple
            - X_train : pd.DataFrame
                Training feature matrix with target columns removed.
            - y_train : pd.Series
                Training target values, int (outperform_binary) for classification 
                or float (excess_return) for regression.
            - X_test : pd.DataFrame
                Test feature matrix with target columns removed.
        """

        train_df = df[df['month_id'] <= str(self.config["data_splitter"]["last_train_date"])]
        test_df = df[df['month_id'] == str(self.config["data_splitter"]["test_date"])]

        X_train = train_df.drop(columns=["outperform_binary", "excess_return"], axis=1)
        X_test = test_df.drop(columns=["outperform_binary", "excess_return"], axis=1)

        if self.task == "classification":
            y_train = train_df["outperform_binary"].astype(int)
        elif self.task == "regression":
            y_train = train_df["excess_return"].astype(float)

        return X_train, y_train, X_test
        