import logging
from pathlib import Path

import pandas as pd
import numpy as np


class MonashIndexData:
    """
    A data builder class for preparing Monash stock market datasets. 

    This class loads the following primary data:
    1. Historical stock data
    2. Company profile
    3. Index
    4. Target and test data

    Then, optionally merges additional datasets:
    1. Funds rate
    2. Inflation rate
    3. Unemployment rate
    4. 5 year treasury
    5. 10 year treasury
    6. VIX index
    
    On top of that, it imputes missing values, and computes rolling and lagged 
    statistical features for downstream modeling tasks.

    Parameters
    ----------
    config : dict
        A configuration dictionary with the necessary keys 
        (please refer to the config.yml file under `data_builder` key)

    Methods
    -------
    build()
        Executes the end-to-end pipeline:
        - Loads primary and optional data.
        - Joins datasets by `month_id` and `stock_id`.
        - Imputes missing values according to the selected strategy.
        - Computes rolling means, standard deviations, medians, lagged features, 
          volatility, volume, price range, and index-based rolling features.
        - Handles missing values in generated rolling features.
        - Returns the processed DataFrame.

    Notes
    -----
    - ID columns (`month_id`, `stock_id`) are never imputed.
    - Rolling features are computed within each `stock_id` group to maintain time-series integrity.
    - `next_month_df` is appended to allow predictions for the upcoming month.
    """


    def __init__(self, config: dict):
        """
        Initialize the monash index data builder with a configuration.

        Parameters
        ----------
        config : dict
            Configuration dictionary containing the file names and preprocessing methods.
        """

        self.config = config["data_builder"]
        self.data_dir = Path(self.config["data_dir"])
        self.use_optional_data = self.config["use_optional_data"]

    def _load_data(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, list | None]:
        """
        Load raw datasets from CSV files.

        Returns
        -------
        stock_df : pd.DataFrame
            Stock-level data (returns, prices, etc.).
        company_df : pd.DataFrame
            Company profile metadata.
        index_df : pd.DataFrame
            Index data for each month.
        train_targets_df : pd.DataFrame
            Target labels for training (per stock and month).
        optional_data_list : list of pd.DataFrame or None
            List of optional datasets if `use_optional_data` is True, otherwise None.
        """

        # Primary data
        stock_df = pd.read_csv(self.data_dir/f"{self.config['stock_data']}")
        company_df = pd.read_csv(self.data_dir/f"{self.config['company_info']}")        
        index_df = pd.read_csv(self.data_dir/f"{self.config['monash_index']}")
        train_targets_df = pd.read_csv(self.data_dir/f"{self.config['train_targets']}")

        optional_data_list = None
        # Optional data
        if self.use_optional_data:
            optional_data_dir = self.data_dir/"optional_data"
            optional_data_list = []
            for fname in optional_data_dir.glob("*.csv"):
                optional_data_list.append(pd.read_csv(fname))

        return stock_df, company_df, index_df, train_targets_df, optional_data_list

    def _join_data(
        self,
        stock_df: pd.DataFrame,
        company_df: pd.DataFrame,
        index_df: pd.DataFrame,
        train_targets_df: pd.DataFrame,
        optional_data_list: list | None
    ) -> pd.DataFrame:
        """
        Merge stock, company, index, and target datasets into a unified DataFrame.

        Parameters
        ----------
        stock_df : pd.DataFrame
            Stock-level data.
        company_df : pd.DataFrame
            Company profile metadata.
        index_df : pd.DataFrame
            Index data.
        train_targets_df : pd.DataFrame
            Target labels.
        optional_data_list : list of pd.DataFrame or None
            List of optional datasets to merge.

        Returns
        -------
        df : pd.DataFrame
            Unified dataset containing all merged information.
        """

        next_month_df = pd.DataFrame({"month_id": ["2023_07"]*len(company_df), "stock_id": company_df["stock_id"].values})
        stock_df = pd.concat([stock_df, next_month_df])
        df = pd.merge(stock_df, company_df, on=["stock_id"], how="left")
        df = pd.merge(df, index_df, on=["month_id"], how="left")
        df = pd.merge(df, train_targets_df, on=["month_id", "stock_id"], how="left")

        if optional_data_list:
            for optional_data in optional_data_list:
                df = pd.merge(df, optional_data, on=["month_id"], how="left")

        return df

    def _impute_missing_columns(self, df: pd.DataFrame, method: str) -> pd.DataFrame:
        """
        Impute missing values in numeric columns (excluding ID columns).

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame.
        method : {"ffill", "median"}
            Strategy for imputation:
            - "ffill" : forward-fill missing values.
            - "median" : fill missing values with column median.

        Returns
        -------
        pd.DataFrame
            DataFrame with imputed values.
        """

        for column in df.columns:
            if column in ["month_id", "stock_id"]:  # never impute ID columns
                continue
            if method == "ffill":
                df[column] = df[column].fillna(method="ffill")
            elif method == "median":
                df[column] = df[column].fillna(df[column].median())
        
        return df

    def _compute_rolling_statistic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute rolling, lagged, volatility, volume, price range, and index-based features.

        Features are calculated within each `stock_id` group to preserve time-series structure.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame containing raw and imputed features.

        Returns
        -------
        pd.DataFrame
            DataFrame with additional engineered features.
        """

        grouped = df.groupby('stock_id')

        # Price-based window returns
        for window in [3, 6, 12]:
            df[f'intramonth_return_rolling_mean_{window}m'] = grouped['intramonth_return'].transform(lambda x: x.rolling(window, min_periods=1).mean())
            df[f'intramonth_return_rolling_std_{window}m'] = grouped['intramonth_return'].transform(lambda x: x.rolling(window, min_periods=1).std())
            df[f'intramonth_return_rolling_median_{window}m'] = grouped['intramonth_return'].transform(lambda x: x.rolling(window, min_periods=1).median())

        # Lagged returns
        for col in ['return_1m', 'return_3m', 'return_6m']:
            for lag in [1, 2]:
                df[f'{col}_lagged_{lag}'] = grouped[col].shift(lag)

        # Volatility rolling features
        for col, windows in [
            ('intramonth_volatility', [3, 6, 12]),
            ('volatility_3m', [3, 6]),
            ('volatility_6m', [3, 6])
        ]:
            for window in windows:
                df[f'{col}_rolling_mean_{window}m'] = grouped[col].transform(lambda x: x.rolling(window, min_periods=1).mean())
                df[f'{col}_rolling_std_{window}m'] = grouped[col].transform(lambda x: x.rolling(window, min_periods=1).std())

        # Volume rolling features
        for col, windows in [
            ('monthly_volume', [3, 6]),
            ('avg_volume_3m', [3, 6]),
            ('volume_ratio', [3, 6])
        ]:
            for window in windows:
                df[f'{col}_rolling_mean_{window}m'] = grouped[col].transform(lambda x: x.rolling(window, min_periods=1).mean())
                df[f'{col}_rolling_sum_{window}m'] = grouped[col].transform(lambda x: x.rolling(window, min_periods=1).sum())

        # Price rolling range
        df['price_range'] = df['month_high_usd'] - df['month_low_usd']
        for window in [3, 6, 12]:
            df[f'price_range_rolling_mean_{window}m'] = grouped['price_range'].transform(lambda x: x.rolling(window, min_periods=1).mean())
            df[f'price_range_ratio_rolling_mean_{window}m'] = grouped['price_range_ratio'].transform(lambda x: x.rolling(window, min_periods=1).mean())

        # Index rolling features
        for window in [3, 6]:
            df[f'index_return_rolling_mean_{window}m'] = grouped['index_return'].transform(lambda x: x.rolling(window, min_periods=1).mean())
            df[f'index_return_rolling_std_{window}m'] = grouped['index_return'].transform(lambda x: x.rolling(window, min_periods=1).std())

        return df

    def _handle_missing_rolling_features(
        self,
        df: pd.DataFrame,
        strategy: str = "drop",
        fill_value: float | int = 0
    ) -> pd.DataFrame:

        """
        Handle missing values specifically for lagged and rolling features.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame with rolling features.
        strategy : {"drop", "fill", "ffill", "bfill"}, default="drop"
            Strategy to handle missing values:
            - "drop" : drop rows with missing rolling features.
            - "fill" : fill missing values with a constant (fill_value).
            - "ffill" : forward-fill missing values.
            - "bfill" : backward-fill missing values.
        fill_value : float or int, default=0
            Value to use when `strategy="fill"`.

        Returns
        -------
        pd.DataFrame
            DataFrame with handled missing rolling features.
        """

        rolling_cols = [c for c in df.columns if "lagged" in c or "rolling" in c]
        if strategy == "drop":
            df = df.dropna(subset=rolling_cols)
        elif strategy == "fill":
            df[rolling_cols] = df[rolling_cols].fillna(fill_value)
        elif strategy == "ffill":
            df[rolling_cols] = df[rolling_cols].fillna(method="ffill")
        elif strategy == "bfill":
            df[rolling_cols] = df[rolling_cols].fillna(method="bfill")

        return df

    
    def build(self) -> pd.DataFrame:
        """
        Execute the complete data preparation pipeline.

        1. Load stock, company, index, target, and optional datasets.
        2. Join datasets into a unified DataFrame.
        3. Impute missing values in raw features.
        4. Compute rolling, lagged, volatility, volume, price range,
           and index-based statistical features.
        5. Handle missing values in rolling features.

        Returns
        -------
        pd.DataFrame
            Final processed dataset ready for training or modeling.
        """

        stock_df, company_df, index_df, train_targets_df, optional_data_list = self._load_data()
        df = self._join_data(stock_df, company_df, index_df, train_targets_df, optional_data_list)

        df = self._impute_missing_columns(df, method=self.config["imputation_strategy"])
        df = df.sort_values(by=["stock_id", "month_id"])
        
        df = self._compute_rolling_statistic_features(df)
        df = self._handle_missing_rolling_features(df, strategy=self.config["build_additional_features"]["handle_missing_rolling_features_strategy"])

        return df