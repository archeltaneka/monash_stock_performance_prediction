from typing import Iterator, Tuple
import pandas as pd


def time_series_cv_splits(
    df: pd.DataFrame, 
    warmup: int = 12, 
    n_splits: int = 5, 
    window_type: str = "expanding"
) -> Iterator[Tuple[pd.Index, pd.Index]]:
    """
    Generate train/validation index splits for time-series cross-validation.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing at least a 'month_id' column for temporal ordering.
    warmup : int, default=12
        Minimum number of months to include in the initial training window.
    n_splits : int, default=5
        Number of validation folds to generate.
    window_type : str, default="expanding"
        Type of rolling window for training data. 
        - "expanding": training window starts at first month and expands.
        - "sliding": training window slides forward keeping size equal to `warmup`.

    Yields
    ------
    train_idx : pd.Index
        Indexes of the training set for the current split.
    val_idx : pd.Index
        Indexes of the validation set for the current split.
    """
    
    months = sorted(df["month_id"].unique())
    total_months = len(months)

    for i in range(warmup, min(total_months, warmup + n_splits)):
        if window_type == "expanding":
            train_months = months[:i]       # start to i-1
        elif window_type == "sliding":
            start_idx = max(0, i - warmup)
            train_months = months[start_idx:i]  # last `warmup` months
        else:
            raise ValueError(f"Invalid window_type: {window_type}. Must be 'expanding' or 'sliding'.")

        val_month = months[i]
        train_idx = df[df["month_id"].isin(train_months)].index
        val_idx   = df[df["month_id"] == val_month].index

        yield train_idx, val_idx