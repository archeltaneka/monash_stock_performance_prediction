from typing import List

from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.compose import ColumnTransformer


def create_feature_preprocessor(
    categorical_features: List[str], 
    ordinal_features: List[str], 
    numerical_features: List[str]
) -> ColumnTransformer:
    """
    Create a preprocessing pipeline for categorical, ordinal, and numerical features.

    This function builds a `ColumnTransformer` that applies:
    - OneHotEncoder to categorical features
    - OrdinalEncoder to ordinal features
    - StandardScaler to numerical features

    Parameters
    ----------
    categorical_features : (List[str])
        List of feature names to be encoded using OneHotEncoder.
    ordinal_features : (List[str])
        List of feature names to be encoded using OrdinalEncoder.
    numerical_features : (List[str]) 
        List of feature names to be scaled using StandardScaler.

    Returns
    -------
    preprocessor : ColumnTransformer
        A scikit-learn `ColumnTransformer` that applies preprocessing to the 
        specified features.
    """

    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_features),
            ("ordinal", OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), ordinal_features),
            ("num", StandardScaler(), numerical_features)
        ],
        remainder="drop"
    )

    return preprocessor