import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from src.model.catboost import CatboostModel


class FeatureImportance:
    """
    Compute and save feature importances for a CatBoost model using SHAP values.

    This class handles:
    - Training a CatBoost model with specified hyperparameters,
    - Computing SHAP feature importances,
    - Saving a summary plot of feature importances,
    - Exporting ordered feature importance values into a CSV file.

    Parameters
    ----------
    config : dict
        Configuration dictionary containing feature importance, model, and 
        output settings.

    Attributes
    ----------
    config : dict
        Configuration dictionary passed at initialization.
    model : CatboostModel
        Wrapper for CatBoost pipeline (includes preprocessing and model).
    """

    def __init__(self, config: dict):
        """
        Initialize the FeatureImportance class.

        Parameters
        ----------
        config : dict
            Configuration dictionary with feature importance parameters.
        """

        self.config = config
        self.model = CatboostModel(config=config, task=config["task"])

    def find_importance(
        self,
        df: pd.DataFrame,
        X: pd.DataFrame,
        y: pd.Series
    ) -> None:
        """
        Train the CatBoost model, compute SHAP values, save plots and feature importances.

        Parameters
        ----------
        df : pd.DataFrame
            Full input dataset, required for preprocessing transformations.
        X : pd.DataFrame
            Feature matrix for training.
        y : pd.Series
            Target vector for training.

        Returns
        -------
        None
            Saves:
            - A SHAP summary plot as an image file,
            - A CSV file containing ordered feature importances.
        """

        # Fit the Catboost model using the best hyperparameters
        params = {
                "iterations": self.config["feature_importance"]["iterations"],
                "depth": self.config["feature_importance"]["depth"],
                "learning_rate": self.config["feature_importance"]["learning_rate"],
                "l2_leaf_reg": self.config["feature_importance"]["l2_leaf_reg"],
                "bagging_temperature": self.config["feature_importance"]["bagging_temperature"]
        }
        self.model.model.named_steps["model"].set_params(**params)
        _ = self.model.fit(
            X=X, y=y, df=df,
            task=self.config["task"],
            n_splits=self.config["feature_importance"]["n_splits"],
            window_type=self.config["feature_importance"]["window_type"],
            warmup=self.config["feature_importance"]["warmup"]
        )

        # Use the tuned model to compute SHAP values and plot the feature importance
        explainer_cat = shap.TreeExplainer(self.model.model.named_steps["model"])
        X_train_processed = self.model.model.named_steps["preprocessor"].transform(df)
        shap_values_cat = explainer_cat.shap_values(X_train_processed)
        shap.summary_plot(
            shap_values_cat,
            X_train_processed,
            feature_names=self.model.model.named_steps["preprocessor"].get_feature_names_out(),
            show=False 
        )
        plt.tight_layout()
        plt.savefig(f'{self.config["feature_importance"]["plot_save_dir"]}/{self.config["task"]}_{self.config["feature_importance"]["plot_save_name"]}', dpi=300)
        plt.close()

        # Save the feature importance order into a dataframe
        importances = self.model.model.named_steps["model"].get_feature_importance()
        feature_names = self.model.model.named_steps["preprocessor"].get_feature_names_out()
        importance_df = pd.DataFrame({
            "feature": feature_names,
            "importance": importances
        })
        importance_df = importance_df.sort_values(by="importance", ascending=False).reset_index(drop=True)
        save_fname = f"{self.config['feature_importance']['importance_result_dir']}/{self.config['task']}_{self.config['feature_importance']['importance_result_save_name']}"
        importance_df.to_csv(save_fname, index=False)
