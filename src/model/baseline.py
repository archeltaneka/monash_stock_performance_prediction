from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyClassifier, DummyRegressor

from src.model.base_model import BaseModel


class DummyModel(BaseModel):
    """
    A simple baseline model that predicts dummy values.

    For classification, predicts randomly with uniform probability.
    For regression, predicts the median value of the training target.

    Attributes
    ----------
    config : dict
        Configuration dictionary containing at least "random_seed".
    model : sklearn.pipeline.Pipeline
        Pipeline containing the dummy estimator.

    Methods
    -------
    tune()
        Placeholder method; no tuning is required for dummy models.
    """

    def __init__(self, config: dict, task: str):
        """
        Initialize the DummyModel.

        Parameters
        ----------
        config : dict
            Configuration dictionary. Must include "random_seed" for reproducibility.
        task : str
            Task type: "classification" or "regression".

        Returns
        -------
        None
        """

        self.config = config

        if task == "classification":
            self.model = Pipeline(steps=[
                ("model", DummyClassifier(strategy="uniform", random_state=self.config["random_seed"])) 
            ])
        elif task == "regression":
            self.model = Pipeline(steps=[
                ("model", DummyRegressor(strategy="median")) 
            ])

    def tune(self):
        """
        Dummy models do not require hyperparameter tuning.

        Returns
        -------
        None
        """
        