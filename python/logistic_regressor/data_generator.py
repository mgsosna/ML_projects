import random
import numpy as np
import pandas as pd
from typing import Union

# Defaults for creating a sigmoid from 1 to 50 with an inflection at 25
DEFAULT_X = np.arange(1, 50, 0.1)
DEFAULT_BETA = 0.5
DEFAULT_INTERCEPT = -12.5


class DataGenerator:
    """
    | Methods for generating data for logistic regression
    """

    def generate_outcomes(self,
                          x: np.ndarray = DEFAULT_X,
                          beta: Union[int, float] = DEFAULT_BETA,
                          intercept: Union[int, float] = DEFAULT_INTERCEPT) -> pd.DataFrame:
        """
        | Generate a dataframe with inputted data, simulated probabilities of
        | of y = 1, and simulated outcomes.
        |
        | -------------------------------------------------------
        | Parameters
        | ----------
        |  x : np.ndarray
        |    Values for a 1-dimensional predictor (the independent variable)
        """
        probs = self.get_sigmoid_probs(x, beta, intercept)
        outcomes = [*map(lambda prob: random.choices((1, 0), (prob, 1-prob))[0], probs)]
        return pd.DataFrame({'x': x, 'prob': probs, 'outcome': outcomes})

    def get_sigmoid_probs(self,
                          x: np.ndarray,
                          beta: Union[int, float],
                          intercept: Union[int, float]) -> np.ndarray:
        """
        | Generate probabilities of y = 1 according to a specified sigmoid
        |
        | ----------------------------------------------------------------
        | Parameters
        | ----------
        |  x : np.ndarray
        |    Values for a 1-dimensional predictor (the independent variable)
        |
        |  beta : int, float
        |    The multiplier for x
        |
        |  intercept : int, float
        |    The intercept
        |
        |
        | Returns
        | -------
        |  np.ndarray
        |    Probabilities of y = 1
        """
        probs = 1 / (1 + np.e ** -(intercept + beta * x))
        return probs





