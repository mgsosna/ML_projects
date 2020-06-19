import numpy as np
from .regression_type import RegressionType
from .data_prepper import DataPrepper


class ModelHelper:
    """
    | Methods common to parameter search in linear and logistic regression
    """
    def __init__(self):
        self.dp = DataPrepper()

    def set_alpha(self,
                  m: int) -> float:
        """
        | Set alpha based on the number of samples.
        |
        | Examples:
        |   10 -> 1e-3
        |   100 -> 1e-5
        |   1000 -> 1e-7
        |
        | ----------------------------------------------------------
        | Parameters
        | ----------
        |  m : int
        |    Number of samples in data
        |
        |
        | Returns
        | -------
        |  float
        """
        return 10 ** -(round(np.log10(m)) * 2 + 1)

    def calculate_cost(self,
                       preds: np.ndarray,
                       actuals: np.ndarray,
                       regression_type: RegressionType) -> np.ndarray:
        """
        | Calculate how far off the predictions are from the actuals. For
        | linear regression, this is just subtracting actuals from predictions.
        | These values are not squared, as the positive/negative sign indicates
        | whether predictions are too high or too low.
        |
        | ---------------------------------------------------------------------
        | Parameters
        | ----------
        |  preds : np.ndarray
        |    Vector of predictions
        |
        |  actuals : np.ndarray
        |    Vector of actual targets
        |
        |
        | Returns
        | -------
        |  np.ndarray
        |    Vector of floats
        """
        if regression_type == RegressionType.LINEAR:
            return preds - actuals
        else:
            -actuals * np.log10(preds) - (1 - actuals) * np.log10(1 - preds)

    def calculate_mse(self,
                      preds: np.ndarray,
                      actuals: np.ndarray,
                      regression_type: RegressionType):
        """
        | Calculate the mean squared error
        |
        | ------------------------------------
        | Parameters
        | ----------
        |  preds : np.ndarray
        |    Predictions
        |
        |  actuals : np.ndarray
        |    Actual values
        |
        |
        | Returns
        | -------
        |  float
        |    Average squared error
        """
        cost = self.calculate_cost(preds, actuals, regression_type)
        return np.mean(cost ** 2)

    def predict(self,
                X: np.ndarray,
                theta: np.ndarray,
                regression_type: RegressionType):
        """
        | Generate predictions based off of predictors (X) and linear regression
        | coefficients (theta)
        |
        | ----------------------------------------------------------------------
        | Parameters
        | ----------
        |  X : np.ndarray
        |    Matrix of predictors. Must include intercept
        |
        |  theta : np.ndarray
        |    Vector of coefficients
        """
        self.dp.check_matrix_vector_shapes(X, theta, axis=1)

        if regression_type == RegressionType.LINEAR:
            return X.dot(theta)
        else:
            return 1 / (1 + np.e ** (-X.dot(theta)))
