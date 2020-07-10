import logging
import numpy as np
from typing import Optional, Tuple

from python.common import DataPrepper, ModelHelper, RegressionType


class LogisticRegressor:
    """
    | Find logistic regression coefficients via gradient descent
    """
    def __init__(self):
        self.dc = DataPrepper()
        self.mh = ModelHelper()

    def gradient_descent(self,
                         X: np.ndarray,
                         y: np.ndarray,
                         alpha: Optional[float] = None,
                         max_iter: int = 10000,
                         min_diff_mse: float = 1e-5) -> Tuple[np.ndarray, list]:
        """
        | Find logistic regression coefficients via gradient descent. Descent occurs
        | for n_iter or until the improvement in MSE falls below stop_thresh.
        |
        | -------------------------------------------------------------------------
        | Parameters
        | ----------
        |  X : np.ndarray
        |    Predictors
        |
        |  y : np.ndarray
        |    Target
        |
        |  alpha : float
        |    The learning rate
        |
        |  max_iter : int
        |    The max number of iterations to run gradient descent, if stop_thresh isn't
        |    reached first
        |
        |  min_diff_mse : float
        |    The improvement in MSE between iterations below which gradient descent stops
        |
        |  scale : bool
        |    Whether to scale the predictors. If True, the value of each feature is subtracted
        |    by the feature mean, then divided by the feature standard deviation
        |
        |
        | Returns
        | -------
        |  np.ndarray
        |    The best fit for linear regression coefficients
        """
        X = self.dc.add_intercept(X)

        # Initialize coefficients at zero
        theta = np.zeros((X.shape[1], 1))

        if alpha is None:
            alpha = self.mh.set_alpha(len(y))

        mse = []

        # Perform descent
        for i in range(max_iter):
            preds = self.mh.predict(X, theta, RegressionType.LOGISTIC)
            theta = self._update_theta(theta, preds, y, X, alpha)
            mse.append(self.mh.calculate_mse(preds, y, RegressionType.LOGISTIC))

            # Stop if change in MSE below threshold
            if i > 1 and (mse[i-1] - mse[i]) < min_diff_mse:
                break

            # If MSE is increasing, stop
            if mse[i] > mse[i-1]:
                logging.error("MSE is increasing; a smaller alpha should be used.")
                return theta, mse

        return theta, mse

    def _update_theta(self,
                      theta: np.ndarray,
                      preds: np.ndarray,
                      y: np.ndarray,
                      X: np.ndarray,
                      alpha: float) -> np.ndarray:
        """
        | Update theta via gradient descent. Derivative of squared errors is
        | multiplied by the learning rate and added to vector of coefficients
        |
        | -------------------------------------------------------------------
        | Parameters
        | ----------
        |  theta : np.ndarray
        |    Vector of coefficients
        |
        |  preds : np.ndarray
        |    Vector of predictions (h(x), or y-hat)
        |
        |  y : np.ndarray
        |    Vector of target variable
        |
        |  X : np.ndarray
        |    Matrix of features
        |
        |  alpha : float
        |    Learning rate
        |
        |
        | Returns
        | -------
        |  np.ndarray
        |    Updated vector of coefficients
        """
        return theta - (1/len(y)) * alpha * (X.T.dot(preds - y))
