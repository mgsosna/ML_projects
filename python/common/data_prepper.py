import numpy as np


class DataPrepper:
    """
    | Methods common to linear and logistic regression
    """

    def check_matrix_vector_shapes(self,
                                   X: np.ndarray,
                                   y: np.ndarray,
                                   axis: int) -> None:
        """
        | Checks whether number of rows or columns in X matches length of y
        | - Columns : normal equation
        | - Rows : predictions with theta vector
        """
        if X.shape[axis] != len(y):
            raise ValueError(f"Length of axis {axis} in X ({X.shape[axis]}) does not match length " +
                             f"of y ({len(y)})")

    def add_intercept(self,
                      X: np.ndarray) -> np.ndarray:
        """
        | Checks whether first column of inputted data is the intercept. If not,
        | adds a column of intercepts to the beginning of the array
        |
        | ----------------------------------------------------------------------
        | Parameters
        | ----------
        |  X : np.ndarray
        |    Array of predictors
        |
        |
        | Returns
        | -------
        |  np.ndarray
        |    Original array if first column is all 1, else original array plus
        |    column with intercept
        """
        if all(X[:, 0] == 1):
            return X
        else:
            return np.c_[np.ones((X.shape[0], 1)), X]

    # TODO: fix issue with Nans
    def scale(self,
              X: np.ndarray) -> np.ndarray:
        """
        | For each column of X, subtract the mean and divide by the standard deviation
        | of the column
        |
        | ----------------------------------------------------------------------------
        | Parameters
        | ----------
        |  X : np.ndarray
        |    Matrix of features
        |
        |
        | Returns
        | -------
        |  np.ndarray
        |    Original matrix with each column scaled by mean and standard deviation
        """
        feat_means = np.apply_along_axis(np.mean, 0, X)
        feat_stds = np.apply_along_axis(np.std, 0, X)

        return (X - feat_means) / feat_stds
