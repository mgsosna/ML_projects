import numpy as np
from typing import Union, Optional, List

# Set global variables
DEFAULT_NOISE = 1.0


class DataGenerator:
    """
    | Methods for generating sample data
    """

    def create_linear_data(self,
                           n_obs: int,
                           n_feat: int,
                           intercept: Optional[Union[int, float]] = None,
                           slopes: Optional[List[float]] = None,
                           noise: Union[int, float, list] = DEFAULT_NOISE) -> dict:
        """
        | Generate data for predictors (X) and target (y). In X, each row represents an
        | observation and each column a feature. Features have positive trends with adjustable
        | noise. y is a vector with increasing values with a small amount of noise.
        |
        | ------------------------------------------------------------------------------------
        | Parameters
        | ----------
        |  n_obs : int
        |    Number of observations
        |
        |  n_feat : int
        |    Number of features
        |
        |  intercept : int, float, or None
        |    The intercept. Random value used if none provided
        |
        |  slopes : list or None
        |    Slopes to use for each feature when generating data. Random values used if
        |    none provided
        |
        |  noise : int, float, list
        |    The noise to add to the linear relationship. Higher values correspond to
        |    more noise. If int or float, same amount of noise is applied to all features.
        |    If list, each value corresponds to noise for each feature
        |
        |
        | Returns
        | -------
        |  dict
        |    {'X': predictors with intercept (np.ndarray),
        |     'y': target (np.ndarray),
        |     'intercept': intercept (float),
        |     'slopes': slopes for each feature (list of floats),
        |     'noise': noise for each feature (list of floats)}
        """
        self._create_linear_data_inputs_valid(n_obs, n_feat, intercept, slopes, noise)

        intercept, slopes, noise = self._process_input_params(intercept, slopes, noise, n_feat)

        # Generate X
        X = np.random.uniform(-100, 100, [n_obs, n_feat])
        X = np.c_[np.ones((X.shape[0], 1)), X]   # add intercept

        # Generate noise
        N = np.random.normal(0, noise, [n_obs, n_feat+1])  # each column is different feature's noise

        # Iteratively adjust y and update X
        betas = np.array([intercept] + slopes)
        y = X.dot(betas) + N.sum(axis=1)

        # Transpose y for proper matrix multiplication later
        y = y.reshape(-1, 1)

        return {'X': X, 'y': y, 'betas': betas, 'noise': noise}

    def _create_linear_data_inputs_valid(self,
                                         n_obs: int,
                                         n_feat: int,
                                         intercept: Optional[Union[int, float]],
                                         slopes,
                                         noise) -> None:
        """
        | Confirm inputs to generate_samples are valid. Performs following checks:
        |   * Checks input dtypes
        |      - n_obs and n_feat are int or np.int64
        |      - intercept is int, np.int64, float, or None
        |      - slopes is list or None
        |      - noise is int, np.int64, float, or list
        |   * If slopes is list, number of slopes = number of features
        |   * If noise is list, number of slopes = number of features
        """
        # Check dtypes
        for arg_tup in [('n_obs', n_obs), ('n_feat', n_feat)]:
            if not isinstance(arg_tup[1], (int, np.int64)):
                raise ValueError(f"Arg {arg_tup[0]} is type {type(arg_tup[1])} but must be int")

        if not isinstance(intercept, (int, np.int64, float, type(None))):
            raise ValueError(f"Arg intercept is type {type(intercept)} but must be int, float, or None")

        if not isinstance(slopes, (list, type(None))):
            raise ValueError(f"Arg slopes is type {type(slopes)} but must be list or None")

        if not isinstance(noise, (int, np.int64, float, list)):
            raise ValueError(f"Arg noise is type {type(noise)} but must be int, float or list")

        # Check number of betas and noise = number of features
        if slopes is not None and len(slopes) != n_feat:
            raise ValueError(f"Number of slopes ({len(slopes)}) must match N features ({n_feat})")

        if isinstance(noise, list) and len(noise) != n_feat:
            raise ValueError(f"Length of noise array ({len(noise)}) must match number of features " +
                             f"({n_feat})")

    def _process_input_params(self,
                              intercept: Optional[Union[int, float]],
                              slopes: Optional[List[float]],
                              noise: Union[int, float, list] ,
                              n_feat: int) -> tuple:
        """
        | Generate random parameters if None's passed in, convert to slopes to
        | floats, and convert noise to a list if an int or float passed in.
        |
        | -----------------------------------------------------------------------------
        | Parameters
        | ----------
        |  intercept : int, float, or None
        |    The intercept
        |
        |  slopes : list or None
        |    Slopes to use for each feature when generating data
        |
        |  noise : int, float, list
        |    The noise to add to the linear relationship
        |
        |   n_feat : int
        |    Number of features
        """
        if intercept is None:
            intercept = np.random.uniform(-100, 100, 1)
        else:
            intercept = float(intercept)

        if slopes is None:
            slopes = np.random.uniform(-10, 10, n_feat)
        else:
            slopes = [float(slope) for slope in slopes]

        if isinstance(noise, (int, float)):
            noise = [noise] * (n_feat + 1)

        return intercept, slopes, noise
