import numpy as np
import pandas as pd

class Node:
    """
    Node in a decision tree. Assumes target_col in df consists of
    0s and 1s.
    """
    def __init__(
        self,
        df: pd.DataFrame,
        target_col: str
    ) -> None:
        self.df = self._check_df(df, target_col)
        self.pk = self.set_pk(target_col)
        self.gini = self.set_gini()
        self.left = None
        self.right = None

    def _check_df(
        self,
        df: pd.DataFrame,
        target_col: str
    ) -> pd.DataFrame:
        assert len(list(df)) > 1, \
            "df must have features"
        assert not set(df[target_col]).difference({0,1}), \
            "target column cannot have values besides {0,1}"
        return df

    def set_pk(self, target_col: str) -> float:
        """
        Sets pk, the proportion of samples that are of the positive class.
        Assumes samples is a list of ints, where 1 is the positive class
        and 0 is the negative class.
        """
        return np.mean(self.df[target_col].values)

    def set_gini(self) -> float:
        """
        Sets the Gini impurity.
        """
        return 1 - self.pk**2 - (1 - self.pk)**2
