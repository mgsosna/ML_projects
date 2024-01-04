import numpy as np
import pandas as pd

class Node:
    """
    Node in a decision tree.

    Parameters
    ----------
    df : pd.DataFrame
      The dataframe (or subset) this node holds. Used for training.
      All columns except target_col are assumed to be features.
    target_col : str
      The column in the dataframe with labels. Must be 0s and 1s, with
      1s being the positive class.
    pk : float
      Proportion of node's df that contain the positive class.
    gini : float
      The node's Gini impurity.
    left : Node
      The left child of the node. None if no child.
    right : Node
      The right child of the node. None if no child.
    feature : str
      The column in the df whose splitting led to the largest reduction
      in weighted Gini impurity in the child nodes.
    threshold : float | int
      The value of the feature column to split the df.
    """
    def __init__(
        self,
        df: pd.DataFrame,
        target_col: str
    ) -> None:
        # For training
        self.df = self._check_df(df, target_col)
        self.target_col = target_col
        self.pk = self._set_pk()
        self.gini = self._set_gini()

        # For training/inference
        self.left = None
        self.right = None

        # For inference
        self.feature = None
        self.threshold = None

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

    def _set_pk(self) -> float:
        """
        Sets pk, the proportion of samples that are of the positive class.
        Assumes samples is a list of ints, where 1 is the positive class
        and 0 is the negative class.
        """
        return np.mean(self.df[self.target_col].values)

    def _set_gini(self) -> float:
        """
        Sets the Gini impurity.
        """
        return 1 - self.pk**2 - (1 - self.pk)**2
