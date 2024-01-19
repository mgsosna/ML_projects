import pandas as pd

from .decision_tree import DecisionTree

class RandomForest:
    """
    Forest of decision trees
    """
    def __init__(
        self,
        df: pd.DataFrame,
        n_trees: int = 100
    ) -> None:
        self.n_trees = n_trees

    def train(self) -> None:
        pass

    def classify(self) -> int:
        pass

    def _bootstrap(self) -> pd.DataFrame:
        """
        Sample rows from self.df with replacement
        """
        return self.df.sample(len(self.df), replace=True)
