import pandas as pd

from .decision_tree import DecisionTree

class RandomForest:
    """
    Forest of decision trees
    """
    def __init__(
        self,
        df: pd.DataFrame,
        target_col: str,
        n_trees: int = 100,
        max_depth: int = 4
    ) -> None:
        self.df = df
        self.target_col = target_col
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.forest = []

    def train(self) -> None:
        bootstrap_dfs = [self._boostrap(self.df) for _ in range(n_trees)]
        self.forest = [DecisionTree(df, self.target_col) for df in bootstrap_dfs]

        # Need a way to specify the subset of columns
        pass

    def classify(self) -> int:
        pass

    def _bootstrap(self) -> pd.DataFrame:
        """
        Sample rows from self.df with replacement
        """
        return self.df.sample(len(self.df), replace=True)
