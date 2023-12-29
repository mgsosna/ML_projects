import pandas as pd

from .node import Node

class DecisionTree:
    """
    Tree of nodes, with methods for building tree in a way that minimizes
    Gini impurity.
    """
    def __init__(
        self,
        root: Node,
        max_depth: int = 10,
        min_samples_leaf: int = 0
    ) -> None:
        self.root = root
        self.max_depth = None
        self.min_samples_leaf = None

    def build_tree(self) -> None:
        pass

    def process_node(self, node: Node, target_col: str):
        cols = list(node.df)
        cols.remove(target_col)

        # TODO: update
        for col in cols:
            pass


    def split_on_feature(
        self,
        node: Node,
        feature: str
    ) -> tuple[Node, Node]:
        """
        Iterate through values of a feature and identify split that minimizes
        weighted Gini impurity in child nodes.
        """
        values = []

        for thresh in node.df[feature].unique():
            if thresh == node.df[feature].max():
                pass
            values.append(self._process_split(node.df, feature, thresh))

        return min(values, key=lambda x: x[0])

    def _process_split(
        self,
        df: pd.DataFrame,
        feature: str,
        threshold: int|float
    ) -> None | tuple[int|float, Node, Node]:
        """
        Splits df on the feature threshold and generates nodes for the data
        subsets. If
        """
        df_lower = self.df[self.df[feature] <= threshold]
        df_upper = self.df[self.df[feature] > threshold]

        # If partition results in nothing, end early
        if len(df_lower) == 0 or len(df_upper) == 0:
            return None

        node_lower = Node(df_lower)
        node_upper = Node(df_upper)

        prop_lower = len(df_lower) / len(self.df)
        prop_upper = len(df_upper) / len(self.df)

        weighted_gini = node_lower.gini * prop_lower + node_upper.gini * prop_upper

        return weighted_gini, node_lower, node_upper
