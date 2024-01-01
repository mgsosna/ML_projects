import numpy as np
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
        self.stack = [root]
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf

    def build_tree(self) -> None:
        features = list(self.root.df)
        features.remove(self.root.target_col)

        current_node = self.root

        while self.stack:
            node = self.stack.pop()

            if node:

                # TODO: figure out what to do if already visited
                node_left, node_right = self.process_node(node, features)
                current_node.left = node_left
                current_node.right = node_right

                self.stack.append(node_left)
                self.stack.append(node_right)

        # TODO: finish. For now just processes the root and first children.
        #node_left, node_right = self.process_node(self.root, features)

        return node_left, node_right

    # TODO: need support for getting to leaf node
    def process_node(self, node: Node|None, features: list[str]) -> tuple[Node|None, Node|None]:
        """
        Iterates through features, identifies split that minimizes
        Gini impurity in child nodes, and identifies feature whose
        split minimizes Gini impurity the most. Then returns child
        nodes split on that feature.
        """
        # Don't process if we're at a leaf
        if not node:
            return None, None

        # Get Gini impurity for best split for each column
        d = {}
        for col in features:
            d[col] = self.split_on_feature(node, col)

        # Select best column to split on
        min_gini = np.inf
        best_feature = None
        for col, tup in d.items():
            if tup[0] < min_gini:
                min_gini = tup[0]
                best_feature = col

        if min_gini is np.inf or best_feature is None:
            raise ValueError("Splitting node was unsuccessful.")

        # Only update if the best split reduces Gini impurity
        if min_gini < node.gini:
            return d[col][1:]

        return None, None

    def split_on_feature(
        self,
        node: Node,
        feature: str
    ) -> tuple[float, Node, Node]:
        """
        Iterate through values of a feature and identify split that minimizes
        weighted Gini impurity in child nodes.
        """
        values = []

        for thresh in node.df[feature].unique():
            if thresh == node.df[feature].max():
                pass
            values.append(self._process_split(node.df, feature, thresh))

        # TODO: add handling for empty list
        values = [v for v in values if v is not None]
        return min(values, key=lambda x: x[0])

    def _process_split(
        self,
        df: pd.DataFrame,
        feature: str,
        threshold: int|float
    ) -> None | tuple[float, Node, Node]:
        """
        Splits df on the feature threshold and generates nodes for the data
        subsets. If
        """
        df_lower = df[df[feature] <= threshold]
        df_upper = df[df[feature] > threshold]

        # If partition results in nothing, end early
        if len(df_lower) == 0 or len(df_upper) == 0:
            return None

        node_lower = Node(df_lower, self.root.target_col)
        node_upper = Node(df_upper, self.root.target_col)

        prop_lower = len(df_lower) / len(df)
        prop_upper = len(df_upper) / len(df)

        weighted_gini = node_lower.gini * prop_lower + node_upper.gini * prop_upper

        return weighted_gini, node_lower, node_upper
