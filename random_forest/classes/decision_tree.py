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
        max_depth: int = 2,
        min_samples_leaf: int = 0
    ) -> None:
        self.root = root
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf

    def predict(self, features: pd.Series) -> int:
        """
        Given a vector of features, traverses the tree
        to generate a predicted label.
        """
        return self.root.classify(features)

    def build_tree(self, verbose: bool = False) -> None:
        """
        Builds tree using depth-first traversal. If verbose,
        prints the node depths as the tree is being built.
        """
        features = list(self.root.df)
        features.remove(self.root.target_col)

        stack = [(self.root, 0)]

        while stack:
            current_node, depth = stack.pop()

            if depth <= self.max_depth:
                left, right = self._process_node(current_node, features)

                if left and right:
                    current_node.left = left
                    current_node.right = right
                    stack.append((left, depth+1))
                    stack.append((right, depth+1))

                if verbose:
                    print(depth)

        return None

    def _process_node(
        self,
        node: Node|None,
        features: list[str]
    ) -> tuple[Node|None, Node|None]:
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
            values.append(self._process_split(node, feature, thresh))

        values = [v for v in values if v is not None]
        return min(values, key=lambda x: x[0])

    def _process_split(
        self,
        node: Node,
        feature: str,
        threshold: int|float
    ) -> tuple[float, Node|None, Node|None]:
        """
        Splits df on the feature threshold and generates nodes for the data
        subsets.
        """
        df_lower = node.df[node.df[feature] <= threshold]
        df_upper = node.df[node.df[feature] > threshold]

        # If threshold doesn't split the data at all, end early
        if len(df_lower) == 0 or len(df_upper) == 0:
            return node.gini, None, None

        node_lower = Node(df_lower, self.root.target_col)
        node_upper = Node(df_upper, self.root.target_col)

        prop_lower = len(df_lower) / len(node.df)
        prop_upper = len(df_upper) / len(node.df)

        weighted_gini = node_lower.gini * prop_lower + node_upper.gini * prop_upper

        return weighted_gini, node_lower, node_upper
