from functools import partial
import numpy as np
import pandas as pd

from classes import DecisionTree, Node

# Set seed for reproducibility
np.random.seed(42)

# Set labels
labels = np.random.choice([0,1], 100)

# Generate data
npn = partial(np.random.normal, scale=1, size=1)

df = pd.DataFrame({
    'feature_1': [npn(2)[0] if x else npn(0)[0] for x in labels],
    'feature_2': [npn(1)[0] if x else npn(0)[0] for x in labels],
    'feature_3': [npn(0.5)[0] if x else npn(0)[0] for x in labels],
    'label': labels
})

# Create node and decision tree
node = Node(df, target_col='label')
decision_tree = DecisionTree(node)

print(node)
print(decision_tree)
