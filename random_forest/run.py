from functools import partial
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

from classes import DecisionTree, Node

# Set seed for reproducibility
np.random.seed(42)

# Data generation helpers. Only parameter to pass in is the mean.
npn = partial(np.random.normal, scale=1, size=1)
npc = partial(np.random.choice, a=[0,1], size=1)

def gen_df(n: int) -> pd.DataFrame:
    labels = np.random.choice([0,1], n)
    return pd.DataFrame({
        'strong_continuous': [npn(2)[0] if x else npn(0)[0] for x in labels],
        'weak_continuous': [npn(0.5)[0] if x else npn(0)[0] for x in labels],
        'strong_categorical': [
            npc(p=[0.8, 0.2])[0] if x else npc(p=[0.5,0.5])[0]
            for x in labels
        ],
        'label': labels
    })

# Create node and decision tree
train_df = gen_df(500)
test_df = gen_df(100)

decision_tree = DecisionTree(train_df, target_col='label')
decision_tree.build_tree(verbose=True)

# Generate predictions
preds = decision_tree.classify(test_df)

# Display results
print(f"Tree info: {vars(decision_tree.root)}")
print(f"Predictions: {preds}")

print(decision_tree.root.pk)
print(decision_tree.root.left.pk)
print(decision_tree.root.right.pk)

# Calculate accuracy
print(f"Accuracy: {round(accuracy_score(test_df['label'], preds), 2)}")
