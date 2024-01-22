from functools import partial
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

from classes import DecisionTree, Node, RandomForest

# Set seed for reproducibility
np.random.seed(42)

# Data generation helpers. Only parameter to pass in is the mean.
npn = partial(np.random.normal, scale=1, size=1)
npc = partial(np.random.choice, a=[0,1], size=1)

# Use this function for a detailed look at decision tree formation
def gen_df(n: int) -> pd.DataFrame:
    labels = np.random.choice([0,1], n)
    return pd.DataFrame({
        'strong_continuous': [npn(3)[0] if x else npn(0)[0] for x in labels],
        'weak_continuous': [npn(1)[0] if x else npn(0)[0] for x in labels],
        'strong_categorical': [
            npc(p=[0.8, 0.2])[0] if x else npc(p=[0.5,0.5])[0]
            for x in labels
        ],
        'label': labels
    })

# TODO: write a function for high-dimensional data
# Use this function for a random forest using many features
#def gen_df_hd(n: int) -> pd.DataFrame:
#    labels = np.random.choice([0,1], n)
#

# Generate data
print("Generating train and test data")
train_df = gen_df(500)
test_df = gen_df(100)

# 1. Decision Tree
print("1. Fitting a decision tree")
decision_tree = DecisionTree(train_df, target_col='label')
decision_tree.build_tree()
tree_preds = decision_tree.classify(test_df)
tree_accuracy = round(accuracy_score(test_df['label'], tree_preds), 3)

# 2. Random Forest
print("2. Fitting a random forest")
forest = RandomForest(train_df, target_col='label', n_trees=20)
forest.train()
forest_preds = forest.classify(test_df)
forest_tree_preds = forest.forest[1].classify(test_df)

forest_accuracy = round(accuracy_score(test_df['label'], forest_preds), 3)
forest_tree_accuracy = round(accuracy_score(test_df['label'], forest_tree_preds), 3)

# Display results
print("Accuracy")
print(f" * Single decision tree: {tree_accuracy}")
print(f" * Tree in random forest: {forest_tree_accuracy}")
print(f" * Full random forest: {forest_accuracy}")
