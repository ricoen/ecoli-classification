import mlpack
from mlpack import decision_tree
import pandas as pd
import numpy as np

df = pd.read_csv("ecoli.csv", header=None)
df.columns = [
    "sequence", "mcg", "gvh", "lip", "chg", "aac", "alm1", "alm2", "class"
]

df.iloc[:, -1] = df.iloc[:, -1].apply(pd.to_numeric)
y_Train = df.iloc[:, -1]
x_Train = df.drop(["sequence", "class"], axis=1)

output = mlpack.preprocess_split(input=x_Train,
                                 input_labels=y_Train,
                                 test_ratio=0.3)
training_set = output["training"]
training_labels = output["training_labels"]
test_set = output["test"]
test_labels = output["test_labels"]

output = mlpack.decision_tree(training=training_set,
                              labels=training_labels,
                              print_training_accuracy=True,
                              maximum_depth=5,
                              minimum_leaf_size=10)
decision_tree = output["output_model"]

output = mlpack.decision_tree(input_model=decision_tree, test=test_set)

correct = np.sum(
    output["predictions"] == np.reshape(test_labels, (test_labels.shape[0], )))
print(str(correct) + " correct out of " + str(len(test_labels)) + " (" + 
    str(100 * float(correct) / float(len(test_labels))) + "%).")
