import mlpack
from mlpack import decision_tree
from mlpack import nbc
import pandas as pd
import numpy as np


df = pd.read_csv("ecoli.csv", header=None)
df.columns = [
    "sequence", "mcg", "gvh", "lip", "chg", "aac", "alm1", "alm2", "class"
]

df.iloc[:, -1] = df.iloc[:, -1]
y_Train = df.iloc[:, -1]
x_Train = df.drop(["sequence", "class"], axis=1)

output = mlpack.preprocess_split(input=x_Train,
                                 input_labels=y_Train,
                                 test_ratio=0.3)
training_set = output["training"]
training_labels = output["training_labels"]
test_set = output["test"]
test_labels = output["test_labels"]

param_dec_tree = decision_tree(training=training_set,
                                      labels=training_labels,
                                      print_training_accuracy=True,
                                      maximum_depth=5,
                                      minimum_leaf_size=10)
dt_model = param_dec_tree["output_model"]

predict_dt = decision_tree(input_model=dt_model, test=test_set)

correct_dec_tree = np.sum(
    predict_dt["predictions"] == np.reshape(test_labels, (
        test_labels.shape[0], )))
print("Decision tree result: " + str(correct_dec_tree) + " correct out of " + 
      str(len(test_labels)) + " (" + str(100 * float(correct_dec_tree) / float(len(test_labels))) + 
      "%).")
