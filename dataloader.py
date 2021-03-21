"""
Load data.
"""

import pandas as pd

DATA_DIR = "digit-recognizer"

# %%

train = pd.read_csv(DATA_DIR + "/train.csv")
test = pd.read_csv(DATA_DIR + "/test.csv")
Y_train = train["label"]
X_train = train.drop(labels=["label"], axis=1)

# free some space
del train
