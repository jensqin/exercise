import os

import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_squared_error

os.chdir("..")

npy_dir = "data/output"
arrays = {
    filename.strip(".npy"): np.load(f"{npy_dir}/{filename}")
    for filename in os.listdir(npy_dir)
    if filename.endswith(".npy")
}
df = pd.DataFrame(arrays)

df["true"] = np.load("data/output/true.npy")
df.describe()
# sns.jointplot("resmlr_ex9_set", "mlr_setbag", data=df, kind="reg")
# sns.jointplot("resmlr_ex9_set", "resmlr", data=df, kind="reg")
df.apply(lambda x: mean_squared_error(x, df["true"])).sort_values()
sns.pairplot(df)
sns.jointplot("mlr", "ridge", data=df, kind="kde")

# 1000 samples, 100 batches, each batch has 10 samples
# each step, we take one batch, calculate the gradient, then do SGD
# each epoch, we take all 100 batches from training data

# learning rate: 0.001, alpha1, alpha2, alpha3, etc.

# tmpp = ((df["ridgemlr_test"] > tmp["y_exp"]) & (tmp["y_exp"] > tmp["y"])) | (
#     (df["ridgemlr_test"] < tmp["y_exp"]) & (tmp["y_exp"] < tmp["y"])
# )


def mse_weighted_average(p):
    est = p * df["ridge"] + (1 - p) * df["mlr"]
    print(p, mean_squared_error(est, df["true"]))


for t in np.linspace(0, 1, 10):
    mse_weighted_average(t)
# 1.355155
