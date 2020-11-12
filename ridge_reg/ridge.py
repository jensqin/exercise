from datetime import datetime

import scipy
import altair as alt
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.datasets import load_boston
from sklearn.model_selection import KFold
from sklearn import metrics
import peak_engines

X, y = load_boston(return_X_y=True)
result = load_boston()
model = peak_engines.RidgeRegressionModel(normalize=True, grouping_mode="none")

start = datetime.now()
model.fit(X, y)
print(f"{datetime.now()-start} seconds")

print("alpha =", model.regularization_)
for i in model.alpha_:
    print(np.log(i))
np.argsort(model.alpha_)
# groups 0 (x <-100): 8, 7
# 1 (-100 < x < -3): 4, 9, 0, 10, 12
# 2 (-3 < x < 0): 1, 5, 11, 3
# 3 (x > 0): 6, 2
yhat = model.predict(X)
metrics.mean_squared_error(yhat, y)

model = peak_engines.RidgeRegressionModel(
    normalize=True, num_groups=2
)
model.fit(X, y)
print("alpha =", model.regularization_)

grouper1 = np.zeros(13)
grouper1[[4, 9, 0, 10, 12]] = 1
grouper1[[1, 5, 11, 3]] = 2
grouper1[[6, 2]] = 3
grouper1 = grouper1.astype("int").tolist()
model = peak_engines.RidgeRegressionModel(normalize=True, grouper=lambda X, y: grouper1)
model.fit(X, y)
print("alpha =", model.regularization_)

skridge = RidgeCV()
start = datetime.now()
skridge.fit(X, y)
print(f"{datetime.now()-start} seconds")
skyhat = skridge.predict(X)
metrics.mean_squared_error(skyhat, y)

kf = KFold(506)
result = list(kf.split(X))


def benchmark(type, n_fold=8, **kwargs):
    """
    benchmark of models
    """
    if type == "fast_ridge":
        if kwargs:
            model = peak_engines.RidgeRegressionModel(normalize=True, **kwargs)
        else:
            model = peak_engines.RidgeRegressionModel(
                normalize=True, grouping_mode="none"
            )
    elif type == "ridge":
        model = Ridge(normalize=True)
    elif type == "ridgecv":
        model = RidgeCV(normalize=True)
    else:
        raise ValueError("Undefined type.")
    scores = []
    modeltime = []
    for train_index, test_index in result:
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        start = datetime.now()
        model.fit(X_train, y_train)
        time_span = (datetime.now() - start).microseconds
        modeltime.append(time_span)
        yhat = model.predict(X_test)
        scores.append(metrics.mean_squared_error(yhat, y_test))
    return pd.DataFrame({"rmse": scores, "time": modeltime})


fr_df = benchmark("fast_ridge", n_fold=8, score="loocv")
fr_df.describe()  # 33.46049333430216, gcv: 31.64
alt.Chart(fr_df).mark_point().encode(x="rmse", y="time")

fr_df = benchmark("fast_ridge", n_fold=8)
fr_df.describe()  # 2-group: 31.85, 'all': 31.62
alt.Chart(fr_df).mark_point().encode(x="rmse", y="time")
# for multi-group, gcv is worse

for i in range(1, 9):
    fr_df = benchmark("fast_ridge", n_fold=8, num_groups=i)
    print(f"{i} groups: loss {fr_df['rmse'].mean()}")


# given number of groups
for i in range(2, 13):
    fr_df = benchmark("fast_ridge", n_fold=8, num_groups=i)
    print(fr_df["rmse"].mean())
# 2 is best 31.58

fr_df = benchmark("fast_ridge", n_fold=8, grouper=lambda X, y: grouper1)
fr_df.describe()  # 31.16
alt.Chart(fr_df).mark_point().encode(x="rmse", y="time")

r_df = benchmark("ridge", 8)
r_df.describe()  # 36.65282989796395
alt.Chart(r_df).mark_point().encode(x="rmse", y="time")

rcv_df = benchmark("ridgecv", 8)
rcv_df.describe()  # 30.449386520871208
# 30.17
# 3M 1M
