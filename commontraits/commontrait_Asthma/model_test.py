# %%
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import load
from sklearn.linear_model import LassoCV
from sklearn.metrics import auc, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# %%
def check_and_fill_cpg(data, selected_cpg):
    for col in selected_cpg:
        if col not in data.columns:
            print(
                f"Column {col} does not exist in the DataFrame. Adding and fill it now (FYI: fill with 0.5)."
            )
            data[col] = 0.5
        else:
            print(f"Column {col} exists in the DataFrame.")

    return data


# %%
import warnings

import pandas as pd
from joblib import dump
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")


# load data
datafile = pd.read_table(
    "/data/LSY/z_preparing_and_parts/commontrait_Asthma/data_test.txt",
    index_col=0,
    header=0,
    delimiter="\t",
)


# isna
has_nan = datafile.isna().any().any()
print("Any NaN values present:", has_nan)


if has_nan:
    # NaN20
    datafile.fillna(0, inplace=True)

    # check again
    has_nan_after_fillna = datafile.isna().any().any()
    if not has_nan_after_fillna:
        print("All NaN values replaced with 0.")
    else:
        print("Some NaN values still present failing to replace with 0.")
else:
    print("No NaN values found.")


# %%
# Sample Exclusions: Samples excluded if ≥1% of CpGs had a detection p-value exceeding 0.05
outlier_threshold2 = datafile.drop("status", axis=1).mean() + 1.96 * datafile.drop(
    "status", axis=1
).std() / np.sqrt(len(datafile))
columns_to_drop = []
for cpg_column in datafile.columns.drop("status"):
    num_outliers = (datafile[cpg_column] > outlier_threshold2[cpg_column]).sum()
    if num_outliers > 0.01 * datafile.shape[1]:
        columns_to_drop.append(cpg_column)
data = datafile.drop(columns=columns_to_drop)

X_test = data.drop("status", axis=1)
y_test = data["status"]


# %%
loaded_model = load("/data/LSY/z_preparing_and_parts/commontrait_Asthma/model.joblib")
selected_features = load(
    "/data/LSY/z_preparing_and_parts/commontrait_Asthma/model_selected_features.joblib"
)

X_test = check_and_fill_cpg(X_test, selected_features)
X_test_selected = X_test[selected_features]

predictions = loaded_model.predict(X_test_selected)


# %%

auc = roc_auc_score(y_test, predictions)
accuracy = accuracy_score(y_test, predictions)


# %%

test_results = pd.DataFrame(
    {"Index": X_test.index, "Predicted": predictions, "Actual": y_test}
)

test_results.to_csv(
    "/data/LSY/z_preparing_and_parts/commontrait_Asthma/prediction_test.txt",
    sep="\t",
    index=False,
    mode="a",
)

test_scores = pd.DataFrame({"auc": [auc], "accuracy": [accuracy]}, index=["test"])

test_scores.to_csv(
    "/data/LSY/z_preparing_and_parts/commontrait_Asthma/performance.txt",
    mode="a",
    index=True,
)
