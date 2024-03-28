# %%
import os
import warnings
import numpy as np
import pandas as pd
from joblib import load
from sklearn.metrics import auc, roc_auc_score, accuracy_score

import os

warnings.filterwarnings("ignore")

current_directory = os.path.dirname(__file__)

if not current_directory.endswith("/"):
    current_directory += "/"

inputfilepath = current_directory+"data_valid.txt"
outputfilepath = current_directory

def check_and_fill_cpg(data, selected_cpg):
    for col in selected_cpg:
        if col not in data.columns:
            print(
                f"Column {col} does not exist in the DataFrame. Adding and fill it with 0.5 now."
            )
            data[col] = 0.5
        else:
            print(f"Column {col} exists in the DataFrame.")

    return data

# load data
datafile = pd.read_table(
    inputfilepath,
    index_col=0,
    header=0,
    delimiter="\t",
)

# isna
has_nan = datafile.isna().any().any()
print("Any NaN values present:", has_nan)


if has_nan:
    # NaN20.5
    datafile.fillna(0.5, inplace=True)

    # check again
    has_nan_after_fillna = datafile.isna().any().any()
    if not has_nan_after_fillna:
        print("All NaN values replaced with 0.5")
    else:
        print("Some NaN values still present failing to replace with 0.5")
else:
    print("No NaN values found.")

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

X_valid = data.drop("status", axis=1)
y_valid = data["status"]

loaded_model = load(current_directory+"saved_model.joblib")
selected_features = load(current_directory+"saved_features.joblib")
X_valid = check_and_fill_cpg(X_valid, selected_features)
X_valid_selected = X_valid[selected_features]
predictions = loaded_model.predict(X_valid_selected)

auc = roc_auc_score(y_valid, predictions)
accuracy = accuracy_score(y_valid, predictions)

valid_results = pd.DataFrame({"Index": X_valid.index, "Predicted": predictions, "Actual": y_valid})
valid_results.to_csv(outputfilepath+"prediction_valid.txt",sep="\t",index=False,mode="a")
valid_scores = pd.DataFrame({"auc": [auc], "accuracy": [accuracy]}, index=["valid"])
valid_scores.to_csv(outputfilepath+"performance.txt",mode="a",index=True)
