import warnings

import pandas as pd
from joblib import load
from sklearn.metrics import accuracy_score, roc_auc_score

warnings.filterwarnings("ignore")


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


datafile = pd.read_table(
    "/data/LSY/prostate_cancer/data_valid.txt", header=0, index_col=0, delimiter="\t"
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
        print("Some NaN values still present after attempting to replace with 0.")
else:
    print("No NaN values found.")


# CpGs with a standard deviation of less than 1% across samples were removed prior to analysis.
column_std = datafile.iloc[:, :-1].std()
std_percentile = column_std.quantile(0.01)
low_std_cpgs = column_std[column_std < std_percentile].index
filtered_data = datafile.drop(columns=low_std_cpgs)


# set X and y
X = filtered_data.iloc[:, :-1]
selected_features = [
    "cg00054525",
    "cg16794576",
    "cg24581650",
    "cg15338327",
    "cg00054525",
    "cg14781281",
]
X = check_and_fill_cpg(X, selected_features)
X_sel6 = X.loc[:, selected_features]
y = filtered_data.iloc[:, -1]

# apply model
loaded_model = load("/data/LSY/prostate_cancer/model6CpGs.joblib")
y_pred_sel6 = loaded_model.predict(X_sel6)
threshold = 0.5
y_binary_pred_sel6 = (y_pred_sel6 >= threshold).astype(int)

results = pd.DataFrame(
    {"Index": X_sel6.index, "Predicted": y_binary_pred_sel6, "Actual": y}
)

auc = roc_auc_score(y, y_pred_sel6)
accuracy = accuracy_score(y, y_binary_pred_sel6)

results.to_csv(
    "/data/LSY/prostate_cancer/performance_valid.txt",
    sep="\t",
    index=False,
    mode="a",
)

with open("/data/LSY/prostate_cancer/performance_valid.txt", "a") as file:
    file.write(f"\nAUC: {auc}")
    file.write(f"\nAccuracy: {accuracy}")
