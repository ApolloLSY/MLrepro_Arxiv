import warnings

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.linear_model import LassoCV, LogisticRegression
from sklearn.metrics import accuracy_score, auc, roc_auc_score

warnings.filterwarnings("ignore")


# load data
datafile = pd.read_table(
    "/data/LSY/z_preparing_and_parts/commontrait_Asthma/data_train.txt",
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


# Sample Exclusions: Samples excluded if ≥1% of CpGs had a detection p-value exceeding 0.05
outlier_threshold2 = datafile.iloc[:, :-1].mean() + 1.96 * datafile.iloc[
    :, :-1
].std() / np.sqrt(len(datafile))
columns_to_drop = []
for cpg_column in datafile.columns[:-1]:
    num_outliers = (datafile[cpg_column] > outlier_threshold2[cpg_column]).sum()
    if num_outliers > 0.01 * datafile.shape[1]:
        columns_to_drop.append(cpg_column)
data = datafile.drop(columns=columns_to_drop)

X_train = data.drop("status", axis=1)
y_train = data["status"]

# lasso to find cpgs
Lasso_model = LassoCV(alphas=[0.01], cv=5, random_state=42)
Lasso_model.fit(X_train, y_train)

selected_features = X_train.columns[Lasso_model.coef_ != 0]

X_train_selected = X_train[selected_features]

# use cpgs in Logistic
logistic_model = LogisticRegression()
logistic_model.fit(X_train_selected, y_train)

predictions = logistic_model.predict(X_train_selected)


auc = roc_auc_score(y_train, predictions)
accuracy = accuracy_score(y_train, predictions)


train_results = pd.DataFrame(
    {"Index": X_train.index, "Predicted": predictions, "Actual": y_train}
)

train_results.to_csv(
    "/data/LSY/z_preparing_and_parts/commontrait_Asthma/prediction_train.txt",
    sep="\t",
    index=False,
    mode="a",
)

train_scores = pd.DataFrame({"auc": [auc], "accuracy": [accuracy]}, index=["train"])

train_scores.to_csv(
    "/data/LSY/z_preparing_and_parts/commontrait_Asthma/performance.txt",
    mode="a",
    index=True,
)


# save model
dump(logistic_model, "/data/LSY/z_preparing_and_parts/commontrait_Asthma/model.joblib")
dump(
    selected_features,
    "/data/LSY/z_preparing_and_parts/commontrait_Asthma/model_selected_features.joblib",
)
