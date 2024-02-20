import warnings

import pandas as pd
from joblib import dump
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")


# load data
datafile = pd.read_table(
    "/data/LSY/prostate_cancer/data_train_and_test.txt",
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


# CpGs with a standard deviation of less than 1% across samples were removed prior to analysis.
column_std = datafile.iloc[:, :-1].std()
std_percentile = column_std.quantile(0.01)
low_std_cpgs = column_std[column_std < std_percentile].index

filtered_data = datafile.drop(columns=low_std_cpgs)


# split train and test data
train_data, test_data = train_test_split(filtered_data, test_size=0.2, random_state=42)

X_train = train_data.iloc[:, :-1]
y_train = train_data.iloc[:, -1]

X_test = test_data.iloc[:, :-1]
y_test = test_data.iloc[:, -1]


"""
# all CpGs model 
# AUC:0.8214285714285714
model_allfeatures = LinearRegression()
model_allfeatures.fit(X_train, y_train)

y_pred = model_allfeatures.predict(X_test)

threshold=0.5
y_binary_pred = (y_pred >= threshold).astype(int)

accuracy = accuracy_score(y_test, y_binary_pred)
print(f"Accuracy: {accuracy}")
"""

# 6 CpGs model 'cg00054525', 'cg16794576', 'cg24581650', 'cg15338327', 'cg00054525', 'cg14781281'
# AUC: 0.9747373
selected_features = [
    "cg00054525",
    "cg16794576",
    "cg24581650",
    "cg15338327",
    "cg00054525",
    "cg14781281",
]

X_train_sel6 = train_data.loc[:, selected_features]
X_test_sel6 = test_data.loc[:, selected_features]

model_6features = LinearRegression()
model_6features.fit(X_train_sel6, y_train)

y_test_pred_sel6 = model_6features.predict(X_test_sel6)
y_train_pred_sel6 = model_6features.predict(X_train_sel6)

threshold = 0.5
y_binary_test_pred_sel6 = (y_test_pred_sel6 >= threshold).astype(int)
y_binary_train_pred_sel6 = (y_train_pred_sel6 >= threshold).astype(int)

# save model
dump(
    model_6features,
    "/data/LSY/prostate_cancer/model6CpGs.joblib",
)

# how to use:
# loaded_model = load('./model.joblib')


# output of train and test performance
train_results = pd.DataFrame(
    {
        "Index": X_train_sel6.index,
        "Predicted": y_binary_train_pred_sel6,
        "Actual": y_train,
    }
)

auc = roc_auc_score(y_train, y_binary_train_pred_sel6)
accuracy = accuracy_score(y_train, y_binary_train_pred_sel6)

train_results.to_csv(
    "/data/LSY/prostate_cancer/performance_train.txt",
    sep="\t",
    index=False,
    mode="a",
)

with open("/data/LSY/prostate_cancer/performance_train.txt", "a") as file:
    file.write(f"\nAUC: {auc}")
    file.write(f"\nAccuracy: {accuracy}")


test_results = pd.DataFrame(
    {
        "Index": X_test_sel6.index,
        "Predicted": y_binary_test_pred_sel6,
        "Actual": y_test,
    }
)

auc = roc_auc_score(y_test, y_binary_test_pred_sel6)
accuracy = accuracy_score(y_test, y_binary_test_pred_sel6)

test_results.to_csv(
    "/data/LSY/prostate_cancer/performance_test.txt",
    sep="\t",
    index=False,
    mode="a",
)

with open("/data/LSY/prostate_cancer/performance_test.txt", "a") as file:
    file.write(f"\nAUC: {auc}")
    file.write(f"\nAccuracy: {accuracy}")
