import warnings

import numpy as np
import pandas as pd
from joblib import dump,load
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, auc, roc_auc_score
from sklearn.model_selection import GridSearchCV
import os
from tqdm import tqdm


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

directory = "/data/LSY/z_preparing_and_parts/"
inpath = "lung_cacner"

current_directory = directory + "/" + inpath + "/" 
inputfilepath = current_directory + "data_train.txt"
datafile = pd.read_table(
    inputfilepath,
    index_col=0,
    header=0,
    delimiter="\t")

metadata = datafile['status']
datafile = datafile.drop('status', axis=1)
## isna
has_nan = datafile.isna().any().any()
print("Any NaN values present:", has_nan)
if has_nan:
    ### NaN20
    datafile.fillna(0.5, inplace=True)
    ### check again
    has_nan_after_fillna = datafile.isna().any().any()
    if not has_nan_after_fillna:
        print("All NaN values replaced with 0.5")
    else:
        print("Some NaN values still present failing to replace with 0.5")
else:
    print("No NaN values found.")

# CDO1, GSHR, HOXA11, HOXB4-1, HOXB4-2, HOXB4-3, HOXB4-4, LHX9, MIR196A1, PTGER4-1, and PTGER4-2
# translated to cpgs:
cpgs = [
    "cg01452847",
    "Cg02422694",
    "cg07015911",
    "Cg07852825",
    "cg08089301",
    "cg08516516",
    "cg09076431",
    "cg09194159",
    "cg11036833",
    "cg12806763",
    "cg14345497",
    "Cg14458834",
    "cg15760840",
    "cg15987088",
    "cg19081437",
    "cg21460081",
    "cg21546671",
    "cg23180938",
    "cg24114154",
    "cg26327071",
    "cg27071460"
]
check_and_fill_cpg(datafile,cpgs)
datafile = datafile[cpgs]
## Sample Exclusions: Samples excluded if ≥1% of CpGs had a detection p-value exceeding 0.05
outlier_threshold2 = datafile.mean() + 1.96 * datafile.std() / np.sqrt(len(datafile))
columns_to_drop = []
for cpg_column in datafile.columns:
    num_outliers = (datafile[cpg_column] > outlier_threshold2[cpg_column]).sum()
    if num_outliers > 0.01 * datafile.shape[1]:
        columns_to_drop.append(cpg_column)
datafile = datafile.drop(columns=columns_to_drop)

X_train = datafile
y_train = metadata

parameters = [{'C':np.logspace(-5, 2, num=16)}]
logreg=LogisticRegression(max_iter=1000)
with tqdm(total=len(parameters)) as pbar:
    grid_search = GridSearchCV(estimator = logreg,  
                            param_grid = parameters,
                            scoring = 'accuracy',
                            cv = 5,
                            verbose=1,
                            n_jobs=16)
    grid_search.fit(X_train, y_train)   
    pbar.update(1)
print("tuned hpyerparameters :(best parameters) ",grid_search.best_params_)
print("best accuracy :",grid_search.best_score_)

best_model = grid_search.best_estimator_
selected_features = X_train.columns[(best_model.coef_!=0).flatten()]

X_train_selected = X_train[selected_features]
best_model.fit(X_train_selected, y_train)

predictions = best_model.predict(X_train_selected)
auc = roc_auc_score(y_train, predictions)
accuracy = accuracy_score(y_train, predictions)


train_results = pd.DataFrame({"Index": X_train.index, "Predicted": predictions, "Actual": y_train})
train_results.to_csv(current_directory+"prediction_train.txt",sep="\t",index=False,mode="a")

train_scores = pd.DataFrame({"auc": [auc], "accuracy": [accuracy]}, index=["train"])
train_scores.to_csv(current_directory+"performance.txt", mode="a", index=True)

dump(best_model, current_directory+"saved_model.joblib")
dump(selected_features,current_directory+"saved_features.joblib")


# test
## load data
inputfilepath2 = current_directory + "data_test.txt"
datafile = pd.read_table(
    inputfilepath2,
    index_col=0,
    header=0,
    delimiter="\t",
)

metadata = datafile['status']
datafile = datafile.drop('status', axis=1)

## isna
has_nan = datafile.isna().any().any()
print("Any NaN values present:", has_nan)


if has_nan:
    ### NaN20.5
    datafile.fillna(0.5, inplace=True)

    #### check again
    has_nan_after_fillna = datafile.isna().any().any()
    if not has_nan_after_fillna:
        print("All NaN values replaced with 0.5")
    else:
        print("Some NaN values still present failing to replace with 0.5")
else:
    print("No NaN values found.")

## Sample Exclusions: Samples excluded if ≥1% of CpGs had a detection p-value exceeding 0.05
outlier_threshold2 = datafile.mean() + 1.96 * datafile.std() / np.sqrt(len(datafile))
columns_to_drop = []
for cpg_column in datafile.columns:
    num_outliers = (datafile[cpg_column] > outlier_threshold2[cpg_column]).sum()
    if num_outliers > 0.01 * datafile.shape[1]:
        columns_to_drop.append(cpg_column)
datafile = datafile.drop(columns=columns_to_drop)

X_test = datafile
y_test = metadata

loaded_model = load(current_directory+"saved_model.joblib")
selected_features = load(current_directory+"saved_features.joblib")
X_test = check_and_fill_cpg(X_test, selected_features)
X_test_selected = X_test[selected_features]
predictions = loaded_model.predict(X_test_selected)

auc = roc_auc_score(y_test, predictions)
accuracy = accuracy_score(y_test, predictions)

test_results = pd.DataFrame({"Index": X_test.index, "Predicted": predictions, "Actual": y_test})
test_results.to_csv(current_directory+"prediction_test.txt",sep="\t",index=False,mode="a")
test_scores = pd.DataFrame({"auc": [auc], "accuracy": [accuracy]}, index=["test"])
test_scores.to_csv(current_directory+"performance.txt",mode="a",index=True)

    
# valid
## load
inputfilepath3 = current_directory + "data_valid.txt"
datafile = pd.read_table(
    inputfilepath3,
    index_col=0,
    header=0,
    delimiter="\t",
)

metadata = datafile['status']
datafile = datafile.drop('status', axis=1)


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

## Sample Exclusions: Samples excluded if ≥1% of CpGs had a detection p-value exceeding 0.05
outlier_threshold2 = datafile.mean() + 1.96 * datafile.std() / np.sqrt(len(datafile))
columns_to_drop = []
for cpg_column in datafile.columns:
    num_outliers = (datafile[cpg_column] > outlier_threshold2[cpg_column]).sum()
    if num_outliers > 0.01 * datafile.shape[1]:
        columns_to_drop.append(cpg_column)
datafile = datafile.drop(columns=columns_to_drop)

X_valid = datafile
y_valid = metadata

loaded_model = load(current_directory+"saved_model.joblib")
selected_features = load(current_directory+"saved_features.joblib")
X_valid = check_and_fill_cpg(X_valid, selected_features)
X_valid_selected = X_valid[selected_features]
predictions = loaded_model.predict(X_valid_selected)

auc = roc_auc_score(y_valid, predictions)
accuracy = accuracy_score(y_valid, predictions)

valid_results = pd.DataFrame({"Index": X_valid.index, "Predicted": predictions, "Actual": y_valid})
valid_results.to_csv(current_directory+"prediction_valid.txt",sep="\t",index=False,mode="a")
valid_scores = pd.DataFrame({"auc": [auc], "accuracy": [accuracy]}, index=["valid"])
valid_scores.to_csv(current_directory+"performance.txt",mode="a",index=True)
