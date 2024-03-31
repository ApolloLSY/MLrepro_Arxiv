import warnings

import numpy as np
import pandas as pd
from joblib import dump, load
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, auc, roc_auc_score,roc_curve
from sklearn.model_selection import GridSearchCV
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

current_directory = os.path.dirname(__file__)

if not current_directory.endswith("/"):
    current_directory += "/"

inputfilepath = current_directory+"data_train.txt"
outputfilepath = current_directory

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
    # NaN20
    datafile.fillna(0.5, inplace=True)
    # check again
    has_nan_after_fillna = datafile.isna().any().any()
    if not has_nan_after_fillna:
        print("All NaN values replaced with 0.5.")
    else:
        print("Some NaN values still present failing to replace with 0.5.")
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

X_train = data.drop("status", axis=1)
y_train = data["status"]

parameters = [{'penalty':['l1']}, 
            {'C':np.logspace(-5, 2, num=16)}]
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
auc_score = roc_auc_score(y_train, predictions)
accuracy = accuracy_score(y_train, predictions)

train_results = pd.DataFrame({"Index": X_train.index, "Predicted": predictions, "Actual": y_train})
train_results.to_csv(current_directory+"prediction_train.txt",sep="\t",index=False,mode="a")

train_scores = pd.DataFrame({"auc": [auc_score], "accuracy": [accuracy]}, index=["train"])
train_scores.to_csv(current_directory+"performance.txt", mode="a", index=True)

dump(best_model, current_directory+"saved_model.joblib")
dump(selected_features,current_directory+"saved_features.joblib")


fpr, tpr, thresholds = roc_curve(y_train, predictions)
roc_auc = auc(fpr, tpr)
youden_j = tpr - (1 - (1 - fpr))
optimal_idx = np.argmax(youden_j)
optimal_threshold = thresholds[optimal_idx]

optimal_sensitivity = tpr[optimal_idx]
optimal_specificity = 1 - fpr[optimal_idx]

plt.figure()
plt.plot(fpr, tpr, color='#c90016', lw=2, label='Train ROC (area = %0.2f)' % roc_auc)
plt.scatter(fpr[optimal_idx], tpr[optimal_idx], marker='o', color='blue')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.text(fpr[optimal_idx], tpr[optimal_idx], f'({optimal_sensitivity:.2f}, {optimal_specificity:.2f})', fontsize=9)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')

# test

inputfilepath = current_directory+"data_test.txt"
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

X_test = data.drop("status", axis=1)
y_test = data["status"]

loaded_model = load(current_directory+"saved_model.joblib")
selected_features = load(current_directory+"saved_features.joblib")
X_test = check_and_fill_cpg(X_test, selected_features)
X_test_selected = X_test[selected_features]
predictions = loaded_model.predict(X_test_selected)

auc_score = roc_auc_score(y_test, predictions)
accuracy = accuracy_score(y_test, predictions)

test_results = pd.DataFrame({"Index": X_test.index, "Predicted": predictions, "Actual": y_test})
test_results.to_csv(outputfilepath+"prediction_test.txt",sep="\t",index=False,mode="a")
test_scores = pd.DataFrame({"auc": [auc_score], "accuracy": [accuracy]}, index=["test"])
test_scores.to_csv(outputfilepath+"performance.txt",mode="a",index=True)


fpr, tpr, thresholds = roc_curve(y_test, predictions)
roc_auc = auc(fpr, tpr)
youden_j = tpr - (1 - (1 - fpr))
optimal_idx = np.argmax(youden_j)
optimal_threshold = thresholds[optimal_idx]

optimal_sensitivity = tpr[optimal_idx]
optimal_specificity = 1 - fpr[optimal_idx]

plt.plot(fpr, tpr, color='#FF6347', lw=2, label='Test ROC (area = %0.2f)' % roc_auc)
plt.scatter(fpr[optimal_idx], tpr[optimal_idx], marker='o', color='blue')
plt.text(fpr[optimal_idx], tpr[optimal_idx], f'({optimal_sensitivity:.2f}, {optimal_specificity:.2f})', fontsize=9)

 
# valid

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

auc_score = roc_auc_score(y_valid, predictions)
accuracy = accuracy_score(y_valid, predictions)

valid_results = pd.DataFrame({"Index": X_valid.index, "Predicted": predictions, "Actual": y_valid})
valid_results.to_csv(outputfilepath+"prediction_valid.txt",sep="\t",index=False,mode="a")
valid_scores = pd.DataFrame({"auc": [auc_score], "accuracy": [accuracy]}, index=["valid"])
valid_scores.to_csv(outputfilepath+"performance.txt",mode="a",index=True)


fpr, tpr, thresholds = roc_curve(y_valid, predictions)
roc_auc = auc(fpr, tpr)
youden_j = tpr - (1 - (1 - fpr))
optimal_idx = np.argmax(youden_j)
optimal_threshold = thresholds[optimal_idx]

optimal_sensitivity = tpr[optimal_idx]
optimal_specificity = 1 - fpr[optimal_idx]


plt.plot(fpr, tpr, color='#ffbf47', lw=2, label='Valid ROC (area = %0.2f)' % roc_auc)
plt.scatter(fpr[optimal_idx], tpr[optimal_idx], marker='o', color='blue', label='Optimal cut-off point')
plt.text(fpr[optimal_idx], tpr[optimal_idx], f'({optimal_sensitivity:.2f}, {optimal_specificity:.2f})', fontsize=9)
plt.title(f"{os.path.basename(os.path.dirname(__file__))} logistic regression")
plt.legend(loc="lower right")

plt.savefig(current_directory + "roc.png")
plt.clf()