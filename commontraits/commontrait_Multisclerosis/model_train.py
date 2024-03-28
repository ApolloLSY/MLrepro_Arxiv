import warnings

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, auc, roc_auc_score
from sklearn.model_selection import GridSearchCV
import os
from tqdm import tqdm

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
auc = roc_auc_score(y_train, predictions)
accuracy = accuracy_score(y_train, predictions)

train_results = pd.DataFrame({"Index": X_train.index, "Predicted": predictions, "Actual": y_train})
train_results.to_csv(current_directory+"prediction_train.txt",sep="\t",index=False,mode="a")

train_scores = pd.DataFrame({"auc": [auc], "accuracy": [accuracy]}, index=["train"])
train_scores.to_csv(current_directory+"performance.txt", mode="a", index=True)

dump(best_model, current_directory+"saved_model.joblib")
dump(selected_features,current_directory+"saved_features.joblib")
