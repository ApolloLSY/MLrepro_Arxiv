
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.feature_selection import RFE
import numpy as np
import pandas as pd

#load
train=pd.read_csv("D:/000Archive/ResearchProjects/DNAm_predictor/pancreattrain", delimiter='\t', header=None)
test=pd.read_csv("D:/000Archive/ResearchProjects/DNAm_predictor/pancreattest", delimiter='\t', header=None)


X_train = train.iloc[:, 1:] 
y_train = train.iloc[:, 0] 
X_test= test.iloc[:, 1:] 
y_test = test.iloc[:, 0] 

imputer = SimpleImputer(strategy='constant',fill_value=0)
X_train = imputer.fit_transform(X_train)
X_test = imputer.fit_transform(X_test)

base_classifier = LogisticRegression(multi_class='ovr')  
rfe = RFE(estimator=base_classifier, n_features_to_select=7) 
model = make_pipeline(rfe, StandardScaler(), base_classifier)
model.fit(X_train, y_train)


X_test_selected = rfe.transform(X_test)
y_pred = model.predict(X_test_selected)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")