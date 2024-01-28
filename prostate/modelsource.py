# Genome-wide DNA methylation measurements in prostate tissues uncovers novel prostate cancer diagnostic biomarkers and transcription factor binding patterns
# https://bmccancer.biomedcentral.com/articles/10.1186/s12885-017-3252-2#Sec2


from sklearn.linear_model import LinearRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split
from joblib import dump

import pandas as pd

import warnings
warnings.filterwarnings('ignore')


#load data
datafile=pd.read_table("D:/000Archive/ResearchProjects/DNAm_predictor/prostate/GSE76938_matrix_processed.txt", header=0, delimiter='\t')
selected_columns = [0] + list(range(1, datafile.shape[1], 2))
data_sel = datafile.iloc[:, selected_columns]


#transpose
data_sel_transposed = data_sel.transpose()


# column name
data_sel_transposed.columns = data_sel_transposed.iloc[0]
data_sel_transposed= data_sel_transposed[1:]


#delete RNA-seq SNP data
#X and Y-chromosomes were removed
data_sel_transposed = data_sel_transposed.filter(regex='^(?!rs)')
data_sel_transposed = data_sel_transposed.filter(regex='^(?!ch)')


# “ Infinium I and II assays showed two distinct bimodal b-value distributions, so we developed a regression method to convert the type I and type II assays to a single bimodal b-distribution"
# this step is skipped


#str2numeric
data_sel_transposed = data_sel_transposed.apply(pd.to_numeric, errors='coerce')


#isna
has_nan = data_sel_transposed.isna().any().any()
print("Any NaN values present:", has_nan)


if has_nan:
    # NaN20
    data_sel_transposed.fillna(0, inplace=True)

    # check again
    has_nan_after_fillna = data_sel_transposed.isna().any().any()
    if not has_nan_after_fillna:
        print("All NaN values replaced with 0.")
    else:
        print("Some NaN values still present after attempting to replace with 0.")
else:
    print("No NaN values found.")


# CpGs with a standard deviation of less than 1% across samples were removed prior to analysis.
column_std = data_sel_transposed.std()
std_percentile = column_std.quantile(0.01)
filtered_data = data_sel_transposed.loc[:, column_std >= std_percentile]


#label reference
#1='cancer'0='benign'
referencedata = pd.DataFrame(index=data_sel_transposed.index)
referencedata = referencedata.rename_axis('TargetID')
referencedata['status'] = [1] * 73 + [0] * 63


#split train and test data
train_data, test_data = train_test_split(filtered_data, test_size=0.2, random_state=42)
train_data = pd.merge(train_data, referencedata, left_index=True, right_index=True, how='left')
test_data= pd.merge(test_data, referencedata, left_index=True, right_index=True, how='left')

X_train = train_data.iloc[:, :-1]  # features
y_train = train_data.iloc[:, -1]  # labels

X_test = test_data.iloc[:, :-1]
y_test = test_data.iloc[:, -1]


'''
# all CpGs model 
# AUC:0.8214285714285714
model_allfeatures = LinearRegression()
model_allfeatures.fit(X_train, y_train)

y_pred = model_allfeatures.predict(X_test)

threshold=0.5
y_binary_pred = (y_pred >= threshold).astype(int)

accuracy = accuracy_score(y_test, y_binary_pred)
print(f"Accuracy: {accuracy}")
'''

# 6 CpGs model 'cg00054525', 'cg16794576', 'cg24581650', 'cg15338327', 'cg00054525', 'cg14781281'
# AUC: 0.9747373
selected_features = ['cg00054525', 'cg16794576', 'cg24581650', 'cg15338327', 'cg00054525', 'cg14781281']
X_train_sel6 = train_data.loc[:,selected_features]
X_test_sel6 = test_data.loc[:,selected_features]

model_6features = LinearRegression()
model_6features.fit(X_train_sel6, y_train)

y_pred_sel6 = model_6features.predict(X_test_sel6)

threshold=0.5
y_binary_pred_sel6 = (y_pred_sel6 >= threshold).astype(int)

accuracy = accuracy_score(y_test, y_binary_pred_sel6)
print(f"Accuracy: {accuracy}")


# save model
dump(model_6features, "D:/000Archive/ResearchProjects/DNAm_predictor/myrepro/prostate/model6CpGs.joblib")

# load model
#loaded_model = load('model.joblib')

'''
coefficients_sel6_features = model_6features.coef_
display(coefficients_sel6_features)
'''


    