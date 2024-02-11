# %%
#Siyuan Li
#last update: 2024/2/1


from sklearn.linear_model import LassoCV, ElasticNetCV,LogisticRegression
from sklearn.model_selection import cross_val_predict,train_test_split,KFold
from sklearn.metrics import roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import pandas as pd
from io import StringIO

# %%
file_path = 'E:\GSE48684_series_matrix.txt'

with open(file_path, 'r') as file:
    lines = file.readlines()

sample_title = []
sample_geo_accession = []

for line in lines:
    if line.startswith("!Sample_title"):
        sample_title = line.split("\t")[0:]
    elif line.startswith("!Sample_geo_accession"):
        sample_geo_accession = line.split("\t")[0:]
        
data = pd.DataFrame({
    'Sample_title': sample_title[1:],
    'Sample_geo_accession': sample_geo_accession[1:]
})

print(data)


# %%
data['status'] = data['Sample_title'].apply(lambda x: 0 if 'normal' in x else (1 if 'CRC' in x else None))

# delete'adenoma'
data = data[data['status'] != None]
# delete 'Sample_title'
data.drop(columns=['Sample_title'], inplace=True)

print(data.head())

# %%

with open(file_path, 'r') as file:
    lines = file.readlines()


filtered_lines = [line for line in lines if not line.startswith('!')]

data2 = pd.read_csv(StringIO('\n'.join(filtered_lines)), sep='\t', header=None, index_col=0)
data2=data2.T

data2.head()

# %%
merged_data = pd.concat([data, data2],axis=1)
merged_data.head()

# %%
'''
CG = ['cg06551493', 'cg01419670', 'cg16530981', 'cg18022036','cg12691488', 'cg17292758', 'cg16170495',
      'cg11240062', 'cg21585512', 'cg24702253', 'cg17187762', 'cg05983326', 'cg06825163',
      'cg11885357', 'cg08829299', 'cg07044115']
beta= [-0.41, 0.4332, 0.2895, -0.5172, -0.3915, -0.3246, -0.2886, 0.2451, -0.5651, 
             0.3615, -0.2445, -0.3951, -0.5089, -0.2504, -0.2357,-0.3607]
'''


'''
result_data.iloc[:, 1:] = result_data.iloc[:, 1:].replace(" ", np.nan).apply(pd.to_numeric)
result_matrix = np.matmul(beta, result_data.iloc[:, 1:].values)

print("矩阵乘法结果：", result_matrix)
'''

# %%
selected_features = ['cg17187762', 'cg02964172']

X_train, X_test, y_train, y_test = train_test_split(merged_data[selected_features], merged_data['status'], test_size=0.2, random_state=42)


rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

rf_model.fit(X_train, y_train)

y_pred = rf_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("模型准确率：", accuracy)




