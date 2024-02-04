# %%
! pip install statsmodels


# %%
#Siyuan Li
#last update: 2024/2/4


from sklearn.linear_model import LassoCV, ElasticNetCV,LogisticRegression
from sklearn.model_selection import cross_val_predict,train_test_split,KFold
from sklearn.metrics import roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor


import pandas as pd
import os

# %%
result_df = pd.read_csv("/data/Clockbase/Methylation/sample_disease.txt",sep=" ", header=0)  

#filter samples with disease label
#selected_columns = df_sample.columns[df_sample.iloc[0].notna()]
#result_df = pd.concat([result_df, df_sample[selected_columns]], axis=1)

result_df.head()


# %%
print(result_df['disease'].unique())
print(result_df['platform'].unique())

# %%
num_rows = result_df.shape[0]
print("数据框有", num_rows, "行。")


# %%
nan_rows = result_df[result_df['disease'].isna()]
num_nan_rows = nan_rows.shape[0]
print(" 'disease' 列中有", num_nan_rows, "行的值是 NaN。")


# %%
'''
result_df = result_df.dropna(subset=['disease'])
print(result_df['disease'].unique())
num_rows = result_df.shape[0]
print("数据框有", num_rows, "行。")
'''

# %%
result_df.head()

# %%
#df2 = pd.read_csv("/data/Clockbase/Methylation/disease/GSE43414.csv.gz",compression='gzip')  
#df2.head()


# %%
#df2t=df2.transpose()
#df2t.head()

# %%
#print(df2t.index)

# %%


# %%
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 100)

# %%
unique_disease = result_df['disease'].unique()
print(unique_disease)

# %%
# find path
folder_path = "/data/Clockbase/Methylation/disease/"
file_names = os.listdir(folder_path)
# output_path
output_path="/data/Clockbase/Methylation/disease_predictor/"

unique_disease = result_df['disease'].unique()
disease_gse_dict = {}
train_ratio = 0.7
# 遍历每个不同的 'project_id'
for disease in unique_disease:
    
    if pd.isna(disease):
        continue
    
    selected_rows1 = result_df.loc[result_df['disease'] == disease]
    unique_project_ids = selected_rows1['project_id'].unique()
    print(f"在 'disease' 是 '{disease}' 的行中，'roject_id' 列有 {len(unique_project_ids)} 种不同的值：")
    print(unique_project_ids)
    
    disease_gse_dict[disease] = list(unique_project_ids)
display(disease_gse_dict)


# %%
def preprocess_data(data):
    # Extracting CpG columns
    cpg_columns = data.columns[:-2]

    # Excluding outliers based on visual inspection
    outlier_threshold1 = 10
    data = data[data[cpg_columns].apply(lambda x: (x <= outlier_threshold1).all(), axis=1)]

    # Sample Exclusions: Samples excluded if ≥1% of CpGs had a detection p-value exceeding 0.05
    outlier_threshold2 = data[cpg_columns].mean() + 1.96 * data[cpg_columns].std() / np.sqrt(len(data))
    columns_to_retain = []
    for cpg_column in cpg_columns:
        num_outliers = (data[cpg_column] > outlier_threshold2[cpg_column]).sum()
        if num_outliers <= 0.01 * len(data):
            columns_to_retain.append(cpg_column)
    data = data[columns_to_retain]

    # Remove CpGs with missing values
    df_cleaned = data.dropna(axis=1)

    # Normalization
    scaler = StandardScaler()
    df_cleaned = scaler.fit_transform(df_cleaned.iloc[:, :-2])

    return df_cleaned

# %%
MODEL_RESULT = pd.DataFrame(columns=['Disease', 'Best_lambda_index','Best_model_coefs', 'AUC'])
for disease in unique_disease:
  
  
    gse_list = disease_gse_dict[disease]
      
    if len(gse_list) == 1:
        train_set = gse_list
        test_set = gse_list #Set it same for now
    else:
        # Calculate the number of training and testing sets
        num_train = max(1, int(len(gse_list) * train_ratio))
        num_test = len(gse_list) - num_train
        
        # Randomly select training and testing sets from all GSE files of the disease
        np.random.shuffle(gse_list)
        train_set = gse_list[:num_train]
        test_set = gse_list[num_train:]



    train_data = pd.DataFrame()
    # Load training set data
    for gse_file in train_set:
        file_path = folder_path + gse_file + ".csv.gz"  
        dt = pd.read_csv(file_path,compression='gzip')
        dt_trans=dt.transpose()
        train_data=pd.concat([train_data, dt_trans], axis=0)
    train_data_matched = train_data.merge(df_sample, left_index=True, right_on=['disease', 'project_id'])

    test_data = pd.DataFrame()
    # Load testing set data
    for gse_file in test_set:
        file_path = folder_path + gse_file + ".csv.gz"  
        dt = pd.read_csv(file_path,compression='gzip')
        dt_trans=dt.transpose()
        test_data=pd.concat([test_data, dt_trans], axis=0)
    test_data_matched = test_data.merge(df_sample, left_index=True, right_on=['disease', 'project_id'])
    
    # Preprocess train_data_matched
    X_train = preprocess_data(train_data_matched)
    y_train = train_data_matched.iloc[:, -2]

    # Preprocess test_data_matched
    test_data_matched_processed = preprocess_data(test_data_matched)
    X_test = preprocess_data(test_data_matched)
    y_test = test_data_matched.iloc[:, -2]
    
    
    
    model = LassoCV(alphas=[1], cv=10, random_state=42)
    model.fit(X_train, y_train)

    best_lambda_index = np.argmin(model.lambda_path_['val_mean_score'])
    best_model_coefs = model.coef_[:, best_lambda_index]

    predictions = model.predict(X_test, lamb=model.lambda_path_['lambda_max'])

  
    auc = roc_auc_score(y_test, predictions )
   
    '''
    fpr, tpr, _ = roc_curve(y_test, predictions)
    roc_auc = auc(fpr, tpr)
    print(f"ROC AUC: {roc_auc}")
    '''
    
    MODEL_RESULT = MODEL_RESULT.append({'Disease': disease, 'Best_lambda_index':best_lambda_index, 'Best_model_coefs': best_model_coefs, 'AUC': auc}, ignore_index=True)


disease_name = disease.replace(" ", "_") + ".csv"
MODEL_RESULT.to_csv(os.path.join(output_path, 'MODEL_RESULT_'+disease_name), index=False)


