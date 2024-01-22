# %% [markdown]
# load data

# %%
from sklearn.linear_model import LassoCV, ElasticNetCV,LogisticRegression,KFold
from sklearn.model_selection import cross_val_predict,train_test_split
from sklearn.metrics import roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScalel
import numpy as np
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.api import OrdinalModel, family



mdata=pd.read_table("D:/000Archive/ResearchProjects/DNAm_predictor/EWASdata/commontraitsmethy.txt", header=0, delimiter=' ')
profile=pd.read_table("D:/000Archive/ResearchProjects/DNAm_predictor/EWASdata/commontraitsprofile.txt", header=0, delimiter=' ')


# %% [markdown]
# Initial Data Cleaning:
# 
# Outlier Identification: Visual inspection of a plot of log median intensity of methylated versus unmethylated signal was performed to identify outliers, which were then excluded from the analysis.
# Sample Exclusions: Samples were excluded based on predicted sex mismatch, as determined by DNA methylation data compared to the sex recorded in the database. Additionally, samples were excluded if ≥1% of CpGs had a detection p-value exceeding 0.05.
# CpG Filtering: Further filtering removed sites with missing values, non-autosomal sites, non-CpG sites, and CpG sites not present on the Illumina 450k array.
# Background Correction: After background correction, probes were removed if poorly detected (P>0.01) in >5% of samples or of low quality (via manual inspection)(We dont need this step).
# Sample Removal: Samples were removed if they had a low call rate (P<0.01 for <95% of probes), a poor match between genotype and SNP control probes, or incorrect DNAm-predicted sex.(We dont need this step)

# %%
cpg_columns = mdata.columns[1:]

# Create subplots for each CpG column
fig, axes = plt.subplots(nrows=len(cpg_columns), ncols=1, figsize=(8, 2 * len(cpg_columns)))

# Plot log median intensity for each CpG column
for i, cpg_column in enumerate(cpg_columns):
    axes[i].boxplot(mdata[cpg_column], vert=False)
    axes[i].set_title(cpg_column)

plt.tight_layout()
plt.show()

# Excluding outliers from the analysis based on visual inspection
outlier_threshold1 = 10
mdata = mdata[mdata[cpg_columns].apply(lambda x: (x <= outlier_threshold1).all(), axis=1)]

# Sample Exclusions: Samples excluded if ≥1% of CpGs had a detection p-value exceeding 0.05
outlier_threshold2 = mdata[cpg_columns].mean() + 1.96 * mdata[cpg_columns].std() / np.sqrt(len(mdata))
columns_to_retain = []
for cpg_column in cpg_columns:
    num_outliers = (mdata[cpg_column] > outlier_threshold2[cpg_column]).sum()
    if num_outliers <= 0.01 * len(mdata):
        columns_to_retain.append(cpg_column)
mdata = mdata[columns_to_retain]

# Remove CpGs with missing values
df_cleaned = mdata.dropna(axis=1)



# %% [markdown]
#  Tenfold cross-validation was applied and the mixing parameter (alpha) was set to 1 to apply a LASSO penalty

# %%

# dummies and normalization
df_cleaned = pd.get_dummies(profile, columns=profile[:,1:], drop_first=True)
scaler= StandardScaler()
df_cleaned=scaler.fit_transform(df_cleaned[:,1:])


num_columns = profile.shape[1]
# They processed all the labels into binary categories
for i in range(1, num_columns):
    X_train = df_cleaned[:,1:]  
    y_train = profile[:,i] 
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=100)

    model = LassoCV(alphas=[1], cv=10, random_state=42, max_iter=10000)
    model.fit(X_train, y_train)


    best_lambda_index = np.argmin(model.lambda_path_['val_mean_score'])
    best_model_coefs = model.coef_[:, best_lambda_index]


    predictions = model.predict(X_test, lamb=model.lambda_path_['lambda_max'])

    print("Best Model Coefficients:", best_model_coefs)
    print("Predictions on Test Set:", predictions)

    auc = roc_auc_score(y_test, predictions )
    print(f'AUC: {auc}')
    
    fpr, tpr, _ = roc_curve(y_test, predictions)
    roc_auc = auc(fpr, tpr)
    print(f"ROC AUC: {roc_auc}")











