# %% [markdown]
# Genome-wide DNA methylation measurements in prostate tissues uncovers novel prostate cancer diagnostic biomarkers and transcription factor binding patterns
# https://bmccancer.biomedcentral.com/articles/10.1186/s12885-017-3252-2#Sec2
# 
# ![e25970b6e64ba1e5af61ffd10bb8050.png](attachment:e25970b6e64ba1e5af61ffd10bb8050.png)

# %% [markdown]
# ### 准备数据

# %%
from sklearn.linear_model import LinearRegression
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

#读入数据

dataraw=pd.read_table("D:/000Archive/ResearchProjects/DNAm_predictor/prostate/GSE76938_matrix_processed.txt", header=0, delimiter='\t')
display(dataraw.head())
selected_columns = [0] + list(range(1, dataraw.shape[1], 2))
data_sel = dataraw.iloc[:, selected_columns]

# %%
#转置
display(data_sel.head())
display(data_sel.tail())
data_sel_transposed = data_sel.transpose()
display(data_sel_transposed.head())

# %%
# 列名整理
data_sel_transposed.columns = data_sel_transposed.iloc[0]
data_sel_transposed= data_sel_transposed[1:]
display(data_sel_transposed)

# %%
#这篇文章除了用甲基化数据也用了RNA-seq SNP数据 也就是cg开头的特征和rs开头的特征 删掉rs
#Transcripts from the X and Y-chromosomes were removed prior to differential expression analysis.
data_sel_transposed = data_sel_transposed.filter(regex='^(?!rs)')
data_sel_transposed = data_sel_transposed.filter(regex='^(?!ch)')
display(data_sel_transposed)

# %% [markdown]
# “ Infinium I and II assays showed two distinct bimodal b-value distributions, so we developed a regression method to convert the type I and type II assays to a single bimodal b-distribution"
# 我下载的数据里没有Infinium I and II的标签区分，暂时也没找到，所以跳过了这一步

# %%
data_sel_transposed = data_sel_transposed.apply(pd.to_numeric, errors='coerce')

# %%
m=data_sel_transposed.iloc[0,0]
display(m)

type(m)

# %%

has_nan = data_sel_transposed.isna().any().any()
print("Any NaN values present:", has_nan)

if has_nan:
    # 将所有NaN值替换为0
    data_sel_transposed.fillna(0, inplace=True)

    # 再次检查是否还有NaN值
    has_nan_after_fillna = data_sel_transposed.isna().any().any()
    if not has_nan_after_fillna:
        print("All NaN values replaced with 0.")
    else:
        print("Some NaN values still present after attempting to replace with 0.")
else:
    print("No NaN values found.")

# %%
display(data_sel_transposed)

# %%
# CpGs with a standard deviation of less than 1% across samples were removed prior to analysis.
column_std = data_sel_transposed.std()
display(column_std)
std_percentile = column_std.quantile(0.01)
filtered_data = data_sel_transposed.loc[:, column_std >= std_percentile]


# %%
display(filtered_data)

# %%
#目标状态reference
#1='cancer'0='benign'
referencedata = pd.DataFrame(index=data_sel_transposed.index)
referencedata = referencedata.rename_axis('TargetID')
referencedata['status'] = [1] * 73 + [0] * 63
display(referencedata)

# %%
from sklearn.model_selection import train_test_split

train_data, test_data = train_test_split(data_sel_transposed, test_size=0.2, random_state=42)
train_data = pd.merge(train_data, referencedata, left_index=True, right_index=True, how='left')
test_data= pd.merge(test_data, referencedata, left_index=True, right_index=True, how='left')
display(train_data)
display(test_data)

# %%


X_train = train_data.iloc[:, :-1]  # 特征
y_train = train_data.iloc[:, -1]  # 目标标签

X_test = test_data.iloc[:, :-1]
y_test = test_data.iloc[:, -1]

model_allfeatures = LinearRegression()
model_allfeatures.fit(X_train, y_train)

y_pred = model_allfeatures.predict(X_test)
display(y_pred)
#y_sig_pred = 1 / (1 + np.exp(-y_pred))
#display(y_sig_pred)
threshold=0.5
y_binary_pred = (y_pred >= threshold).astype(int)

accuracy = accuracy_score(y_test, y_binary_pred)
print(f"Accuracy: {accuracy}")

# %% [markdown]
# 所有CpGs跑出来的线性模型的AUC是0.8214285714285714

# %% [markdown]
# 附录里给出了他们找出的6个特征 分别是'cg00054525', 'cg16794576', 'cg24581650', 'cg15338327', 'cg00054525', 'cg14781281'
# 
# 他们的AUC是0.9747373
# 
# ![Alt text](54b31f9aa42b6977e6a1b7394c7f891.png)
# 
# ![34fd7941995847e59b6750cc33d0af8.png](attachment:34fd7941995847e59b6750cc33d0af8.png)

# %%
selected_features = ['cg00054525', 'cg16794576', 'cg24581650', 'cg15338327', 'cg00054525', 'cg14781281']
X_train_sel6 = train_data.loc[:,selected_features]
X_test_sel6 = test_data.loc[:,selected_features]

model_6features = LinearRegression()
model_6features.fit(X_train_sel6, y_train)

y_pred_sel6 = model_6features.predict(X_test_sel6)
display(y_pred_sel6)
#y_sig_pred_sel6 = 1 / (1 + np.exp(-y_pred_sel6))
#display(y_sig_pred_sel6)
threshold=0.5
y_binary_pred_sel6 = (y_pred_sel6 >= threshold).astype(int)

accuracy = accuracy_score(y_test, y_binary_pred_sel6)
print(f"Accuracy: {accuracy}")

# %% [markdown]
# 发现不能用sigmoid函数，一用AUC就没差别了

# %%
coefficients_sel6_features = model_6features.coef_
display(coefficients_sel6_features)


