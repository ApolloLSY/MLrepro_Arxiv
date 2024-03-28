
# Article:
https://bmccancer.biomedcentral.com/articles/10.1186/s12885-017-3252-2



# folder notes:
train and test dataset matrix: 
    4:1
    GSE66351_series_matrix.txt | GSE80970_series_matrix.txt
validation dataset matrix: 
    GSE208623_series_matrix.txt

prepare train dataset: prepare_data_train.py
prepare test dataset: prepare_data_test.py
prepare validation dataset: prepare_data_valid.py

processed train dataset: data_train.txt
processed test dataset: data_test.txt
processed validation dataset: data_valid.txt

train model: model_test.py
apply on test: model_train.py
apply on validation: model_valid.py

performance of train dataset: performance_train.txt
performance of test dataset: performance_test.txt
performance of validation dataset: performance_valid.txt


# Special notes
Some CpGs do not exist in validation dataset.
I filled it with 0.5 in model_valid.py
but AUC is a bit low (0.61)


