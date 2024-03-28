

# folder notes:
train and test dataset matrix: GSE85566_series_matrix.txt + GSE104471_series_matrix.txt   4:1 proportion
    (GSE56553_series_matrix.txt was used as test dataset before. 
    But it's performance is poor, for there are only 1 essential feature in this dataset.
    It is deprecated now. )
validation dataset matrix: GSE52074_series_matrix.txt

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