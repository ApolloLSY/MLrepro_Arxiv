
# folder notes:
train and test dataset matrix: GSE87648_series_matrix.txt | GSE81961_series_matrix.txt
validation dataset matrix: GSE32148_series_matrix.txt

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
GSE81961 dataset behaves poor for the lack of more than half essential cpgs.
filling missing cpgs with 0 or 0.5 cannot change its performance.(auc 0.40 vs 0.43)

