

# folder notes:
train and test dataset matrix: GSE109905_series_matrix.txt + GSE164563_series_matrix.txt(whole blood)
validation dataset matrix: GSE108785_series_matrix.txt(whole blood, but too small) + GSE109042_series_matrix.txt(epithelial cells)

prepare train and test dataset: prepare_data_train_and_test.py
prepare validation dataset: prepare_data_valid.py

processed train and test dataset: data_train_and_test.txt
processed validation dataset: data_valid.txt

train model and apply on test: model_train_and_test_.py (In this Article, train:test=4:1 in GSE76938, random)
apply on validation: model_valid.py




