

### folder notes:
train and test dataset matrix:: GSE97362_series_matrix.txt | GSE116300_series_matrix.txt 
validation dataset matrix: GSE218186_series_matrix.txt

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



### special notes
GSE116300_series_matrix & GSE218186_series_matrix is different from formal matrix (!lines + 1 table), their authors uploaded table in other file.

### cmd help
python modelapply.py -h

usage: modelapply.py [-h] [--datafile file] 

optional arguments:
  -h, --help            show this help message and exit
  --datafile FILE           DNAm data file name

###
example:
python modelapply.py --datafile GSE76938 matrix processed.txt
###
