last update: 24/02/20


### Article:
https://bmccancer.biomedcentral.com/articles/10.1186/s12885-017-3252-2



### folder notes:
train and test dataset matrix: GSE76938_series_matrix.txt (In this Article, train:test=4:1 in GSE76938, random)
validation dataset matrix: GSE127985_series_matrix.txt

prepare train and test dataset: prepare_data_train_and_test.py
prepare validation dataset: prepare_data_valid.py

processed train and test dataset: data_train_and_test.txt
processed validation dataset: data_valid.txt

train model and apply on test: model_train_and_test_.py (In this Article, train:test=4:1 in GSE76938, random)
apply on validation: model_valid.py

performance of train dataset: performance_train.txt
performance of test dataset: performance_test.txt
performance of valid dataset: performance_valid.txt



### special notes
cg16794576 (1 of 6 essential CpGs) does not exist in validation dataset.
I filled it with 0.5 in model_valid.py

### cmd help
python modelapply.py -h

usage: modelapply.py [-h] [--datafile file] 

optional arguments:
  -h, --help            show this help message and exit
  --datafile FILE           DNAm data file name


### example:
python modelapply.py --datafile GSE76938 matrix processed.txt




train and test dataset:[GSE76938](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE76938).
validation dataset：[GSE127985](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE127985)

