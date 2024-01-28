'''
python main.py -h

usage: main.py [-h] [--datafile file] 

optional arguments:
  -h, --help            show this help message and exit
  --datafile FILE           DNAm data file name
  
'''

from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pandas as pd
from joblib import load
import os
import sys
import argparse

import warnings
warnings.filterwarnings('ignore')

import time




def parse_arguments(parser):
    
    parser.add_argument('--datafile', type=str,
                        help='DNAm data file name')
    
    args = parser.parse_args()

    return args

  
    
def main(args):

    print(args)
    
    datafile=args.datafile
        



    #load data

    dataraw=pd.read_table(datafile, header=0, delimiter='\t')
    selected_columns = [0] + list(range(1, dataraw.shape[1], 2))
    data_sel = dataraw.iloc[:, selected_columns]
    

    #transpose
    data_sel_transposed = data_sel.transpose()
 
  
    # column name
    data_sel_transposed.columns = data_sel_transposed.iloc[0]
    data_sel_transposed= data_sel_transposed[1:]
  
    
    #delete RNA-seq SNP data
    #X and Y-chromosomes were removed
    data_sel_transposed = data_sel_transposed.filter(regex='^(?!rs)')
    data_sel_transposed = data_sel_transposed.filter(regex='^(?!ch)')

    
    # “ Infinium I and II assays showed two distinct bimodal b-value distributions, so we developed a regression method to convert the type I and type II assays to a single bimodal b-distribution"
    # this step is skipped


    #str2numeric
    data_sel_transposed = data_sel_transposed.apply(pd.to_numeric, errors='coerce')

    
    #isna
    has_nan = data_sel_transposed.isna().any().any()
    print("Any NaN values present:", has_nan)

    if has_nan:
        # NaN20
        data_sel_transposed.fillna(0, inplace=True)

        # check again
        has_nan_after_fillna = data_sel_transposed.isna().any().any()
        if not has_nan_after_fillna:
            print("All NaN values replaced with 0.")
        else:
            print("Some NaN values still present after attempting to replace with 0.")
    else:
        print("No NaN values found.")

    
    # CpGs with a standard deviation of less than 1% across samples were removed prior to analysis.
    column_std = data_sel_transposed.std()
    std_percentile = column_std.quantile(0.01)
    filtered_data = data_sel_transposed.loc[:, column_std >= std_percentile]




    #apply model
    X = filtered_data.iloc[:, :]
    selected_features = ['cg00054525', 'cg16794576', 'cg24581650', 'cg15338327', 'cg00054525', 'cg14781281']
    X_sel6 = X.loc[:,selected_features]

    #
    loaded_model = load("D:/000Archive/ResearchProjects/DNAm_predictor/myrepro/prostate/model6CpGs.joblib")

    y_pred_sel6 = loaded_model.predict(X_sel6)
    
    threshold=0.5
    y_binary_pred_sel6 = (y_pred_sel6 >= threshold).astype(int)

    print(y_binary_pred_sel6)


  

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Training model')
    args = parse_arguments(parser)
    print(args)
    start_time = time.time()
    main(args)
    print("--- %s seconds ---" % (time.time() - start_time))
    