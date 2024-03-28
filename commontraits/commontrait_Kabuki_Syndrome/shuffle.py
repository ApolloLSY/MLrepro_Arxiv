import pandas as pd
from sklearn.model_selection import train_test_split
import multiprocessing
import os

def shuffle(path):
    if not path.endswith("/"):
        path += "/"
    
    train_0_path = path+"data_train.txt"
    test_0_path = path+"data_test.txt"

    data1 = pd.read_table(train_0_path,index_col=0,header=0,delimiter="\t")
    data2 = pd.read_table(test_0_path,index_col=0,header=0,delimiter="\t")

    concatenated_data = pd.concat([data1, data2], ignore_index=False, join='outer')
    train_data, test_data = train_test_split(concatenated_data, test_size=0.2, random_state=42)

    train_data.to_csv(path+"data_train.txt",index=True,header=True,sep="\t")
    test_data.to_csv(path+"data_test.txt",index=True,header=True,sep="\t")


def shuffle_paths(paths):
    with multiprocessing.Pool() as pool:
        pool.map(shuffle, paths)

"""
         /data/LSY/finetuneing_cases/commontrait_Crohn", 
         "/data/LSY/finetuneing_cases/commontrait_Down",
         "/data/LSY/finetuneing_cases/commontrait_Kabuki_Syndrome"
         "/data/LSY/finetuneing_cases/commontrait_AD"
"""

current_directory = os.path.dirname(__file__)
paths = [current_directory]


shuffle_paths(paths)
