
author:Siyuan Li

last update: 24/02/05



They were originally predicting common traits, but we used their method to predict diseases, with a training set of 28 diseases on [EWAS](https://ngdc.cncb.ac.cn/ewas/datahub/download).


In ```source.py```, divide GSE files containing the same disease into training and testing sets by file, train and test each one individually, and finally output the model parameters and AUC.
