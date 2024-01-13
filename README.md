# Disease predictor

## No.1 Cancerlocator

[文献](https://genomebiology.biomedcentral.com/articles/10.1186/s13059-017-1191-5?site=n_detail_link#Sec7)

预测多个癌症的 是少有的给了code的文献[github](https://github.com/jasminezhoulab/CancerLocator/tree/v1.0.1) 不过是java写的而且很长 

用的自己产生的模拟数据

我前天读完了他们写的几个类中的两个 还有两个类没读完 感觉内容太多了 而且他们的方法是概率模型 就先放着了 (所以cancerlocator.py没写好 先不要下载）



## No.2 pancreatic cancer predictor

[文献](https://www.tandfonline.com/doi/full/10.1080/15592294.2019.1568178?scroll=top&needAccess=true)

没给code 

用的GEO公开数据

>All data used in this study are available in TCGA Data Portal (https://tcga-data.nci.nih.gov) and GEO (https://www.ncbi.nlm.nih.gov/geo/) with accession numbers of GSE69914, GSE76938, GSE48684, GSE73549, GSE65820, GSE66836, GSE89852, GSE58999 and GSE38240

我昨天按他们说的写了一个demo 但是他们指路的数据下载下来有几十个G 操作起来实在太慢了 就先放着了

## No.3 prostate cancer predictor

[文献](https://bmccancer.biomedcentral.com/articles/10.1186/s12885-017-3252-2#Sec2)

没给code 

用的[GEO公开数据](http://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE76938)

我用线性模型跑了一下和他们的结果差不多 选他们给出的特征的话在我电脑上得到的AUC（0.96）和他们的（0.97）很接近  不过coefficients不太对

# Tissue predictor


