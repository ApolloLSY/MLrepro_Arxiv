# Disease predictor

## ~~No.1 Cancerlocator （cell free dna 不是我们要的）~~

| 属性                                      | 内容   |
|--------------------------------------------|-------|
| 内容   | 预测多个癌症  | 
| 来源   | [文献](https://genomebiology.biomedcentral.com/articles/10.1186/s13059-017-1191-5?site=n_detail_link#Sec7)   | 
| code   | [github-JAVA](https://github.com/jasminezhoulab/CancerLocator/tree/v1.0.1)    | 
| data   | 自己产生的模拟数据  | 
| model  | probability model  | 

我前天读完了他们写的几个类中的两个 还有两个类没读完 感觉内容太多了 而且他们的方法是概率模型 就先放着了 (所以cancerlocator.py没写好 先不要下载）



## No.2 pancreatic cancer predictor

| 1                                      | 2   |
|--------------------------------------------|-------|
| 内容   | 预测是否有胰腺癌  | 
| 来源   | [文献](https://www.tandfonline.com/doi/full/10.1080/15592294.2019.1568178?scroll=top&needAccess=true) | 
| code   | 未公开   | 
| data   | GEO公开数据  | 
| model  | logistic regression  |

>All data used in this study are available in TCGA Data Portal (https://tcga-data.nci.nih.gov) and GEO (https://www.ncbi.nlm.nih.gov/geo/) with accession numbers of GSE69914, GSE76938, GSE48684, GSE73549, GSE65820, GSE66836, GSE89852, GSE58999 and GSE38240

我昨天按他们说的写了一个demo 但是他们指路的数据下载下来有几十个G 操作起来实在太慢了 就先放着了

## No.3 prostate cancer predictor

| 1                                      | 2   |
|--------------------------------------------|-------|
| 内容   | 预测是否有前列腺癌  | 
| 来源   | [文献](https://bmccancer.biomedcentral.com/articles/10.1186/s12885-017-3252-2) | 
| code   | 未公开   | 
| data   | [GEO公开数据](http://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE76938)  | 
| model  | linear mixed model  |

我用线性模型跑了一下和他们的结果差不多，选他们给出的特征在我电脑上得到的AUC（0.96）和他们的（0.97）很接近，不过coefficients不太对。

## No.4 common traits predictor

| 1                                      | 2   |
|--------------------------------------------|-------|
| 内容   | 预测多个traits | 
| 来源   | [文献](https://genomebiology.biomedcentral.com/articles/10.1186/s13059-018-1514-1) | 
| code   | 未公开   | 
| data   | Generation Scotland（但获取需要GS Access Committee审查）  | 
| model  | logistic regression  |


traits处理的时候全是二分的，所以每一个都是logistic regression
>there were 652 controls and 242 cases for obesity, 745 light-to-moderate drinkers and 150 heavy drinkers, 418 non-smokers and 102 current smokers, and 229 and 666 individuals with > 11 and ≤ 11 years of full-time education, respectively. Following dichotomization of the cholesterol-related variables, there were 531 and 354 individuals with high and low total cholesterol, respectively; 89 and 723 individuals with high and low HDL cholesterol, respectively; 637 and 175 individuals with high and low LDL with remnant cholesterol, respectively; and 307 and 502 with high and low total:HDL cholesterol ratios, respectively.

供对照的结果：

| Trait                                      | AUC   | 95% Confidence Interval (CI)       |
|--------------------------------------------|-------|-------------------------------------|
| Current Smokers vs. non-smokers            | 0.98  | 0.97–1.00 (Fig. 2)                  |
| Obesity vs. Non-obesity                    | 0.67  | 0.63–0.71                           |
| High HDL vs. Low HDL                       | 0.70  | 0.64–0.75                           |
| Light-to-Moderate Drinkers vs. Heavy Drinkers| 0.73  | 0.69–0.78                           |
| > 11 vs. ≤ 11Years of Full-time Education    | 0.59  | 0.55–0.63 (Fig. 2)                  |
| Total Cholesterol High vs. Low              | 0.61  | 0.57–0.64                           |
| High LDL vs. Low LDL                        | 0.53  | 0.48–0.58                           |
| TC:HDL ratio High vs. Low               | 0.61  | 0.57–0.65                           |

# Tissue predictor



