
# Summary

![image](https://github.com/ApolloLSY/MLrepro_Arxiv/assets/78656349/5fda8fde-16e8-4945-904f-8bb9af493e02)

![image](https://github.com/ApolloLSY/MLrepro_Arxiv/assets/78656349/18ea2dbd-7b5a-40cd-9626-fcbc50c51587)

Using the above search terms, research papers were selected from 51 and 27 articles with citation counts exceeding 35 and utilizing 5mc Array. Subsequently, reproduction will be conducted.

|number| Article                                      | Function            | Author                  | Year     | Citations| Code                                               | Data                                          | Model                     |
|----------------------------------------------|----------------------------------------------|-----------------------------------|-----------------|------|--|----------------------------------------------------|------------------------------------------------|---------------------------|
| [Potential DNA methylation biomarkers for pan-cancer diagnosis and prognosis](https://www.tandfonline.com/doi/full/10.1080/15592294.2019.1568178?scroll=top&needAccess=true) | Pancreatic Cancer Predictor     | Wubin DingCenter  | 2019 |71| Unpublished                                        | Unpublished                                    | Logistic Regression       |
| [DNA methylation measurements in prostate tissues](https://bmccancer.biomedcentral.com/articles/10.1186/s12885-017-3252-2) | Prostate Cancer Predictor       | Marie K. Kirby   | 2017 |39| Unpublished                                        | [GEO Public Data](http://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE76938) | Linear Mixed Model         |
| [Epigenetic prediction of complex traits and death](https://genomebiology.biomedcentral.com/articles/10.1186/s13059-018-1514-1) | Common Traits Predictor         | Daniel L. McCartney | 2018 | 96|Unpublished                                        | Generation Scotland (Need approval from GS Access Committee) | Logistic Regression       |
| [DNA methylation as a predictor of fetal alcohol spectrum disorder](https://clinicalepigeneticsjournal.biomedcentral.com/articles/10.1186/s13148-018-0439-6) | Predictor of Fetal Alcohol Spectrum Disorder | Alexandre A. Lussier | 2018 |37| Unpublished                                        | GSE50759                                       | Stochastic Gradient Boosting |
| [Epigenetic prediction of coronary heart disease](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0190549#sec002)   | predictor of coronary heart disease  | Meeshanthini V. Dogan  | 2018 |75| unpublished                                          | [dbGAP](https://dbgap.ncbi.nlm.nih.gov)          | Random Forest classification model  |
| [EWAS of metabolic syndrome and its components](https://www.nature.com/articles/s41598-020-77506-z#Abs1) | Predictor of metabolic syndrome  | Marja-Liisa Nuotio (2020) | 2020 | 25 | Unpublished                                          | NFBC1966 (needing request and permission from THL Biobank and University of Oulu) | Random Forest classification model  |
| [EWAS of Incident Type 2 Diabetes in a British Population](https://diabetesjournals.org/diabetes/article/68/12/2315/39835/Epigenome-Wide-Association-Study-of-Incident-Type) | Predictor of Type 2 diabetes  | Alexia Cardona (2019) | 2019 | 44 | Unpublished                                          | EPIC-Norfolk (needing request and permission)  | Logistic regression model  |
| [EWAS of Cardiovascular Disease Risk and Interactions](https://www.ahajournals.org/doi/10.1161/JAHA.119.015299) | Cardiovascular Disease predictor  | Kenneth Westerman (2020) | 2020 | 21 | [GitHub](https://github.com/kwest​erman/​meth_cvd)  | LBC 1936  | Study-specific regression models: CSL cross-study learner  |
| [potential blood biomarkers for Parkinson’s disease](https://clinicalepigeneticsjournal.biomedcentral.com/articles/10.1186/s13148-019-0621-5) | Parkinson Disease predictor  | Changliang Wang (2020) | 2020 | 37 | Unpublished                                          | [DNA methylation data](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE111629 )  | Linear model  |
| [predict Alzheimer’s disease progression using temporal DNA methylation data in peripheral blood](https://www.sciencedirect.com/science/article/pii/S2001037022004639?via%3Dihub#s0010) | Alzheimer’s disease predictor  | Li Chen (2022) | 2022 | 2 | [Published](https://github.com/lichen-lab/MTAE)  | ADNI public data  | Unsupervised dimensionality reduction model  |
| [Evaluation and prediction of late-onset Alzheimer’s disease](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0248375) | Another Alzheimer’s disease predictor  | Ray O. Bahado-Singh (2021) | 2021 | 19 | Unpublished                                          | Unpublished  | DL model  |
| [DNA methylation signatures of dementia risk](https://alz-journals.onlinelibrary.wiley.com/doi/full/10.1002/dad2.12078) | Dementia predictor  | Rosie M. Walker | 2020 | 3 | Unpublished                                          | Unpublished  | Linear regression model |
| [DNA methylation signature predict colorectal cancer susceptibility](https://bmccancer.biomedcentral.com/articles/10.1186/s12885-020-07194-5#Sec2) | Colorectal Cancer predictor  | Justina Ucheojor Onwuka | 2020 | 11 | Unpublished                                          | GSE51032  | Stepwise logistic regression |
| [DNA methylation age of blood predicts future onset of lung cancer in the women's health initiative](https://www.aging-us.com/article/100809/text) | Female Lung Cancer predictor  | Morgan E. Levine | 2015 | 11 | Unpublished                                          | Women's Health Initiative  | Logistic regression models and ran Cox models |
| [DNA methylation profiles for predicting breast cancer risk](https://febs.onlinelibrary.wiley.com/doi/10.1002/1878-0261.12594) | Breast Cancer predictor  | Zhong Guan | 2019 | 9 | Unpublished                                          | Unpublished  | Multivariable logistic regression |
| [EWAS Indicates Hypomethylation of MTRNR2L8 in Large-Artery Atherosclerosis Stroke](https://www.ahajournals.org/doi/10.1161/STROKEAHA.118.023436) | Atherosclerosis Stroke predictor  | Yupei Shen | 2019 | 27 | Unpublished                                          | Unpublished  | Linear regression |
| [DNA Epigenomic Prediction of Cerebral Palsy](https://www.mdpi.com/1422-0067/20/9/2075) | Cerebral Palsy predictor  | Ray O. Bahado-Singh | 2019 | 15 | Unpublished                                          | Unpublished  | Univariate logistic regression |
| [Epigenetic Signatures of Cigarette Smoking](https://www.ahajournals.org/doi/10.1161/CIRCGENETICS.116.001506) | Smoking Habits Examination  | Roby Joehanes | 2016 | 508 | Unpublished                                          | 16 cohorts of the Cohorts for Heart and Aging Research in Genetic Epidemiology Consortium  | Linear regression |
| [CancerLocator](https://genomebiology.biomedcentral.com/articles/10.1186/s13059-017-1191-5?site=n_detail_link) | Multiple Cancer Predictor        | Shuli Kang       | 2017 |168| [GitHub - JAVA](https://github.com/jasminezhoulab/CancerLocator/tree/v1.0.1) | Simulated data generated by themselves          | Probability Model         |
| [tissue predictor](https://doi.org/10.1093/nar/gkt1380) | tissue predictor    |Baoshan Ma    |   2014 |101| unpublished | [ECACC](http://www.hpacultures.org.uk/collections/ecacc.jsp)   | Linear prediction model&Support vector machine&Cross-validation        |



# Disease predictor

## pancreatic cancer predictor


| Tags                                      | Content   |
|--------------------------------------------|-------|
| function   | pancreatic cancer predictor  | 
|  source  | [Integrative analysis identifies potential DNA methylation biomarkers for pan-cancer diagnosis and prognosis](https://www.tandfonline.com/doi/full/10.1080/15592294.2019.1568178?scroll=top&needAccess=true)  Wubin DingCenter(2019) | 
| code   | unpublished   | 
| data   | unpublished  | 
| model  | logistic regression  |

>All data used in this study are available in TCGA Data Portal (https://tcga-data.nci.nih.gov) and GEO (https://www.ncbi.nlm.nih.gov/geo/) with accession numbers of GSE69914, GSE76938, GSE48684, GSE73549, GSE65820, GSE66836, GSE89852, GSE58999 and GSE38240

logistic regression based on the selected seven features was used to construct the tumor-normal diagnostic model. OneVsRestClassifier with estimator of logistic regression was employed to train the tumor specific multiclass classifier. They randomly split the full dataset into training and test sets with 4:1 ratio in both tumor-normal diagnostic model and tissue-specific classifier.

## prostate cancer predictor

| Tags                                      | Content   |
|--------------------------------------------|-------|
| function   | prostate cancer predictor  | 
|  source  | [Genome-wide DNA methylation measurements in prostate tissues uncovers novel prostate cancer diagnostic biomarkers and transcription factor binding patterns](https://bmccancer.biomedcentral.com/articles/10.1186/s12885-017-3252-2)  Marie K. Kirby(2017) | 
| code   | unpublished   | 
| data   | [GEO public data](http://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE76938)  | 
| model  | linear mixed model  |

They calculated the methylation beta score as: b = IntensityMethylated/(IntensityMethylated + IntensityUnmethylated) and converted data points that were not significant above background intensity to NAs.  CpGs having greater than 10% missing values prior to normalization were removed. CpGs with a standard deviation of less than 1% across samples were removed. Linear mixed model analysis of the methylation data was performed with patient as a random effect, and age and ethnicity as fixed effects.The p-values were adjusted using the Benjamini and Hochberg method

our AUC（0.96）is similar to their AUC（0.97）

## common traits predictor

| Tags                                      | Content   |
|--------------------------------------------|-------|
| function   | common traits predictor  | 
| source   | [Epigenetic prediction of complex traits and death](https://genomebiology.biomedcentral.com/articles/10.1186/s13059-018-1514-1) Daniel L. McCartney（2018）| 
| code   |unpublished   | 
| data   | Generation Scotland（need approval from GS Access Committee）  | 
| model  | LASSO regression&linear mixed model  |





They model the traits of interest as the outcomes and the CpGs as the predictors and train optimized predictors using penalized regression methods, then apply these predictors to approximately 900 individuals to determine: 

(1) the proportion of variance the DNAm predictors explain in the outcomes;

(2) the extent to which these proportions are independent from the contribution of genetics; 

(3) the accuracy with which the DNAm predictors can identify obese individuals, college-educated individuals, heavy drinkers, high cholesterol levels, and current smokers if provided with a random DNA sample from the population; 

(4) the extent to which they can predict health outcomes, such as mortality, and if they do so independently from the phenotypic measure.

Methylation preparation:
>Filtering for outliers, sex mismatches, non-blood samples, poorly detected probes, and samples was performed(Additional file 4). CpGs with missing values, non-autosomal and non-CpG sites, and any sites not present on the Illumina 450 k array was removed.

LASSO regression in Generation Scotland:
>Penalized regression models were run using the glmnet library in R. Tenfold cross-validation was applied and the mixing parameter (alpha) was set to 1 to apply a LASSO penalty. Coefficients for the model with the lambda value corresponding to the minimum mean cross-validated error were extracted and applied to the corresponding CpGs in an out of sample prediction cohort to create the DNAm predictors.

Prediction analysis in the Lothian Birth Cohort 1936:
>Area under the curve (AUC) estimates were estimated for binary categorizations of traits. Ordinal logistic regression was used for the categorical smoking variable (never, ex, current smoker). ROC curves were developed for smoking status, obesity, high/low alcohol consumption, college education and cholesterol variables, and AUC estimates were estimated for binary categorizations of these variables using the pROC library in R.Sex was included as a covariate in all models. Correction for multiple testing was applied using the Bonferroni method.

DNAm predictors classify phenotype extremes
>652 controls and 242 cases for obesity, 745 light-to-moderate drinkers and 150 heavy drinkers, 418 non-smokers and 102 current smokers, 229 and 666 individuals with > 11 and ≤ 11 years of full-time education, 531 and 354 individuals with high and low total cholesterol,  89 and 723 individuals with high and low HDL cholesterol,  637 and 175 individuals with high and low LDL with remnant cholesterol, 307 and 502 with high and low total:HDL cholesterol ratios.

Results for comparison:

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


## fetal alcohol spectrum disorder predictor

| Tags                                      | Content   |
|--------------------------------------------|-------|
| function   | predictor of fetal alcohol spectrum disorder  | 
| source   | [DNA methylation as a predictor of fetal alcohol spectrum disorder](https://clinicalepigeneticsjournal.biomedcentral.com/articles/10.1186/s13148-018-0439-6) Alexandre A. Lussier（2018）| 
| code   |unpublished   | 
| data   | GSE50759  | 
| model  |stochastic gradient boosting  |

filter:(1) probes on X and Y chromosomes (n = 11,648), (2) SNP probes (n = 65), (3) probes with bead count < 3 in 10% of samples (n = 726), (4) probes with 10% of samples with a detection p value > 0.01 (n = 11,864), and (5) probes with a polymorphic CpG and non-specific probes (N = 19,337 SNP-CpG and 10,484 non-specific probes)
linear modeling was performed on the 648 differentially methylated probes identified in the initial NDN study and found in the present dataset using the limma package in R and a model that included clinical status and all identified SVs as covariates.Benjamini-Hochberg method was used.

## coronary heart disease predictor

| Tags                                      | Content   |
|--------------------------------------------|-------|
| function   | predictor of coronary heart disease  | 
| source   | [Integrated genetic and epigenetic prediction of coronary heart disease in the Framingham Heart Study](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0190549#sec002) Meeshanthini V. Dogan（2018）| 
| code   |unpublished   | 
| data   | [dbGAP] (https://dbgap.ncbi.nlm.nih.gov)  | 
| model  |Random Forest classification model  |

To reduce the number of DNA methylation loci:
first, the correlation was calculated between the 472,822 CpG sites and CHD status. CpG sites were retained if the point bi-serial correlation with CHD was at least 0.1. A total of 138,815 CpG sites remained. 
Subsequently, Pearson correlation between those 138,815 sites was calculated. If the Pearson correlation between two loci was at least 0.8, the loci with a smaller point bi-serial correlation (i.e. less correlated with CHD) was discarded. In the end, 107,799 DNA methylation loci (~23%) remained for model training
 A grid search using GridSearchCV was employed to perform 10-fold cross-validation hyper-parameter tuning (maximum features: auto, minimum samples for each split: 2–10, information gain criterion: entropy or gini, maximum tree depth: 500–2500, number of trees: 10000–30000) of the models.All eight final tuned models were saved for testing on the test dataset where majority voting was used to ensemble the votes of these models.

 ## metabolic syndrome predictor

| Tags                                      | Content   |
|--------------------------------------------|-------|
| function   | predictor of coronary heart disease  | 
| source   | [An epigenome-wide association study of metabolic syndrome and its components](https://www.nature.com/articles/s41598-020-77506-z#Abs1) Marja-Liisa Nuotio（2020）| 
|citations|25|
| code   |unpublished   | 
| data   | NFBC1966 (needing request and permission from THL Biobank and University of Oulu | 
| model  |Random Forest classification model  |

using regression analysis fitting generalised linear models.Analyses were adjusted for age, sex, smoking status (defined either as current or never/ex-smokers), alcohol consumption (grams/week), cell subtype proportion, study- specific technical covariates (described in detail in Supplementary material), and the first five genetic principal components of the data to control for potential population substructure.

 ## Type 2 diabetes predictor

| Tags                                      | Content   |
|--------------------------------------------|-------|
| function   | predictor of Type 2 diabetes  | 
| source   | [Epigenome-Wide Association Study of Incident Type 2 Diabetes in a British Population: EPIC-Norfolk Study](https://diabetesjournals.org/diabetes/article/68/12/2315/39835/Epigenome-Wide-Association-Study-of-Incident-Type) Alexia Cardona（2019）| 
|citations|44|
| code   |unpublished   | 
| data   | EPIC-Norfolk (needing request and permission  | 
| model  |logistic regression model  |

 ## Cardiovascular Disease predictor

| Tags                                      | Content   |
|--------------------------------------------|-------|
| function   | Cardiovascular Disease predictor | 
| source   | [Epigenomic Assessment of Cardiovascular Disease Risk and Interactions With Traditional Risk Metrics](https://www.ahajournals.org/doi/10.1161/JAHA.119.015299) Kenneth Westerman（2020）| 
|citations|21|
| code   |[github](https://github.com/kwest​erman/​meth_cvd)   | 
| data   | LBC 1936  | 
| model  |study‐specific regression models:CSL cross‐study learner  |

 ## Parkinson Disease predictor

| Tags                                      | Content   |
|--------------------------------------------|-------|
| function   | Parkinson  Disease predictor | 
| source   | [Identification of potential blood biomarkers for Parkinson’s disease by gene expression and DNA methylation data integration analysis](https://clinicalepigeneticsjournal.biomedcentral.com/articles/10.1186/s13148-019-0621-5) Changliang Wang（2020）| 
|citations|37|
| code   |unpublished   | 
| data   | [DNA methylation data](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE111629 )  | 
| model  | linear model  |


 ## Alzheimer’s disease predictor

| Tags                                      | Content   |
|--------------------------------------------|-------|
| function   |Alzheimer’s diseasepredictor | 
| source   | [Multi-task deep autoencoder to predict Alzheimer’s disease progression using temporal DNA methylation data in peripheral blood ](https://www.sciencedirect.com/science/article/pii/S2001037022004639?via%3Dihub#s0010)Li Chen（2022）| 
|citations|2|
| code   |[published](https://github.com/lichen-lab/MTAE)   | 
| data   | ADNI public data  | 
| model  | unsupervised dimensionality reduction model  |

 ## another Alzheimer’s disease predictor

| Tags                                      | Content   |
|--------------------------------------------|-------|
| function   |Alzheimer’s diseasepredictor | 
| source   | [Artificial intelligence and leukocyte epigenomics: Evaluation and prediction of late-onset Alzheimer’s disease](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0248375)Ray O. Bahado-Singh（2021）| 
|citations|19|
| code   |unpublished | 
| data   | unpublished  | 
| model  | DL model  |

 ## dementia predictor

| Tags                                      | Content   |
|--------------------------------------------|-------|
| function   |dementia predictor | 
| source   | [Epigenome-wide analyses identify DNA methylation signatures of dementia risk](https://alz-journals.onlinelibrary.wiley.com/doi/full/10.1002/dad2.12078)Rosie M. Walker（2020）| 
|citations|3|
| code   |unpublished | 
| data   | unpublished  | 
| model  | linear regression model |

 ## colorectal cancer predictor

| Tags                                      | Content   |
|--------------------------------------------|-------|
| function   |colorectal cancer predictor | 
| source   | [A panel of DNA methylation signature from peripheral blood may predict colorectal cancer susceptibility](https://bmccancer.biomedcentral.com/articles/10.1186/s12885-020-07194-5#Sec2)Justina Ucheojor Onwuka（2020）| 
|citations|11|
| code   |unpublished | 
| data   | GSE51032  | 
| model  | stepwise logistic regression |

 ## female lung cancer predictor

| Tags                                      | Content   |
|--------------------------------------------|-------|
| function   |female lung cancer | 
| source   | [DNA methylation age of blood predicts future onset of lung cancer in the women's health initiative](https://www.aging-us.com/article/100809/text)Morgan E. Levine（2015）| 
|citations|11|
| code   |unpublished | 
| data   |Women's Health Initiative | 
| model  | logistic regression models and ran Cox models |

 ## breast cancer predictor

| Tags                                      | Content   |
|--------------------------------------------|-------|
| function   |breast cancer predictor | 
| source   | [Individual and joint performance of DNA methylation profiles, genetic risk score and environmental risk scores for predicting breast cancer risk](https://febs.onlinelibrary.wiley.com/doi/10.1002/1878-0261.12594)Zhong Guan（2019）| 
|citations|9|
| code   |unpublished | 
| data   |unpublished  | 
| model  | multivariable logistic regression |

 ## Atherosclerosis Stroke cancer predictor

| Tags                                      | Content   |
|--------------------------------------------|-------|
| function   |Atherosclerosis Stroke predictor | 
| source   | [Epigenome-Wide Association Study Indicates Hypomethylation of MTRNR2L8 in Large-Artery Atherosclerosis Stroke](https://www.ahajournals.org/doi/10.1161/STROKEAHA.118.023436)Yupei Shen（2019）| 
|citations|27|
| code   |unpublished | 
| data   |unpublished  | 
| model  | linear regression |

 ## Cerebral Palsy predictor

| Tags                                      | Content   |
|--------------------------------------------|-------|
| function   |Cerebral Palsy predictor | 
| source   | [Deep Learning/Artificial Intelligence and Blood-Based DNA Epigenomic Prediction of Cerebral Palsy](https://www.mdpi.com/1422-0067/20/9/2075)Ray O. Bahado-Singh（2019）| 
|citations|15|
| code   |unpublished | 
| data   |unpublished  | 
| model  | univariate logistic regression |


 ## smoking habits examination

| Tags                                      | Content   |
|--------------------------------------------|-------|
| function   |smoking habits examination | 
| source   | [Epigenetic Signatures of Cigarette Smoking](https://www.ahajournals.org/doi/10.1161/CIRCGENETICS.116.001506)Roby Joehanes（2016）| 
|citations|508|
| code   |unpublished | 
| data   |16 cohorts of the Cohorts for Heart and Aging Research in Genetic Epidemiology Consortium  | 
| model  |linear regression |



## ~~Cancerlocator （cell free dna）~~

| Tags                                      | Content   |
|--------------------------------------------|-------|
| function   | multiple cancer predictor  | 
| source   | [CancerLocator: non-invasive cancer diagnosis and tissue-of-origin prediction using methylation profiles of cell-free DNA](https://genomebiology.biomedcentral.com/articles/10.1186/s13059-017-1191-5?site=n_detail_link#Sec7) Shuli Kang(2017)   | 
| code   | [github-JAVA](https://github.com/jasminezhoulab/CancerLocator/tree/v1.0.1)    | 
| data   | simulated data generated by themselves  | 
| model  | probability model  | 


# Tissue predictor


| Tags                                      | Content   |
|--------------------------------------------|-------|
| function   |tissue predictor  | 
| source   | [Predicting DNA methylation level across human tissues](https://doi.org/10.1093/nar/gkt1380) Baoshan Ma(2014)   | 
| code   | unpublished   | 
| data   | [ECACC](http://www.hpacultures.org.uk/collections/ecacc.jsp)  | 
| model  | Linear prediction model&Support vector machine&Cross-validation| 



normalization:
>HumanMethylation450 array, we used the pipeline developed by Touleimat and Tost (21). Individual data points with detection P > 0.01 or number of beads <3 were treated as missing data. Samples with >20% missing probes were treated as missing data. 
linear model was developed.
SVM model is constructed in a similar manner as the linear regression method. For a given CpG site j, we used xij and yij as the training data set to build an SVM model.[their R package](http://www.hsph.harvard.edu/liming-liang/cross-tissue-methylation/).


# Project Progress and Plan

## Overview
Reproduce the "Tissue Predictor" and "Disease Predictor" models. This document will track the progress and detailed plan of the project.

## Progress Overview

## Task List
### 1. Literature Review
- [ ] Read papers and select suitable ones (high quality & reproducible).

### 2. Data Collection and Preparation
- [ ] Download datasets for reproduction.
- [ ] Data preprocessing: cleaning, standardizing, and splitting the dataset.

### 3. Model Replication
#### 3.1 Disease Predictor
##### 3.1.1 prostate Predictor
- [ ] Download original code or model parameters.
- [ ] Implement the Disease Predictor model.
- [ ] Train and test on the original literature dataset (if available).
- [ ] Evaluate the performance of the Predictor model using literature appendix materials.

##### 3.1.2 pancreat cancer Predictor
- [ ] Download original code or model parameters.
- [ ] Implement the Disease Predictor model.
- [ ] Train and test on the original literature dataset (if available).
- [ ] Evaluate the performance of the Predictor model using literature appendix materials.

##### 3.1.3 conmmon traits Predictor
- [ ] Download original code or model parameters.
- [ ] Implement the Disease Predictor model.
- [ ] Train and test on the original literature dataset (if available).
- [ ] Evaluate the performance of the Predictor model using literature appendix materials.

##### 3.1.4 fetal alcohol spectrum disorder Predictor
- [ ] Download original code or model parameters.
- [ ] Implement the Disease Predictor model.
- [ ] Train and test on the original literature dataset (if available).
- [ ] Evaluate the performance of the Predictor model using literature appendix materials.

#### 3.2 Tissue Predictor
- [ ] Download original code or model parameters.
- [ ] Implement the Tissue Predictor model.
- [ ] Train and test on the original literature dataset (if available).
- [ ] Evaluate the performance of the Predictor model using literature appendix materials.

### 4. Application to Our Data
- [ ] Application 3.1.1
- [ ] Application 3.1.2
- [ ] Application 3.1.3
- [ ] Application 3.1.4
- [ ] Application 3.2
- [ ] Compare differences between different models and our model.

## Timeline


| Phase                     | Task                                                                      | Deadline   |
|---------------------------|---------------------------------------------------------------------------|------------|
| 1. Literature Review      | Read papers and select suitable ones                                      | 1.11-1.13  |
| 2. Data Collection        | Download datasets for reproduction                                        | 1.14  |
|                           | Data preprocessing: cleaning, standardizing, and splitting the dataset    | 1.15       |
| 3. Model Replication      | 3.1 Disease Predictor                                                   |       1.16     |
|                           |   3.1.1 Prostate Predictor                                              |   1.16       |
|                           |     - [ ] Download original code or model parameters                    | 1.16      |
|                           |     - [ ] Implement the Disease Predictor model                          | 1.16      |
|                           |     - [ ] Train and test on the original literature dataset (if available)| 1.16       |
|                           |   3.1.2 Pancreatic Cancer Predictor                                    |      1.17      |
|                           |     - [ ] Download original code or model parameters                    |1.17  |
|                           |     - [ ] Implement the Disease Predictor model                          | 1.17      |
|                           |     - [ ] Train and test on the original literature dataset (if available)| 1.17       |
|                           |     - [ ] Evaluate the performance of the Predictor model using literature appendix materials | 1.17 |
|                           |   3.1.3 Common Traits Predictor                                        |      1.18      |
|                           |     - [ ] Download original code or model parameters                    | 1.18       |
|                           |     - [ ] Implement the Disease Predictor model                          | 1.18       |
|                           |     - [ ] Train and test on the original literature dataset (if available)| 1.18      |
|                           |     - [ ] Evaluate the performance of the Predictor model using literature appendix materials | 1.18 |
|                           |   3.1.4 Fetal Alcohol Spectrum Disorder Predictor                      |       1.19     |
|                           |     - [ ] Download original code or model parameters                    | 1.19      |
|                           |     - [ ] Implement the Disease Predictor model                          | 1.19      |
|                           |     - [ ] Train and test on the original literature dataset (if available)| 1.19      |
|                           |     - [ ] Evaluate the performance of the Predictor model using literature appendix materials | 1.19 |
|                           | 3.2 Tissue Predictor                                                    |      1.20      |
|                           |   - [ ] Download original code or model parameters                      | 1.20      |
|                           |   - [ ] Implement the Tissue Predictor model                             |1.20    |
|                           |   - [ ] Train and test on the original literature dataset (if available) | 1.20     |
|                           |   - [ ] Evaluate the performance of the Predictor model using literature appendix materials | 1.20|
| 4. Application to Our Data | - [ ] Application 3.1.1                                                | 1.21    |
|                           | - [ ] Application 3.1.2                                                | 1.21         |
|                           | - [ ] Application 3.1.3                                                | 1.21        |
|                           | - [ ] Application 3.1.4                                                | 1.22        |
|                           | - [ ] Application 3.2                                                  | 1.22      |
|                           | - [ ] Compare differences between different models and our model       | 1.22      |

Taking into account potential unexpected emergencies or unplanned situations, there is a 2-day buffer. The project is scheduled to be completed by January 24th.
