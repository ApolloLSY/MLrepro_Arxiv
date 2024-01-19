
# Summary

![image](https://github.com/ApolloLSY/MLrepro_Arxiv/assets/78656349/5fda8fde-16e8-4945-904f-8bb9af493e02)

![image](https://github.com/ApolloLSY/MLrepro_Arxiv/assets/78656349/18ea2dbd-7b5a-40cd-9626-fcbc50c51587)

Using the above search terms, research papers were selected from 51 and 27 articles with citation counts exceeding 35 and utilizing 5mc Array. Subsequently, reproduction will be conducted.

| Article                                      | Function            | Author                  | Year     | Citations| Code                                               | Data                                          | Model                     |
|----------------------------------------------|-----------------------------------|-----------------|------|--|----------------------------------------------------|------------------------------------------------|---------------------------|
| [Integrative analysis](https://www.tandfonline.com/doi/full/10.1080/15592294.2019.1568178?scroll=top&needAccess=true) | Pancreatic Cancer Predictor     | Wubin DingCenter  | 2019 |71| Unpublished                                        | Unpublished                                    | Logistic Regression       |
| [Genome-wide DNA methylation measurements](https://bmccancer.biomedcentral.com/articles/10.1186/s12885-017-3252-2) | Prostate Cancer Predictor       | Marie K. Kirby   | 2017 |39| Unpublished                                        | [GEO Public Data](http://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE76938) | Linear Mixed Model         |
| [Epigenetic prediction of complex traits and death](https://genomebiology.biomedcentral.com/articles/10.1186/s13059-018-1514-1) | Common Traits Predictor         | Daniel L. McCartney | 2018 | 96|Unpublished                                        | Generation Scotland (Need approval from GS Access Committee) | Logistic Regression       |
| [DNA methylation as a predictor of fetal alcohol spectrum disorder](https://clinicalepigeneticsjournal.biomedcentral.com/articles/10.1186/s13148-018-0439-6) | Predictor of Fetal Alcohol Spectrum Disorder | Alexandre A. Lussier | 2018 |37| Unpublished                                        | GSE50759                                       | Stochastic Gradient Boosting |
| [CancerLocator](https://genomebiology.biomedcentral.com/articles/10.1186/s13059-017-1191-5?site=n_detail_link) | Multiple Cancer Predictor        | Shuli Kang       | 2017 |168| [GitHub - JAVA](https://github.com/jasminezhoulab/CancerLocator/tree/v1.0.1) | Simulated data generated by themselves          | Probability Model         |
| [tissue predictor](https://doi.org/10.1093/nar/gkt1380) | tissue predictor    |Baoshan Ma|2014     |   2014 |101| [unpublished | ECACC](http://www.hpacultures.org.uk/collections/ecacc.jsp)   | Linear prediction model&Support vector machine&Cross-validation        |



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




## ~~Cancerlocator （cell free dna）~~

| Tags                                      | Content   |
|--------------------------------------------|-------|
| function   | multiple cancer predictor  | 
| source   | [CancerLocator: non-invasive cancer diagnosis and tissue-of-origin prediction using methylation profiles of cell-free DNA](https://genomebiology.biomedcentral.com/articles/10.1186/s13059-017-1191-5?site=n_detail_link#Sec7) Shuli Kang(2017)   | 
| code   | [github-JAVA](https://github.com/jasminezhoulab/CancerLocator/tree/v1.0.1)    | 
| data   | simulated data generated by themselves  | 
| model  | probability model  | 


# Tissue predictor

## 

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
