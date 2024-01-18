



# Summary

| Article                                      | Function            | Author                  | Year     | Citations| Code                                               | Data                                          | Model                     |
|----------------------------------------------|-----------------------------------|-----------------|------|--|----------------------------------------------------|------------------------------------------------|---------------------------|
| [Integrative analysis](https://www.tandfonline.com/doi/full/10.1080/15592294.2019.1568178?scroll=top&needAccess=true) | Pancreatic Cancer Predictor     | Wubin DingCenter  | 2019 |71| Unpublished                                        | Unpublished                                    | Logistic Regression       |
| [Genome-wide DNA methylation measurements](https://bmccancer.biomedcentral.com/articles/10.1186/s12885-017-3252-2) | Prostate Cancer Predictor       | Marie K. Kirby   | 2017 |39| Unpublished                                        | [GEO Public Data](http://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE76938) | Linear Mixed Model         |
| [Epigenetic prediction of complex traits and death](https://genomebiology.biomedcentral.com/articles/10.1186/s13059-018-1514-1) | Common Traits Predictor         | Daniel L. McCartney | 2018 | 96|Unpublished                                        | Generation Scotland (Need approval from GS Access Committee) | Logistic Regression       |
| [CancerLocator](https://genomebiology.biomedcentral.com/articles/10.1186/s13059-017-1191-5?site=n_detail_link) | Multiple Cancer Predictor        | Shuli Kang       | 2017 |168| [GitHub - JAVA](https://github.com/jasminezhoulab/CancerLocator/tree/v1.0.1) | Simulated data generated by themselves          | Probability Model         |
| [tissue predictor](https://doi.org/10.1093/nar/gkt1380) | Multiple Cancer Predictor        | tissue predictor       | 2014 |101| [? | ?   | ?        |




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


## prostate cancer predictor

| Tags                                      | Content   |
|--------------------------------------------|-------|
| function   | prostate cancer predictor  | 
|  source  | [Genome-wide DNA methylation measurements in prostate tissues uncovers novel prostate cancer diagnostic biomarkers and transcription factor binding patterns](https://bmccancer.biomedcentral.com/articles/10.1186/s12885-017-3252-2)  Marie K. Kirby(2017) | 
| code   | unpublished   | 
| data   | [GEO public data](http://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE76938)  | 
| model  | linear mixed model  |
our AUC（0.96）is similar to their AUC（0.97）

## common traits predictor

| Tags                                      | Content   |
|--------------------------------------------|-------|
| function   | common traits predictor  | 
| source   | [Epigenetic prediction of complex traits and death](https://genomebiology.biomedcentral.com/articles/10.1186/s13059-018-1514-1) Daniel L. McCartney（2018）| 
| code   |unpublished   | 
| data   | Generation Scotland（need approval from GS Access Committee）  | 
| model  | logistic regression  |




Through large meta-analysis projects, methylation signals at individual CpG sites have been associated with educational attainment, smoking, alcohol consumption, cholesterol levels, and BMI [5,6,7,8,9,10,11,12,13]. Such studies have also used methylation predictors (from a combination of CpG sites) to predict the phenotype of interest in independent cohorts. For example, 7% of the variance in BMI and 2% of the variance in educational attainment can be explained by their respective predictors [5, 14]. Moreover, DNA methylation has been reported to explain 0.74% and 9.51% of the variation in total and high-density lipoprotein (HDL) cholesterol levels, respectively [11].
>We model the traits of interest as the outcomes and the CpGs as the predictors and train optimized predictors using penalized regression methods. We then apply these predictors to an independent cohort study of approximately 900 individuals to determine: (1) the proportion of variance the DNAm predictors explain in the outcomes; (2) the extent to which these proportions are independent from the contribution of genetics; (3) the accuracy with which the DNAm predictors can identify obese individuals, college-educated individuals, heavy drinkers, high cholesterol levels, and current smokers if provided with a random DNA sample from the population; and (4) the extent to which they can predict health outcomes, such as mortality, and if they do so independently from the phenotypic measure.

Methylation preparation in Generation Scotland:
>Filtering for outliers, sex mismatches, non-blood samples, poorly detected probes, and samples was performed. A full description is provided in Additional file 4. Further filtering was then carried out to remove CpGs with missing values, non-autosomal and non-CpG sites, and any sites not present on the Illumina 450 k array. The latter criterion enabled prediction into the LBC study

LASSO regression in Generation Scotland:
>Penalized regression models were run using the glmnet library in R [31, 32]. Tenfold cross-validation was applied and the mixing parameter (alpha) was set to 1 to apply a LASSO penalty. Coefficients for the model with the lambda value corresponding to the minimum mean cross-validated error were extracted and applied to the corresponding CpGs in an out of sample prediction cohort to create the DNAm predictors.

Prediction analysis in the Lothian Birth Cohort 1936:
>Area under the curve (AUC) estimates were estimated for binary categorizations of BMI, smoking, alcohol consumption, college education, and cholesterol variables. Linear regression models were used to identify the proportion of phenotypic variance explained by the corresponding DNAm predictor and to determine whether this was independent of the polygenic (genetic) signal for each phenotype. Ordinal logistic regression was used for the categorical smoking variable (never, ex, current smoker). Age and sex were considered as covariates, the phenotypic measure was the dependent variable, and the polygenic score or DNAm predictor were the independent variables of interest. Incremental R2 estimates were calculated between the null model and the models with the predictors of interest. An additive genetic and epigenetic model for BMI in the LBC1936 has been reported previously, although a different DNAm predictor, based on unrelated individuals, was derived from the GS data [44]. ROC curves were developed for smoking status, obesity, high/low alcohol consumption, college education and cholesterol variables, and AUC estimates were estimated for binary categorizations of these variables using the pROC library in R [45]. Cox proportional hazards survival models [46] were used to examine whether the phenotype, polygenic score, or DNAm predictor explained mortality risk and if they do so independently of one another. Sex was included as a covariate in all models. Correction for multiple testing was applied using the Bonferroni method.

DNAm predictors classify phenotype extremes
>there were 652 controls and 242 cases for obesity, 745 light-to-moderate drinkers and 150 heavy drinkers, 418 non-smokers and 102 current smokers, and 229 and 666 individuals with > 11 and ≤ 11 years of full-time education, respectively. Following dichotomization of the cholesterol-related variables, there were 531 and 354 individuals with high and low total cholesterol, respectively; 89 and 723 individuals with high and low HDL cholesterol, respectively; 637 and 175 individuals with high and low LDL with remnant cholesterol, respectively; and 307 and 502 with high and low total:HDL cholesterol ratios, respectively.

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
| code   | ?   | 
| data   | ?  | 
| model  | ?  | 



