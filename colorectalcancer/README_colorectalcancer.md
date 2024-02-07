Differentially methylated markers associated with CRC risk were screened on the training dataset using LASSO selection and stepwise logistic regression analysis. sixteen-CpG panel overlapping between the two methods were selected to calculate a methylation risk score (MRS)

In the article, they set these attributes as a best model to calculate MRS:
```MRS=beta*CG
CG = ['cg06551493', 'cg01419670', 'cg16530981', 'cg18022036','cg12691488', 'cg17292758', 'cg16170495',
      'cg11240062', 'cg21585512', 'cg24702253', 'cg17187762', 'cg05983326', 'cg06825163',
      'cg11885357', 'cg08829299', 'cg07044115']
beta= [-0.41, 0.4332, 0.2895, -0.5172, -0.3915, -0.3246, -0.2886, 0.2451, -0.5651, 
             0.3615, -0.2445, -0.3951, -0.5089, -0.2504, -0.2357,-0.3607]
```

The MRS (range, − 5.59 to 4.35) was significantly higher for CRC subjects than in healthy normal subjects (P <  0.000), with a median MRS of 1.68 (IQR, 1.43) in CRC subjects and − 0.430 (IQR, 2.89) in healthy normal subjects.

The MRS showed a good predictive ability for discriminating between CRC and healthy normal subjects (AUC, 0.85; 95% CI: 0.81, 0.89).

Similar to the training dataset, the MRS showed a good predictive ability for discriminating between CRC and healthy normal subjects (AUC, 0.82; 95% C: 0.76, 0.88)
