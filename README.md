# Affect of social factors on Alzheimer's Disease onset using machine learning models



## Overview
The dataset has been meticulously gathered, and the goal is to establish a
robust machine-learning model which can identify the combinations of social
factors contributing to the development of Alzheimer’s disease. The methodology
follows a structured approach, beginning with data pre-processing procedures.
Subsequently, we will perform feature analysis to enhance features’ relevance
and effectiveness. Once the feature analysis is completed, next step is to exhibit the
preparation of our data which will create feature combinations. Then several classification models are created and then 
their performance was evaluated by accuracy.
## Dataset
The data utilized in this study were sourced from the Open Access Series of
Imaging Studies (OASIS) website. OASIS has made available a comprehensive
cross-sectional MRI image dataset that is freely accessible for educational
purposes and scientific investigations. The dataset comprises over 400 individuals
belonging to various age groups and includes information regarding their
diagnosed status for dementia. The dataset consists of 436 individuals across
a wide range of 18 to 96 years of age.

## Data exploration
To explore the combinations of features and their contribution to Alzheimer’s
disease development, our focus centered on examining correlations. Here is the heat map showing the correlation between features.
![image](https://github.com/user-attachments/assets/8a22ca68-4fec-4016-8cb3-c2927ce8a5f7)

Within the heatmap, the relationship between the social factor ‘Education’ and the normalized whole
brain volume (nWBV) is noticeble. The observed correlation coefficient between these two
variables was determined to be ‘0.15’, indicating a relatively modest level of
correlation. However, given the significance of this relationship in supporting
our model, it is good to conduct a more comprehensive analysis to deepen the
understanding of this correlation. 
## Feature engineering
The 'Nan' values were removed. After that non-relvant features like preffered hand is removed.
A new target feature engineered using clinical rating of Alzeimer's disease for the models. Based on the heatmap of correlation, highly correlated features were discarded. 
## Models
To support the hypothesis, 15 feature combinations are created of social as well as other factors.
After that five classification models are developed: ‘KNearest
Neighbours’ (k-NN), ‘Support Vector Machine’ (SVM), ‘Gaussian NB’,
‘Random Forest’ (RF), and ‘Multilayer Preceptor’ (MLP).
To ensure reliability in the analysis, the 5-fold cross-validation applied on the models.
The resulting mean accuracies offer valuable insights into the probabilistic
associations between specific feature combinations and the likelihood
of Alzheimer’s disease development, ultimately enhancing the understanding of
the complex interplay between factors contributing to this condition.

## Results
It is important to note that the hypothesis was predicated on a strong correlation
between education and nWBV. The prediction accuracy for
this correlation is 67.64%, indicating a moderate strength of association. Consequently, the accuracy falls short of providing robust evidence to support a
strong correlation between Education and nWBV.
Despite the moderate correlation observed, additional factors
and considerations are necessary to comprehensively understand the complexity
of a correlation between Education and nWBV.
