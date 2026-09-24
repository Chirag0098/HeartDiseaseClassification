# Heart Disease Classification Using Machine Learning

## AICTE \| IBM SkillsBuild Data Analytics with AI Internship 2026 \| BharatCares

**Project by:** Chirag Vishwakarma\
**Institution:** Ramniranjan Jhunjhunwala College of Arts, Science and
Commerce (Autonomous)\
**Internship Period:** 17 August 2026 -- 21 September 2026\
**Submission Date:** 24 September 2026

------------------------------------------------------------------------

## 1. Project Overview

This project develops and compares machine-learning classification
models for predicting the presence or absence of heart disease from
structured clinical and demographic data.

The project follows an end-to-end data analytics and machine-learning
workflow:

**Data Acquisition → Data Understanding → Data Cleaning → Exploratory
Data Analysis → Feature Engineering → Train-Test Split → Feature Scaling
→ Model Training → Model Evaluation → Model Selection**

The project is intended as an academic and internship learning project.
The model results should not be interpreted as clinical diagnostic
advice or as evidence of clinical deployment readiness.

------------------------------------------------------------------------

## 2. Objectives

The main objectives of the project are to:

-   Understand the structure and quality of the heart-disease dataset.
-   Explore numerical and categorical variables.
-   Identify and handle missing values.
-   Remove variables with substantial missingness.
-   Transform the original target into a binary classification problem.
-   Prepare categorical and numerical variables for machine learning.
-   Train multiple classification algorithms.
-   Compare models using accuracy, weighted precision, weighted recall,
    and weighted F1-score.
-   Select the model with the highest weighted recall according to the
    project's evaluation criterion.

------------------------------------------------------------------------

## 3. Dataset

The project uses the **Heart Disease UCI dataset**, downloaded in the
notebook using KaggleHub.

The notebook uses:

``` text
heart_disease_uci.csv
```

The dataset contains **920 observations and 16 original columns**.

### Dataset sources

  Source            Records
  --------------- ---------
  Cleveland             304
  Hungary               293
  VA Long Beach         200
  Switzerland           123
  **Total**         **920**

### Important variables

The dataset contains demographic, clinical, exercise-related, and
diagnostic variables including:

-   `age`
-   `sex`
-   `dataset`
-   `cp`
-   `trestbps`
-   `chol`
-   `fbs`
-   `restecg`
-   `thalch`
-   `exang`
-   `oldpeak`
-   `num`

The original target variable `num` contains values from 0 to 4.

For this project, it is transformed into a binary target:

-   `0` → No disease
-   `1, 2, 3, 4` → Disease present

The resulting target variable is named:

``` text
num_trans
```

------------------------------------------------------------------------

## 4. Data Preprocessing

The following preprocessing steps are implemented in the notebook.

### 4.1 Removing highly incomplete variables

The following variables are removed:

``` text
slope
ca
thal
```

The identifier column `id` is also removed because it does not provide
meaningful predictive information.

### 4.2 Target transformation

The original multi-level target is transformed into a binary
classification target.

``` text
num = 0       → num_trans = 0
num = 1–4     → num_trans = 1
```

The resulting target distribution is:

  Class               Count
  ----------------- -------
  No disease            411
  Disease present       509

### 4.3 Missing-value treatment

Numerical variables are imputed using their median.

Categorical variables are imputed using their most frequent category.

### 4.4 Invalid zero values

Zero values in:

-   `chol`
-   `trestbps`

are replaced using the corresponding median values.

### 4.5 Categorical encoding

Categorical variables are converted into numerical representations using
one-hot encoding.

The notebook applies one-hot encoding with the first category dropped.

### 4.6 Train-test split

The dataset is divided into:

-   **80% training data**
-   **20% testing data**

A stratified split is used to preserve the target-class distribution.

``` text
random_state = 42
```

### 4.7 Feature scaling

`StandardScaler` is used to standardize the feature variables.

The scaler is fitted on the training data and then applied to both
training and test data.

------------------------------------------------------------------------

## 5. Machine Learning Models

Six classification algorithms are evaluated.

  Model                    Configuration
  ------------------------ -----------------------------------------------------
  Logistic Regression      Balanced class weights
  Decision Tree            Gini criterion, balanced class weights
  Random Forest            100 estimators, balanced class weights
  Bernoulli Naive Bayes    Default BernoulliNB
  Support Vector Machine   Linear kernel, balanced class weights
  XGBoost                  Class-imbalance adjustment using `scale_pos_weight`

------------------------------------------------------------------------

## 6. Evaluation Metrics

The models are evaluated using:

### Accuracy

Measures the proportion of correctly classified observations.

### Precision

Measures how many observations predicted as positive are actually
positive.

### Recall

Measures how many actual positive observations are correctly identified.

### F1-score

Combines precision and recall into a single measure using their harmonic
mean.

The project uses **weighted** precision, recall, and F1-score.

### Model selection criterion

The notebook selects the model with the **highest weighted recall**.

------------------------------------------------------------------------

## 7. Results

The reported test-set performance is:

  ------------------------------------------------------------------------
  Model              Accuracy      Precision         Recall       F1-score
  ------------ -------------- -------------- -------------- --------------
  Logistic             0.8370         0.8367         0.8370         0.8367
  Regression                                                

  Decision             0.7554         0.7552         0.7554         0.7553
  Tree                                                      

  Random               0.8261         0.8259         0.8261         0.8256
  Forest                                                    

  Bernoulli            0.8207         0.8210         0.8207         0.8196
  Naive Bayes                                               

  Support          **0.8424**     **0.8422**     **0.8424**     **0.8421**
  Vector                                                    
  Machine                                                   

  XGBoost              0.8152         0.8149         0.8152         0.8147
  ------------------------------------------------------------------------

According to the project's predefined selection criterion, **Support
Vector Machine (SVM)** is the selected model because it records the
highest weighted recall:

``` text
Weighted Recall = 0.8424
```

Its other reported test-set metrics are:

``` text
Accuracy  = 0.8424
Precision = 0.8422
F1-score  = 0.8421
```

------------------------------------------------------------------------

## 8. Technology Stack

### Programming Language

-   Python

### Development Environment

-   Jupyter Notebook

### Data Analysis

-   Pandas
-   NumPy

### Data Visualization

-   Matplotlib
-   Seaborn

### Machine Learning

-   Scikit-learn
-   XGBoost

### Dataset Acquisition

-   KaggleHub

------------------------------------------------------------------------

## 9. Project Structure

A typical project folder can contain:

``` text
Heart-Disease-Classification/
│
├── Heart_Disease_classification.ipynb
├── requirements.txt
├── README.md
└── report/
    └── Chirag_Vishwakarma_Heart_Disease_Project_Report.docx
```

------------------------------------------------------------------------

## 10. Installation

Create a Python environment and install the required packages:

``` bash
pip install -r requirements.txt
```

The required packages are:

``` text
pandas
numpy
matplotlib
seaborn
kagglehub
scikit-learn
xgboost
```

------------------------------------------------------------------------

## 11. Running the Project

1.  Clone or download the project.
2.  Install the dependencies from `requirements.txt`.
3.  Open `Heart_Disease_classification.ipynb` in Jupyter Notebook or
    JupyterLab.
4.  Run the notebook cells sequentially.
5.  Ensure internet access is available if the notebook needs to
    download the dataset through KaggleHub.
6.  Review the exploratory visualizations, model outputs, performance
    metrics, and confusion matrix.

------------------------------------------------------------------------

## 12. Reproducibility

The notebook uses fixed random seeds where applicable, including:

``` text
random_state = 42
```

This helps reproduce the same train-test partition and model results
when the same dataset version and compatible software environment are
used.

------------------------------------------------------------------------

## 13. Key Learning Outcomes

Through this project, the following practical skills are demonstrated:

-   Data acquisition
-   Data cleaning
-   Missing-value analysis
-   Exploratory data analysis
-   Feature engineering
-   Categorical encoding
-   Feature scaling
-   Stratified sampling
-   Classification modelling
-   Handling class imbalance
-   Model comparison
-   Evaluation metric interpretation
-   Confusion-matrix analysis
-   Machine-learning workflow documentation

------------------------------------------------------------------------

## 14. Limitations

The results should be interpreted within the scope of this project.

-   The experiment uses one dataset and one held-out train-test split.
-   External validation is not performed.
-   Variables with substantial missingness are removed.
-   The original multi-level target is converted into a binary outcome.
-   Hyperparameter tuning is not performed systematically.
-   Cross-validation is not used for the reported model comparison.
-   The project does not establish clinical validity.
-   The results should not be used as a substitute for professional
    medical diagnosis.

------------------------------------------------------------------------

## 15. Future Scope

Potential extensions include:

-   Stratified k-fold cross-validation.
-   Hyperparameter optimization.
-   ROC-AUC and PR-AUC analysis.
-   Sensitivity and specificity analysis.
-   Probability calibration.
-   Threshold optimization.
-   Alternative missing-value strategies.
-   Feature-selection techniques.
-   SHAP or other explainability methods.
-   External validation using an independent dataset.
-   Model monitoring and versioning for a future deployment pipeline.

------------------------------------------------------------------------

## 16. Conclusion

This project demonstrates a complete machine-learning workflow for
binary heart-disease classification. The analysis begins with data
acquisition and exploratory analysis and progresses through data
cleaning, missing-value treatment, target transformation, feature
encoding, scaling, classification modelling, and comparative evaluation.

Six classification algorithms are evaluated. Based on the project's
predefined criterion of highest weighted recall, the Support Vector
Machine achieves the strongest reported result, with a weighted recall
of **0.8424**.

The project provides practical experience in applying data analytics and
machine-learning techniques to a real-world-style healthcare dataset
while emphasizing that predictive performance on a single dataset does
not establish clinical effectiveness or generalizability.

------------------------------------------------------------------------

## 17. Author

**Chirag Vishwakarma**

Ramniranjan Jhunjhunwala College of Arts, Science and Commerce
(Autonomous)

**AICTE \| IBM SkillsBuild Data Analytics with AI Internship 2026 \|
BharatCares**

**Internship:** 17 August 2026 -- 21 September 2026\
**Submission:** 24 September 2026
