# Diabetes Prediction Using Support Vector Machines

## Abstract
Diabetes mellitus is a chronic metabolic disorder that affects millions worldwide and is associated with severe complications if not detected early.  
This project implements a machine learning pipeline to predict diabetes using clinical features, demonstrating how data-driven methods can assist in early diagnosis and risk assessment.

---

## 1. Introduction
Early detection of diabetes is essential for effective management and prevention of complications.  
Traditional threshold-based clinical methods may fail to capture complex patterns between physiological variables.  
Machine learning approaches, such as Support Vector Machines (SVM), provide an opportunity to model these relationships and predict the likelihood of diabetes in patients based on clinical data.

The objectives of this project are:
- To develop a robust classification model for diabetes prediction
- To apply proper preprocessing techniques to clinical data
- To evaluate model performance using meaningful metrics

---

## 2. Dataset
The dataset used is the **Pima Indians Diabetes Dataset**, a well-known benchmark in biomedical machine learning.

### Features:
- Glucose concentration  
- Blood pressure  
- Body Mass Index (BMI)  
- Insulin level  
- Age and other physiological measurements

### Target:
- Binary outcome: presence or absence of diabetes

The dataset contains both positive (diabetic) and negative (non-diabetic) samples, suitable for supervised classification.

---

## 3. Methodology

### 3.1 Train-Test Split
Data was split into training and testing sets (80/20) using **stratified sampling** to preserve the class distribution.

```python
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, stratify=Y, random_state=2
)
