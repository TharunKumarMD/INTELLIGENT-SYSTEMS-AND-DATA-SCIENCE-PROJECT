# Refining Decision Tree Gini for Income Prediction Harnessing a Multi Model Strategy with the Adult Dataset

## Overview
This project aims to enhance income prediction accuracy by employing a multi-model strategy using various machine learning techniques on the Adult dataset. The main focus is on optimizing the Decision Tree Gini model and comparing it with other algorithms such as Logistic Regression, SVM, and Naive Bayes.

## Table of Contents
1. [Dataset](#dataset)
2. [Installation](#installation)
3. [Usage](#usage)
4. [Data Preprocessing](#data-preprocessing)
5. [Modeling](#modeling)
6. [Evaluation Metrics](#evaluation-metrics)
7. [Results](#results)
8. [Conclusion](#conclusion)
9. [License](#license)

## Dataset
The Adult dataset contains demographic information, which is used to predict whether an individual earns more or less than $50,000 per year. 

## Installation
Make sure to install the required libraries. You can do this using pip:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn graphviz
```

## Usage
To run the code, execute the following command in your terminal or Jupyter notebook:

```python
python main.py
```

Replace `main.py` with the name of your script if it differs.

## Data Preprocessing
The data preprocessing steps include:
- Importing the dataset
- Handling missing values by replacing "?" with NaN
- Filling missing values with the mode of each column
- Encoding categorical variables using Label Encoding

```python
# Example code for preprocessing
import pandas as pd
import numpy as np
from sklearn import preprocessing

# Load dataset
df = pd.read_csv('adult.csv')

# Data preprocessing steps...
```

## Modeling
We employ multiple machine learning models:
- **Decision Tree Classifier**
- **Logistic Regression**
- **Support Vector Machine (SVM)**
- **Naive Bayes**

### Example of Decision Tree Implementation
```python
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=100)

# Fit Decision Tree model
dt_clf = DecisionTreeClassifier(criterion="gini", random_state=100)
dt_clf.fit(X_train, y_train)

# Predictions
y_pred = dt_clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred) * 100)
```

## Evaluation Metrics
The models are evaluated using the following metrics:
- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

```python
from sklearn.metrics import classification_report, confusion_matrix

# Print evaluation metrics
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
```

## Results
Include a summary of the results from each model and any visualizations such as confusion matrices or ROC curves.

## Conclusion
Summarize the findings and the effectiveness of the different models. Discuss any potential improvements or future work.

