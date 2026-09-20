# Diabetes-Prediction-using-Machine-Learning

A machine learning project that predicts the likelihood of diabetes in patients based on diagnostic health measurements. Three classification algorithms — **K-Nearest Neighbors (KNN)**, **Gaussian Naive Bayes (GNB)**, and an **Artificial Neural Network (ANN)** — are trained and compared on the classic Pima Indians Diabetes dataset.

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Project Workflow](#project-workflow)
- [Models](#models)
- [Visualization](#visualization)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [License](#license)

## Overview

This project builds and evaluates binary classification models to predict whether a patient has diabetes (`Outcome = 1`) or not (`Outcome = 0`) using 8 diagnostic features. It includes data preprocessing (handling invalid zero values, feature scaling), model training, and evaluation via confusion matrices, ROC curves, classification reports, and a final accuracy comparison chart.

## Dataset

The dataset (`final dataset.csv`) is the well-known **Pima Indians Diabetes Dataset**, containing 768 records of female patients with the following features:

| Column | Description |
|---|---|
| `Pregnancies` | Number of times pregnant |
| `Glucose` | Plasma glucose concentration |
| `BloodPressure` | Diastolic blood pressure (mm Hg) |
| `SkinThickness` | Triceps skin fold thickness (mm) |
| `Insulin` | 2-Hour serum insulin (mu U/ml) |
| `BMI` | Body mass index |
| `DiabetesPedigreeFunction` | Diabetes likelihood based on family history |
| `Age` | Age in years |
| `Outcome` | Target variable — `1` = diabetic, `0` = non-diabetic |


## Project Workflow

1. **Data Loading & Cleaning**
   - Load the CSV file and drop duplicate rows.
   - Several columns (`Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, `BMI`) contain biologically invalid `0` values, which are treated as missing data and imputed with the **column mean**.

2. **Feature Scaling & Train-Test Split**
   - Features (`X`) are standardized using `StandardScaler` (zero mean, unit variance).
   - Data is split into **80% training / 20% testing** using `train_test_split`, stratified on the target variable to preserve class balance (`random_state=7`).

3. **Model Training & Evaluation**
   - Each model is trained on the scaled training data and evaluated on the held-out test set (154 samples: 100 non-diabetic, 54 diabetic).
   - For every model, the script reports:
     - Confusion matrix (TN, FP, FN, TP)
     - Accuracy and misclassification rate
     - ROC AUC score
     - Full classification report (precision, recall, F1-score)
     - A side-by-side plot of the **confusion matrix heatmap** and the **ROC curve**

4. **Model Comparison**
   - A final bar chart compares the test accuracy of all three models.

## Models

| Model | Algorithm | Key Parameters |
|---|---|---|
| KNN | `KNeighborsClassifier` | `n_neighbors=5` |
| Naive Bayes | `GaussianNB` | default parameters |
| ANN | `MLPClassifier` | `hidden_layer_sizes=(12,)`, `solver='adam'`, `max_iter=5000`, `random_state=2` |

## Visualization

### 1. K-Nearest Neighbors (KNN)
![KNN](<KNN.png>)
![KNN Confusion Matrix & ROC plot](<KNN CM & ROC.png>)

### 2. Gaussian Naive Bayes
![Gaussian Naive Bayes](<Gaussian Naive Bayes.png>)
![Gaussian Naive Bayes CM & ROC plot](<Gaussian Naive Bayes CM & ROC.png>)

### 3. Artificial Neural Network (ANN)
![Artificial Neural Network](<Artificial Neural Network.png>)
![ANN Confusion Matrix & ROC plot](<ANN CM & ROC.png>)

### 4. Model Comparison
![Algorithms Comparison](<Algorithms Comparison.png>)




## Results

Evaluated on the 154-sample test set (100 non-diabetic, 54 diabetic):

| Model | Accuracy | Misclassification Rate | ROC AUC | Precision (Diabetic) | Recall (Diabetic) | F1-score (Diabetic) |
|---|---|---|---|---|---|---|
| KNN | 75.32% | 24.68% | 0.8089 | 0.6667 | 0.5926 | 0.6275 |
| Gaussian Naive Bayes | 78.57% | 21.43% | 0.8226 | 0.7234 | 0.6296 | 0.6733 |
| **ANN (MLP)** | 77.27% | 22.73% | **0.8480** | 0.7111 | 0.5926 | 0.6465 |

**Confusion matrices (TN / FP / FN / TP):**

| Model | TN | FP | FN | TP |
|---|---|---|---|---|
| KNN | 84 | 16 | 22 | 32 |
| Gaussian Naive Bayes | 87 | 13 | 20 | 34 |
| ANN | 87 | 13 | 22 | 32 |

**Key takeaways:**
- **Gaussian Naive Bayes** achieved the highest raw accuracy (78.57%).
- The **ANN** achieved the highest ROC AUC (0.848), indicating the best overall ability to distinguish diabetic from non-diabetic cases across classification thresholds.
- All three models show noticeably lower recall on the diabetic class (~59–63%) than on the non-diabetic class, reflecting the class imbalance in the dataset — a meaningful share of diabetic cases are missed (false negatives) and should be considered before any clinical use.


## Installation

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

## Usage

1. Make sure `final dataset.csv` is in the same directory as the script (or update the file path in the code).
2. Run the script:

```bash
python "Diabetes Prediction Code.py"
```

3. The script will print evaluation metrics for each model to the console and display:
   - Confusion matrix + ROC curve plots for KNN, Naive Bayes, and ANN
   - A final bar chart comparing accuracy across all three models

## Project Structure

```
.
├── Diabetes Prediction Code.py       # Main script: preprocessing, training, evaluation
├── final dataset.csv                 # Diabetes dataset 
├── KNN.png                           # KNN metrics output
├── KNN CM & ROC.png                # KNN confusion matrix + ROC curve
├── Gaussian Naive Bayes.png           # Naive Bayes metrics output
├── Gaussian Naive Bayes CM & ROC.png  # Naive Bayes confusion matrix + ROC curve
├── Artificial Neural Network.png      # ANN metrics output
├── ANN CM & ROC.png                   # ANN confusion matrix + ROC curve
├── Algorithms Comparison.png          # Algorithms(KNN,Naive Bayes, ANN) Comparison based on accuracies 
└── README.md                         # Project documentation
```

## Requirements

- Python 3.8+
- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn

## License

This project is intended for educational and research purposes. Please check the license terms of the original Pima Indians Diabetes Dataset before any commercial or clinical use.
