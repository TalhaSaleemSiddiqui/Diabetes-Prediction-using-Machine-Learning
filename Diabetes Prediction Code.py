import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, roc_curve

warnings.filterwarnings('ignore')

# 1. Dataset Loading & Cleaning
df = pd.read_csv('final dataset.csv')
df = df.drop_duplicates()

# Replace 0 with NaN for biologically impossible zero values
zero_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
df[zero_cols] = df[zero_cols].replace(0, np.nan)

# 2. Features & Target Definition
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# 3. Train-Test Split FIRST 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=7, stratify=y
)

# 4. Impute Missing Values using Train Medians
for col in zero_cols:
    train_median = X_train[col].median()
    X_train[col] = X_train[col].fillna(train_median)
    X_test[col] = X_test[col].fillna(train_median)

# 5. Feature Scaling (Fit on Train, Transform on Test)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 6. Evaluation Helper Function
def evaluate_model(model, model_name, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
    
    cm = confusion_matrix(y_test, y_pred)  
    acc = accuracy_score(y_test, y_pred) * 100  
    auc = roc_auc_score(y_test, y_prob) if y_prob is not None else 0.0

    print(f"\n================ {model_name} ================")  
    print(f"Accuracy Rate: {acc:.2f}%")  
    print(f"ROC AUC Score: {auc:.4f}")  
    print("\nClassification Report:\n", classification_report(y_test, y_pred, digits=4))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # Confusion Matrix
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',  
                xticklabels=['Not Diabetes', 'Diabetes'],  
                yticklabels=['Not Diabetes', 'Diabetes'], ax=axes[0])  
    axes[0].set_title(f'Confusion Matrix - {model_name}')  
    axes[0].set_xlabel('Predicted')  
    axes[0].set_ylabel('Actual')

    # ROC Curve  
    if y_prob is not None:
        fpr, tpr, _ = roc_curve(y_test, y_prob)  
        axes[1].plot(fpr, tpr, color='orange', label=f'ROC curve (AUC = {auc:.2f})')  
        axes[1].plot([0, 1], [0, 1], color='darkblue', linestyle='--')  
        axes[1].set_xlabel('False Positive Rate')  
        axes[1].set_ylabel('True Positive Rate')  
        axes[1].set_title(f'ROC Curve - {model_name}')  
        axes[1].legend(loc='lower right')

    plt.tight_layout()  
    plt.show()

    return acc

# 7. Model Training & Comparison
accuracies = {}

# KNN
knn = KNeighborsClassifier(n_neighbors=7)
accuracies['KNN'] = evaluate_model(knn, 'K-Nearest Neighbors (KNN)', X_train_scaled, y_train, X_test_scaled, y_test)

# Naive Bayes
nb = GaussianNB()
accuracies['Naive Bayes'] = evaluate_model(nb, 'Gaussian Naive Bayes', X_train_scaled, y_train, X_test_scaled, y_test)

# ANN
ann = MLPClassifier(solver='adam', max_iter=5000, hidden_layer_sizes=(16, 8), random_state=7)
accuracies['ANN'] = evaluate_model(ann, 'Artificial Neural Network (ANN)', X_train_scaled, y_train, X_test_scaled, y_test)

# Comparison Bar Chart
plt.figure(figsize=(7, 5))
bars = plt.bar(accuracies.keys(), accuracies.values(), color=['#f39c12', '#e74c3c', '#f1c40f'])
plt.ylabel('Accuracy (%)')
plt.title('Algorithms Comparison Based on Accuracies')
plt.ylim(50, 100)
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 1, f'{yval:.2f}%', ha='center', va='bottom')
plt.tight_layout()
plt.show()
