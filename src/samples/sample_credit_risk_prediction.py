from pprint import pprint

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_curve, roc_auc_score, auc, accuracy_score, precision_score, recall_score, f1_score

from tinydag.graph import Graph
from tinydag.node import Node


def load_data(url):
    column_names = [
        "existing checking account", "duration", "credit history", "purpose",
        "credit amount", "savings", "employment", "installment rate", "personal status",
        "other debtors", "residence since", "property", "age", "other installment plans",
        "housing", "number of existing credits", "job", "liable people", "telephone",
        "foreign worker", "credit risk"
    ]
    german_credit_df = pd.read_csv(url, sep=' ', names=column_names)
    return {"data": german_credit_df}


def perform_eda(data):
    plt.figure(figsize=(12, 8))
    numeric_features = ["duration", "credit amount", "age", "number of existing credits"]
    for i, feature in enumerate(numeric_features, 1):
        plt.subplot(2, 2, i)
        sns.histplot(data[feature], bins=20, kde=True)
        plt.title(f"{feature} Histogram")
    plt.tight_layout()

    plt.figure(figsize=(18, 24))
    categorical_features = [
        "existing checking account", "credit history", "purpose", "savings",
        "employment", "personal status", "other debtors", "property",
        "other installment plans", "housing", "job", "telephone", "foreign worker"
    ]
    for i, feature in enumerate(categorical_features, 1):
        plt.subplot(5, 3, i)
        sns.countplot(data=data, x=feature)
        plt.title(f"{feature} Count Plot")
        plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def preprocess_data(data):
    data['credit risk'] = data['credit risk'].map({1: 0, 2: 1})
    X = data.drop('credit risk', axis=1)
    y = data['credit risk']
    X = pd.get_dummies(X, drop_first=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return {"X_train": X_train_scaled, "X_test": X_test_scaled, "y_train": y_train, "y_test": y_test}


def train_models(X_train, y_train):
    classifiers = {
        "Logistic Regression": LogisticRegression(),
        "Support Vector Machine": SVC(probability=True),
        "k-Nearest Neighbors": KNeighborsClassifier(),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42)
    }
    trained_models = {}
    for name, classifier in classifiers.items():
        classifier.fit(X_train, y_train)
        trained_models[name] = classifier
    return {"models": trained_models}


def plot_roc_curves(X_test, y_test, models):
    plt.figure(figsize=(10, 8))
    for name, classifier in models.items():
        y_prob = classifier.predict_proba(X_test)[:, 1]
        fpr, tpr, thresholds = roc_curve(y_test, y_prob)
        auc_score = auc(fpr, tpr)
        plt.plot(fpr, tpr, label='%s (AUC = %0.2f)' % (name, auc_score))
    plt.plot([0, 1], [0, 1], linestyle='--', color='black', label='Random Guessing')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for Classifiers')
    plt.legend()
    plt.show()


def calculate_metrics(X_test, y_test, models):
    metrics = {}
    for name, model in models.items():
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        metrics[name] = {"Accuracy": accuracy, "Precision": precision, "Recall": recall, "F1-score": f1}
    return {"metrics": metrics}


def main():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"

    nodes = [
        Node(["url"], load_data, "data_loader", ["data"]),
        Node(["data_loader/data"], perform_eda, "eda"),
        Node(["data_loader/data"], preprocess_data, "preprocessor", ["X_train", "X_test", "y_train", "y_test"]),
        Node(["preprocessor/X_train", "preprocessor/y_train"], train_models, "model_trainer", ["models"]),
        Node(["preprocessor/X_test", "preprocessor/y_test", "model_trainer/models"], plot_roc_curves, "roc_curves"),
        Node(["preprocessor/X_test", "preprocessor/y_test", "model_trainer/models"], calculate_metrics,
             "metrics_calculator", ["metrics"]),
    ]

    graph = Graph(nodes)
    print("Graph: ", graph)
    graph.render()

    data = {"url": url}
    graph.check()
    results = graph.calculate(data, parallel=True)
    pprint(f"{results['metrics_calculator/metrics']}")


if __name__ == "__main__":
    main()
