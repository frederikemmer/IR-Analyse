import numpy as np
import pandas as pd
import os
import time
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.preprocessing import StandardScaler

from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


# 20 Farben für die Visualisierung
distinct_colors = [
    [0.12156863, 0.46666667, 0.70588235],  # Blau
    [1.        , 0.49803922, 0.05490196],  # Orange
    [0.17254902, 0.62745098, 0.17254902],  # Grün
    [0.83921569, 0.15294118, 0.15686275],  # Rot
    [0.58039216, 0.40392157, 0.74117647],  # Lila
    [0.54901961, 0.3372549 , 0.29411765],  # Braun
    [0.89019608, 0.46666667, 0.76078431],  # Pink
    [0.49803922, 0.49803922, 0.49803922],  # Grau
    [0.7372549 , 0.74117647, 0.13333333],  # Gelb
    [0.09019608, 0.74509804, 0.81176471],  # Türkis
    [0.12156863, 0.46666667, 0.70588235],  # Blau
    [1.        , 0.49803922, 0.05490196],  # Orange
    [0.17254902, 0.62745098, 0.17254902],  # Grün
    [0.83921569, 0.15294118, 0.15686275],  # Rot
    [0.58039216, 0.40392157, 0.74117647],  # Lila
    [0.54901961, 0.3372549 , 0.29411765],  # Braun
    [0.89019608, 0.46666667, 0.76078431],  # Pink
    [0.49803922, 0.49803922, 0.49803922],  # Grau
    [0.7372549 , 0.74117647, 0.13333333],  # Gelb
    [0.09019608, 0.74509804, 0.81176471]   # Türkis
]

# Daten einlesen
data_path = 'data/Sample-Database V1 (Sphere).csv'  # Setzen Sie hier Ihren Dateipfad
data = pd.read_csv(data_path, sep=';')

# Daten vorbereiten
X = data.drop(columns=["#", "Material"]).values # Features
y = data['Material'].values                     # Zielvariable


# Testdaten generieren und skalieren
def test_data():
    # Daten aufteilen
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Daten skalieren
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test

test_data()

def generate_accuracy_scores(models, mean_from=100):
    index = 0
    # Modelle trainieren und bewerten
    accuracy_scores = [[], []]
    for name, model in models.items():
        accuracies = []
        for _ in range(mean_from):
            X_train_scaled, X_test_scaled, y_train, y_test = test_data()
            model.fit(X_train_scaled, y_train)
            accuracy = model.score(X_test_scaled, y_test)
            accuracies.append(accuracy)

        average_accuracy = np.mean(accuracies)

        accuracy_scores[0].append(name)
        accuracy_scores[1].append(average_accuracy)

        index += 1
        print(f'{name.ljust(50)} Genauigkeit: {average_accuracy:.4f} - {index}/{len(models)} Modelle berechnet')
        
    return accuracy_scores


def logitstic_regression(c_values=[0.01, 0.05, 0.1, 0.5, 1, 2, 5], max_iter=5000, mean_from=10):
    # Modelle mit verschiedenen Parametern definieren
    models = {}

    for c in c_values:
        model_name = f"Lineare Regression - C={c}"
        models[model_name] = LogisticRegression(C=c, max_iter=max_iter, class_weight='balanced')

    accuracy_scores = generate_accuracy_scores(models, mean_from=mean_from)
    return accuracy_scores

def svm(c_values=[10], kernel_values=['rbf'], gamma_values=['scale'], mean_from=10):
    # Modelle mit verschiedenen Parametern definieren
    models = {}

    for c in c_values:
        for kernel in kernel_values:
            for gamma in gamma_values:
                model_name = f"SVM - C={c}, Kernel={kernel}, Gamma={gamma}"
                models[model_name] = SVC(kernel=kernel, C=c, gamma=gamma, class_weight='balanced')

    accuracy_scores = generate_accuracy_scores(models, mean_from=mean_from)
    return accuracy_scores



def plot_accuracy_scores(*accuracy_scores, xlabel, ylabel, title,  savefig=False, filename=None):
    plt.figure(figsize=(15, 10), layout='tight', dpi=100) 
    
    for i in range(len(accuracy_scores)):
        plt.plot(accuracy_scores[i][0], accuracy_scores[i][1], color=distinct_colors[i])
        
    plt.xticks(rotation=45)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    # plt.legend()
    plt.title(title)
    
    if savefig:
        file_path = os.path.join(os.path.expanduser("~"), "Desktop", filename)
        plt.savefig(file_path)
    else:
        plt.show()



# Test svm mit verschiedenen "C" Werten
svm_scores = svm(c_values=[0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10, 20], mean_from=100)
plot_accuracy_scores(svm_scores, xlabel="C", ylabel="Genauigkeit",
                     title="Genauigkeit der SVM mit verschiedenen C-Werten",
                     savefig=True, filename="svm_c")

# Test svm mit verschiedenen "Kernel" Werten
svm_scores = svm(kernel_values=['linear', 'poly', 'rbf', 'sigmoid'], mean_from=100)
plot_accuracy_scores(svm_scores, xlabel="Kernel", ylabel="Genauigkeit",
                     title="Genauigkeit der SVM mit verschiedenen Kernel-Werten",
                     savefig=True, filename="svm_kernel")


logistic_regression_scores = logitstic_regression(c_values=[0.01, 0.05, 0.1, 0.5, 1, 2, 5], mean_from=50)

# Test svm mit verschiedenen "Gamma" Werten
svm_scores = svm(gamma_values=[0.01, 0.05, 0.1, 0.5, 1, 2, 5], mean_from=100)
plot_accuracy_scores(svm_scores, logistic_regression_scores, xlabel="Gamma", ylabel="Genauigkeit",
                     title="Genauigkeit der SVM mit verschiedenen Gamma-Werten",
                     savefig=True, filename="svm_gamma")

