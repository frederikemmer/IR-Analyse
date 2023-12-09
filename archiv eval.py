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


pass
'''
def get_accuracy_all():
    # Modelle definieren
    models = {
        "Neuronales Netz": MLPClassifier(max_iter=1000),
        "Lineare Regression": LinearRegression(),  # Ersetzt logistische Regression
        "KNN": KNeighborsClassifier(),
        "SVM": SVC()
    }

    # Modelle trainieren und bewerten
    accuracy_scores = []
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)

        # Genauigkeit für lineare Regression anders berechnen
        if name == "Lineare Regression":
            # Hier Ihre Methode zur Bewertung der linearen Regression
            # Zum Beispiel: Berechnung des R^2-Wertes oder eines anderen passenden Maßes
            accuracy = model.score(X_test_scaled, y_test)
        else:
            accuracy = accuracy_score(y_test, y_pred)

        accuracy_scores.append(accuracy)
        print(f'{name} Genauigkeit: {accuracy:.4f}')

    # Ergebnisse visualisieren
    plt.figure(figsize=(10, 6))
    plt.bar(models.keys(), accuracy_scores, color='skyblue')
    plt.ylabel('Genauigkeit')
    plt.title('Modellvergleich')
    plt.show()

def get_accuracy_all_params():
    # Modelle mit verschiedenen Hyperparametern definieren
    models = {
        "Neuronales Netz - 1 Schicht": MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000),
        "Neuronales Netz - 2 Schichten": MLPClassifier(hidden_layer_sizes=(100, 100), max_iter=1000),
        "Logistische Regression - C=1": LogisticRegression(C=1),
        "Logistische Regression - C=0.5": LogisticRegression(C=0.5),
        "KNN - n_neighbors=3": KNeighborsClassifier(n_neighbors=3),
        "KNN - n_neighbors=5": KNeighborsClassifier(n_neighbors=5),
        "SVM - linear": SVC(kernel='linear'),
        "SVM - rbf": SVC(kernel='rbf')
    }

    # Modelle trainieren und bewerten
    accuracy_scores = []
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        accuracy_scores.append(accuracy)
        print(f'{name} Genauigkeit: {accuracy:.4f}')

    # Ergebnisse visualisieren
    plt.figure(figsize=(15, 6))
    plt.bar(models.keys(), accuracy_scores, color='skyblue')
    plt.xticks(rotation=45)
    plt.ylabel('Genauigkeit')
    plt.title('Modellvergleich mit verschiedenen Hyperparametern')
    plt.show()



def get_accuracy_nn():
    # Neuronale Netz-Modelle mit verschiedenen Hyperparametern definieren
    nn_models = {
        "NN - 1 Schicht (100)": MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000),
        "NN - 2 Schichten (100, 50)": MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000),
        "NN - 3 Schichten (100, 50, 25)": MLPClassifier(hidden_layer_sizes=(100, 50, 25), max_iter=1000),
        "NN - relu, adam": MLPClassifier(activation='relu', solver='adam', max_iter=1000),
        "NN - tanh, adam": MLPClassifier(activation='tanh', solver='adam', max_iter=1000),
        "NN - relu, sgd": MLPClassifier(activation='relu', solver='sgd', learning_rate_init=0.01, max_iter=1000),
        "NN - tanh, sgd": MLPClassifier(activation='tanh', solver='sgd', learning_rate_init=0.01, max_iter=1000)
    }

    # Neuronale Netz-Modelle trainieren und bewerten
    accuracy_scores = []
    for name, model in nn_models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        accuracy_scores.append(accuracy)
        print(f'{name} Genauigkeit: {accuracy:.4f}')

    # Ergebnisse visualisieren
    plt.figure(figsize=(15, 6))
    plt.bar(nn_models.keys(), accuracy_scores, color='skyblue')
    plt.xticks(rotation=45)
    plt.ylabel('Genauigkeit')
    plt.title('Vergleich von Neuronalen Netz-Modellen mit verschiedenen Hyperparametern')
    plt.show()

def get_accuracy_nn2():
    # Neuronale Netz-Modelle mit verschiedenen Hyperparametern für SGD definieren
    nn_sgd_models = {
        "NN - 1 Schicht, lr=0.01": MLPClassifier(solver='sgd', hidden_layer_sizes=(100,), learning_rate_init=0.01, max_iter=1000),
        "NN - 2 Schichten, lr=0.01": MLPClassifier(solver='sgd', hidden_layer_sizes=(100, 50), learning_rate_init=0.01, max_iter=1000),
        "NN - 3 Schichten, lr=0.01": MLPClassifier(solver='sgd', hidden_layer_sizes=(100, 50, 25), learning_rate_init=0.01, max_iter=1000),
        "NN - 1 Schicht, lr=0.001": MLPClassifier(solver='sgd', hidden_layer_sizes=(100,), learning_rate_init=0.001, max_iter=1000),
        "NN - relu, lr=0.01": MLPClassifier(solver='sgd', activation='relu', learning_rate_init=0.01, max_iter=1000),
        "NN - tanh, lr=0.01": MLPClassifier(solver='sgd', activation='tanh', learning_rate_init=0.01, max_iter=1000),
        "NN - relu, Momentum": MLPClassifier(solver='sgd', activation='relu', learning_rate_init=0.01, momentum=0.9, max_iter=1000),
        "NN - tanh, Momentum": MLPClassifier(solver='sgd', activation='tanh', learning_rate_init=0.01, momentum=0.9, max_iter=1000)
    }

    # Neuronale Netz-Modelle trainieren und bewerten
    accuracy_scores = []
    for name, model in nn_sgd_models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        accuracy_scores.append(accuracy)
        print(f'{name} Genauigkeit: {accuracy:.4f}')

    # Ergebnisse visualisieren
    plt.figure(figsize=(15, 6))
    plt.bar(nn_sgd_models.keys(), accuracy_scores, color='skyblue')
    plt.xticks(rotation=45)
    plt.ylabel('Genauigkeit')
    plt.title('Vergleich von Neuronalen Netz-Modellen (SGD) mit verschiedenen Hyperparametern')
    plt.show()
    
# mitteln aus 10 Durchgängen
def get_accuracy_nn3():
    # Neuronale Netz-Modelle mit verschiedenen Hyperparametern für SGD definieren
    nn_sgd_models = {
        "NN - 1 Schicht, lr=0.01": MLPClassifier(solver='sgd', hidden_layer_sizes=(100,), learning_rate_init=0.01, max_iter=1000),
        "NN - 2 Schichten, lr=0.01": MLPClassifier(solver='sgd', hidden_layer_sizes=(100, 50), learning_rate_init=0.01, max_iter=1000),
        "NN - 3 Schichten, lr=0.01": MLPClassifier(solver='sgd', hidden_layer_sizes=(100, 50, 25), learning_rate_init=0.01, max_iter=1000),
        "NN - 1 Schicht, lr=0.001": MLPClassifier(solver='sgd', hidden_layer_sizes=(100,), learning_rate_init=0.001, max_iter=1000),
        "NN - relu, lr=0.01": MLPClassifier(solver='sgd', activation='relu', learning_rate_init=0.01, max_iter=1000),
        "NN - tanh, lr=0.01": MLPClassifier(solver='sgd', activation='tanh', learning_rate_init=0.01, max_iter=1000),
        "NN - relu, Momentum": MLPClassifier(solver='sgd', activation='relu', learning_rate_init=0.01, momentum=0.9, max_iter=1000),
        "NN - tanh, Momentum": MLPClassifier(solver='sgd', activation='tanh', learning_rate_init=0.01, momentum=0.9, max_iter=1000)
    }

    # Neuronale Netz-Modelle mehrmals trainieren und Genauigkeit bewerten
    accuracy_scores = {name: [] for name in nn_sgd_models}

    for name, model in nn_sgd_models.items():
        for _ in range(10):  # Jedes Modell 10 Mal laufen lassen
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)
            accuracy_scores[name].append(accuracy)

    # Mittelwerte der Genauigkeit berechnen
    average_accuracies = {name: np.mean(scores) for name, scores in accuracy_scores.items()}
    print("Durchschnittliche Genauigkeit für jedes Modell:")
    for name, avg_accuracy in average_accuracies.items():
        print(f'{name}: {avg_accuracy:.4f}')

    # Ergebnisse visualisieren
    plt.figure(figsize=(15, 6))
    plt.bar(average_accuracies.keys(), average_accuracies.values(), color='skyblue')
    plt.xticks(rotation=45)
    plt.ylabel('Durchschnittliche Genauigkeit')
    plt.title('Durchschnittliche Genauigkeit von Neuronalen Netz-Modellen (SGD) mit verschiedenen Hyperparametern')
    plt.show()
  
# 1-10 Schichten
def get_accuracy_nn4():
    # Neuronale Netz-Modelle mit verschiedenen Hyperparametern für SGD definieren
    nn_sgd_models = {
        "NN - 1 Schicht": MLPClassifier(hidden_layer_sizes=(100,),
            solver='sgd', activation='tanh', learning_rate_init=0.01, momentum=0.9, max_iter=5000),
        "NN - 2 Schichten": MLPClassifier(hidden_layer_sizes=(100, 100),
            solver='sgd', activation='tanh', learning_rate_init=0.01, momentum=0.9, max_iter=5000),
        "NN - 3 Schichten": MLPClassifier(hidden_layer_sizes=(100, 100, 100),
            solver='sgd', activation='tanh', learning_rate_init=0.01, momentum=0.9, max_iter=5000),
        "NN - 4 Schichten": MLPClassifier(hidden_layer_sizes=(100, 100, 100, 100),
            solver='sgd', activation='tanh', learning_rate_init=0.01, momentum=0.9, max_iter=5000),
        "NN - 5 Schichten": MLPClassifier(hidden_layer_sizes=(100, 100, 100, 100, 100),
            solver='sgd', activation='tanh', learning_rate_init=0.01, momentum=0.9, max_iter=5000),
        "NN - 6 Schichten": MLPClassifier(hidden_layer_sizes=(100, 100, 100, 100, 100, 100),
            solver='sgd', activation='tanh', learning_rate_init=0.01, momentum=0.9, max_iter=5000),
        "NN - 7 Schichten": MLPClassifier(hidden_layer_sizes=(100, 100, 100, 100, 100, 100, 100),
            solver='sgd', activation='tanh', learning_rate_init=0.01, momentum=0.9, max_iter=5000),
        "NN - 8 Schichten": MLPClassifier(hidden_layer_sizes=(100, 100, 100, 100, 100, 100, 100, 100),
            solver='sgd', activation='tanh', learning_rate_init=0.01, momentum=0.9, max_iter=5000),
        "NN - 9 Schichten": MLPClassifier(hidden_layer_sizes=(100, 100, 100, 100, 100, 100, 100, 100, 100),
            solver='sgd', activation='tanh', learning_rate_init=0.01, momentum=0.9, max_iter=5000),
        "NN - 10 Schichten": MLPClassifier(hidden_layer_sizes=(100, 100, 100, 100, 100, 100, 100, 100, 100, 100),
            solver='sgd', activation='tanh', learning_rate_init=0.01, momentum=0.9, max_iter=5000)
    }

    # Neuronale Netz-Modelle trainieren und bewerten
    accuracy_scores = []
    for name, model in nn_sgd_models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        accuracy_scores.append(accuracy)
        print(f'{name} Genauigkeit: {accuracy:.4f}')

    # Ergebnisse visualisieren
    plt.figure(figsize=(15, 6))
    plt.bar(nn_sgd_models.keys(), accuracy_scores, color='skyblue')
    plt.xticks(rotation=45)
    plt.ylabel('Genauigkeit')
    plt.title('Vergleich von Neuronalen Netz-Modellen 1-10 Schichten')
    plt.show()

# nur relu & tanh
def get_accuracy_nn5():
    # Neuronale Netz-Modelle mit verschiedenen Hyperparametern definieren
    nn_models = {
        "NN - 50 Neuronen, relu": MLPClassifier(hidden_layer_sizes=(50,), activation='relu', solver='sgd', learning_rate_init=0.01, max_iter=1000),
        "NN - 100 Neuronen, relu": MLPClassifier(hidden_layer_sizes=(100,), activation='relu', solver='sgd', learning_rate_init=0.01, max_iter=1000),
        "NN - 150 Neuronen, relu": MLPClassifier(hidden_layer_sizes=(150,), activation='relu', solver='sgd', learning_rate_init=0.01, max_iter=1000),
        "NN - 200 Neuronen, relu": MLPClassifier(hidden_layer_sizes=(200,), activation='relu', solver='sgd', learning_rate_init=0.01, max_iter=1000),
        "NN - 300 Neuronen, relu": MLPClassifier(hidden_layer_sizes=(300,), activation='relu', solver='sgd', learning_rate_init=0.01, max_iter=1000),
        "NN - 50 Neuronen, tanh": MLPClassifier(hidden_layer_sizes=(50,), activation='tanh', solver='sgd', learning_rate_init=0.01, max_iter=1000),
        "NN - 100 Neuronen, tanh": MLPClassifier(hidden_layer_sizes=(100,), activation='tanh', solver='sgd', learning_rate_init=0.01, max_iter=1000),
        "NN - 150 Neuronen, tanh": MLPClassifier(hidden_layer_sizes=(150,), activation='tanh', solver='sgd', learning_rate_init=0.01, max_iter=1000),
        "NN - 200 Neuronen, tanh": MLPClassifier(hidden_layer_sizes=(200,), activation='tanh', solver='sgd', learning_rate_init=0.01, max_iter=1000),
        "NN - 300 Neuronen, tanh": MLPClassifier(hidden_layer_sizes=(300,), activation='tanh', solver='sgd', learning_rate_init=0.01, max_iter=1000)
    }

    # Neuronale Netz-Modelle trainieren und bewerten
    accuracy_scores = []
    for name, model in nn_models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        accuracy_scores.append(accuracy)
        print(f'{name} Genauigkeit: {accuracy:.4f}')

    # Ergebnisse visualisieren
    plt.figure(figsize=(15, 6))
    plt.bar(nn_models.keys(), accuracy_scores, color='skyblue')
    plt.xticks(rotation=45)
    plt.ylabel('Genauigkeit')
    plt.title('Vergleich von Neuronalen Netz-Modellen mit einer Schicht und verschiedenen Parametern')
    plt.show()
    
# nur relu & tanh (x10)
def get_accuracy_nn6():
    # Neuronale Netz-Modelle mit verschiedenen Hyperparametern definieren
    nn_models = {
        "NN - 50 Neuronen, relu": MLPClassifier(hidden_layer_sizes=(50,), activation='relu', solver='sgd', learning_rate_init=0.01, max_iter=1000),
        "NN - 100 Neuronen, relu": MLPClassifier(hidden_layer_sizes=(100,), activation='relu', solver='sgd', learning_rate_init=0.01, max_iter=1000),
        "NN - 150 Neuronen, relu": MLPClassifier(hidden_layer_sizes=(150,), activation='relu', solver='sgd', learning_rate_init=0.01, max_iter=1000),
        "NN - 200 Neuronen, relu": MLPClassifier(hidden_layer_sizes=(200,), activation='relu', solver='sgd', learning_rate_init=0.01, max_iter=1000),
        "NN - 300 Neuronen, relu": MLPClassifier(hidden_layer_sizes=(300,), activation='relu', solver='sgd', learning_rate_init=0.01, max_iter=1000),
        "NN - 50 Neuronen, tanh": MLPClassifier(hidden_layer_sizes=(50,), activation='tanh', solver='sgd', learning_rate_init=0.01, max_iter=1000),
        "NN - 100 Neuronen, tanh": MLPClassifier(hidden_layer_sizes=(100,), activation='tanh', solver='sgd', learning_rate_init=0.01, max_iter=1000),
        "NN - 150 Neuronen, tanh": MLPClassifier(hidden_layer_sizes=(150,), activation='tanh', solver='sgd', learning_rate_init=0.01, max_iter=1000),
        "NN - 200 Neuronen, tanh": MLPClassifier(hidden_layer_sizes=(200,), activation='tanh', solver='sgd', learning_rate_init=0.01, max_iter=1000),
        "NN - 300 Neuronen, tanh": MLPClassifier(hidden_layer_sizes=(300,), activation='tanh', solver='sgd', learning_rate_init=0.01, max_iter=1000)
    }

    # Neuronale Netz-Modelle mehrmals trainieren und Genauigkeit bewerten
    accuracy_scores = {name: [] for name in nn_models}

    for name, model in nn_models.items():
        for _ in range(10):  # Jedes Modell 10 Mal laufen lassen
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)
            accuracy_scores[name].append(accuracy)

    # Mittelwerte der Genauigkeit berechnen
    average_accuracies = {name: np.mean(scores) for name, scores in accuracy_scores.items()}
    print("Durchschnittliche Genauigkeit für jedes Modell:")
    for name, avg_accuracy in average_accuracies.items():
        print(f'{name}: {avg_accuracy:.4f}')

    # Ergebnisse visualisieren
    plt.figure(figsize=(15, 6))
    plt.bar(average_accuracies.keys(), average_accuracies.values(), color='skyblue')
    plt.xticks(rotation=45)
    plt.ylabel('Durchschnittliche Genauigkeit')
    plt.title('Durchschnittliche Genauigkeit von Neuronalen Netz-Modellen (SGD) mit verschiedenen Hyperparametern')
    plt.show()

# nur tanh (x10) 10-512 schichten  
def get_accuracy_nn7():
    # Neuronale Netz-Modelle mit verschiedenen Hyperparametern definieren
    nn_models = {
        "NN - 50 Neuronen, relu": MLPClassifier(hidden_layer_sizes=(50,), activation='relu', solver='sgd', learning_rate_init=0.01, max_iter=1000),
        "NN - 50 Neuronen, relu": MLPClassifier(hidden_layer_sizes=(50,), activation='relu', solver='sgd', learning_rate_init=0.01, max_iter=1000),
        "NN - 100 Neuronen, relu": MLPClassifier(hidden_layer_sizes=(100,), activation='relu', solver='sgd', learning_rate_init=0.01, max_iter=1000),
        "NN - 150 Neuronen, relu": MLPClassifier(hidden_layer_sizes=(150,), activation='relu', solver='sgd', learning_rate_init=0.01, max_iter=1000),
        "NN - 200 Neuronen, relu": MLPClassifier(hidden_layer_sizes=(200,), activation='relu', solver='sgd', learning_rate_init=0.01, max_iter=1000),
        "NN - 250 Neuronen, relu": MLPClassifier(hidden_layer_sizes=(250,), activation='relu', solver='sgd', learning_rate_init=0.01, max_iter=1000),
        "NN - 300 Neuronen, relu": MLPClassifier(hidden_layer_sizes=(300,), activation='relu', solver='sgd', learning_rate_init=0.01, max_iter=1000),
        "NN - 350 Neuronen, relu": MLPClassifier(hidden_layer_sizes=(350,), activation='relu', solver='sgd', learning_rate_init=0.01, max_iter=1000),
        "NN - 400 Neuronen, relu": MLPClassifier(hidden_layer_sizes=(400,), activation='relu', solver='sgd', learning_rate_init=0.01, max_iter=1000),
        "NN - 450 Neuronen, relu": MLPClassifier(hidden_layer_sizes=(450,), activation='relu', solver='sgd', learning_rate_init=0.01, max_iter=1000),
        "NN - 500 Neuronen, relu": MLPClassifier(hidden_layer_sizes=(500,), activation='relu', solver='sgd', learning_rate_init=0.01, max_iter=1000)
    }

    # Neuronale Netz-Modelle mehrmals trainieren und Genauigkeit bewerten
    accuracy_scores = {name: [] for name in nn_models}

    for name, model in nn_models.items():
        for _ in range(10):  # Jedes Modell 10 Mal laufen lassen
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)
            accuracy_scores[name].append(accuracy)

    # Mittelwerte der Genauigkeit berechnen
    average_accuracies = {name: np.mean(scores) for name, scores in accuracy_scores.items()}
    print("Durchschnittliche Genauigkeit für jedes Modell:")
    for name, avg_accuracy in average_accuracies.items():
        print(f'{name}: {avg_accuracy:.4f}')

    # Ergebnisse visualisieren
    plt.figure(figsize=(15, 6))
    plt.bar(average_accuracies.keys(), average_accuracies.values(), color='skyblue')
    plt.xticks(rotation=45)
    plt.ylabel('Durchschnittliche Genauigkeit')
    plt.title('Durchschnittliche Genauigkeit von Neuronalen Netz-Modellen (SGD) mit verschiedenen Hyperparametern')
    plt.show()

# nur tanh (x10)
def get_accuracy_nn8():
    # Neuronale Netz-Modelle mit verschiedenen Hyperparametern definieren
    nn_models = {
        "150 Neur, learning_rate_init=0.01": MLPClassifier(hidden_layer_sizes=(50,), activation='relu', solver='sgd', learning_rate_init=0.01, max_iter=5000),
        "150 Neur, learning_rate_init=0.05": MLPClassifier(hidden_layer_sizes=(50,), activation='relu', solver='sgd', learning_rate_init=0.05, max_iter=5000),
        "150 Neur, learning_rate_init=0.10": MLPClassifier(hidden_layer_sizes=(50,), activation='relu', solver='sgd', learning_rate_init=0.10, max_iter=5000),
        "150 Neur, learning_rate_init=0.50": MLPClassifier(hidden_layer_sizes=(50,), activation='relu', solver='sgd', learning_rate_init=0.50, max_iter=5000),
        "150 Neur, learning_rate_init=0.75": MLPClassifier(hidden_layer_sizes=(50,), activation='relu', solver='sgd', learning_rate_init=0.75, max_iter=5000),
        "150 Neur, learning_rate_init=0.90": MLPClassifier(hidden_layer_sizes=(50,), activation='relu', solver='sgd', learning_rate_init=0.90, max_iter=5000),
        "150 Neur, learning_rate_init=1.00": MLPClassifier(hidden_layer_sizes=(50,), activation='relu', solver='sgd', learning_rate_init=1.00, max_iter=5000),
        "150 Neur, learning_rate_init=5.00": MLPClassifier(hidden_layer_sizes=(50,), activation='relu', solver='sgd', learning_rate_init=5.00, max_iter=5000),
    }

    # Neuronale Netz-Modelle mehrmals trainieren und Genauigkeit bewerten
    accuracy_scores = {name: [] for name in nn_models}

    for name, model in nn_models.items():
        for _ in range(10):  # Jedes Modell 10 Mal laufen lassen
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)
            accuracy_scores[name].append(accuracy)

    # Mittelwerte der Genauigkeit berechnen
    average_accuracies = {name: np.mean(scores) for name, scores in accuracy_scores.items()}
    print("Durchschnittliche Genauigkeit für jedes Modell:")
    for name, avg_accuracy in average_accuracies.items():
        print(f'{name}: {avg_accuracy:.4f}')

    # Ergebnisse visualisieren
    plt.figure(figsize=(15, 6))
    plt.bar(average_accuracies.keys(), average_accuracies.values(), color='skyblue')
    plt.xticks(rotation=45)
    plt.ylabel('Durchschnittliche Genauigkeit')
    plt.title('Durchschnittliche Genauigkeit von Neuronalen Netz-Modellen (SGD) mit verschiedenen Hyperparametern')
    plt.show()

# Anzahl Schichten
def get_accuracy_nn9():
    # Neuronale Netz-Modelle mit verschiedenen Hyperparametern definieren
    nn_models_50 = {
        "NN - 1 Schicht, 50 Neuronen, relu": MLPClassifier(hidden_layer_sizes=(50,), activation='relu', solver='sgd', learning_rate_init=0.01, max_iter=1000),
        "NN - 2 Schichten, 50 Neuronen, relu": MLPClassifier(hidden_layer_sizes=(50, 50), activation='relu', solver='sgd', learning_rate_init=0.01, max_iter=1000),
        "NN - 3 Schichten, 50 Neuronen, relu": MLPClassifier(hidden_layer_sizes=(50, 50, 50), activation='relu', solver='sgd', learning_rate_init=0.01, max_iter=1000),
        "NN - 4 Schichten, 50 Neuronen, relu": MLPClassifier(hidden_layer_sizes=(50, 50, 50, 50), activation='relu', solver='sgd', learning_rate_init=0.01, max_iter=1000),
        "NN - 5 Schichten, 50 Neuronen, relu": MLPClassifier(hidden_layer_sizes=(50, 50, 50, 50, 50), activation='relu', solver='sgd', learning_rate_init=0.01, max_iter=1000),
        "NN - 6 Schichten, 50 Neuronen, relu": MLPClassifier(hidden_layer_sizes=(50, 50, 50, 50, 50, 50), activation='relu', solver='sgd', learning_rate_init=0.01, max_iter=1000),
        "NN - 7 Schichten, 50 Neuronen, relu": MLPClassifier(hidden_layer_sizes=(50, 50, 50, 50, 50, 50, 50), activation='relu', solver='sgd', learning_rate_init=0.01, max_iter=1000),
        "NN - 8 Schichten, 50 Neuronen, relu": MLPClassifier(hidden_layer_sizes=(50, 50, 50, 50, 50, 50, 50, 50), activation='relu', solver='sgd', learning_rate_init=0.01, max_iter=1000),
        "NN - 9 Schichten, 50 Neuronen, relu": MLPClassifier(hidden_layer_sizes=(50, 50, 50, 50, 50, 50, 50, 50, 50), activation='relu', solver='sgd', learning_rate_init=0.01, max_iter=1000),
        "NN - 10 Schichten, 50 Neuronen, relu": MLPClassifier(hidden_layer_sizes=(50, 50, 50, 50, 50, 50, 50, 50, 50, 50), activation='relu', solver='sgd', learning_rate_init=0.01, max_iter=1000)
    }
    nn_models_100 = {
        "NN - 1 Schicht, 100 Neuronen, relu": MLPClassifier(hidden_layer_sizes=(100,), activation='relu', solver='sgd', learning_rate_init=0.01, max_iter=1000),
        "NN - 2 Schichten, 100 Neuronen, relu": MLPClassifier(hidden_layer_sizes=(100, 100), activation='relu', solver='sgd', learning_rate_init=0.01, max_iter=1000),
        "NN - 3 Schichten, 100 Neuronen, relu": MLPClassifier(hidden_layer_sizes=(100, 100, 100), activation='relu', solver='sgd', learning_rate_init=0.01, max_iter=1000),
        "NN - 4 Schichten, 100 Neuronen, relu": MLPClassifier(hidden_layer_sizes=(100, 100, 100, 100), activation='relu', solver='sgd', learning_rate_init=0.01, max_iter=1000),
        "NN - 5 Schichten, 100 Neuronen, relu": MLPClassifier(hidden_layer_sizes=(100, 100, 100, 100, 100), activation='relu', solver='sgd', learning_rate_init=0.01, max_iter=1000),
        "NN - 6 Schichten, 100 Neuronen, relu": MLPClassifier(hidden_layer_sizes=(100, 100, 100, 100, 100, 100), activation='relu', solver='sgd', learning_rate_init=0.01, max_iter=1000),
        "NN - 7 Schichten, 100 Neuronen, relu": MLPClassifier(hidden_layer_sizes=(100, 100, 100, 100, 100, 100, 100), activation='relu', solver='sgd', learning_rate_init=0.01, max_iter=1000),
        "NN - 8 Schichten, 100 Neuronen, relu": MLPClassifier(hidden_layer_sizes=(100, 100, 100, 100, 100, 100, 100, 100), activation='relu', solver='sgd', learning_rate_init=0.01, max_iter=1000),
        "NN - 9 Schichten, 100 Neuronen, relu": MLPClassifier(hidden_layer_sizes=(100, 100, 100, 100, 100, 100, 100, 100, 100), activation='relu', solver='sgd', learning_rate_init=0.01, max_iter=1000),
        "NN - 10 Schichten, 100 Neuronen, relu": MLPClassifier(hidden_layer_sizes=(100, 100, 100, 100, 100, 100, 100, 100, 100, 100), activation='relu', solver='sgd', learning_rate_init=0.01, max_iter=1000)
    }
    # Neuronale Netz-Modelle mit 150 Neuronen definieren
    nn_models_150 = {
        "NN - 1 Schicht, 150 Neuronen, relu": MLPClassifier(hidden_layer_sizes=(150,), activation='relu', solver='sgd', learning_rate_init=0.01, max_iter=1000),
        "NN - 2 Schichten, 150 Neuronen, relu": MLPClassifier(hidden_layer_sizes=(150, 150), activation='relu', solver='sgd', learning_rate_init=0.01, max_iter=1000),
        "NN - 3 Schichten, 150 Neuronen, relu": MLPClassifier(hidden_layer_sizes=(150, 150, 150), activation='relu', solver='sgd', learning_rate_init=0.01, max_iter=1000),
        "NN - 4 Schichten, 150 Neuronen, relu": MLPClassifier(hidden_layer_sizes=(150, 150, 150, 150), activation='relu', solver='sgd', learning_rate_init=0.01, max_iter=1000),
        "NN - 5 Schichten, 150 Neuronen, relu": MLPClassifier(hidden_layer_sizes=(150, 150, 150, 150, 150), activation='relu', solver='sgd', learning_rate_init=0.01, max_iter=1000),
        "NN - 6 Schichten, 150 Neuronen, relu": MLPClassifier(hidden_layer_sizes=(150, 150, 150, 150, 150, 150), activation='relu', solver='sgd', learning_rate_init=0.01, max_iter=1000),
        "NN - 7 Schichten, 150 Neuronen, relu": MLPClassifier(hidden_layer_sizes=(150, 150, 150, 150, 150, 150, 150), activation='relu', solver='sgd', learning_rate_init=0.01, max_iter=1000),
        "NN - 8 Schichten, 150 Neuronen, relu": MLPClassifier(hidden_layer_sizes=(150, 150, 150, 150, 150, 150, 150, 150), activation='relu', solver='sgd', learning_rate_init=0.01, max_iter=1000),
        "NN - 9 Schichten, 150 Neuronen, relu": MLPClassifier(hidden_layer_sizes=(150, 150, 150, 150, 150, 150, 150, 150, 150), activation='relu', solver='sgd', learning_rate_init=0.01, max_iter=1000),
        "NN - 10 Schichten, 150 Neuronen, relu": MLPClassifier(hidden_layer_sizes=(150, 150, 150, 150, 150, 150, 150, 150, 150, 150), activation='relu', solver='sgd', learning_rate_init=0.01, max_iter=1000)
    }

    # Neuronale Netz-Modelle mit 200 Neuronen definieren
    nn_models_200 = {
        "NN - 1 Schicht, 200 Neuronen, relu": MLPClassifier(hidden_layer_sizes=(200,), activation='relu', solver='sgd', learning_rate_init=0.01, max_iter=1000),
        "NN - 2 Schichten, 200 Neuronen, relu": MLPClassifier(hidden_layer_sizes=(200, 200), activation='relu', solver='sgd', learning_rate_init=0.01, max_iter=1000),
        "NN - 3 Schichten, 200 Neuronen, relu": MLPClassifier(hidden_layer_sizes=(200, 200, 200), activation='relu', solver='sgd', learning_rate_init=0.01, max_iter=1000),
        "NN - 4 Schichten, 200 Neuronen, relu": MLPClassifier(hidden_layer_sizes=(200, 200, 200, 200), activation='relu', solver='sgd', learning_rate_init=0.01, max_iter=1000),
        "NN - 5 Schichten, 200 Neuronen, relu": MLPClassifier(hidden_layer_sizes=(200, 200, 200, 200, 200), activation='relu', solver='sgd', learning_rate_init=0.01, max_iter=1000),
        "NN - 6 Schichten, 200 Neuronen, relu": MLPClassifier(hidden_layer_sizes=(200, 200, 200, 200, 200, 200), activation='relu', solver='sgd', learning_rate_init=0.01, max_iter=1000),
        "NN - 7 Schichten, 200 Neuronen, relu": MLPClassifier(hidden_layer_sizes=(200, 200, 200, 200, 200, 200, 200), activation='relu', solver='sgd', learning_rate_init=0.01, max_iter=1000),
        "NN - 8 Schichten, 200 Neuronen, relu": MLPClassifier(hidden_layer_sizes=(200, 200, 200, 200, 200, 200, 200, 200), activation='relu', solver='sgd', learning_rate_init=0.01, max_iter=1000),
        "NN - 9 Schichten, 200 Neuronen, relu": MLPClassifier(hidden_layer_sizes=(200, 200, 200, 200, 200, 200, 200, 200, 200), activation='relu', solver='sgd', learning_rate_init=0.01, max_iter=1000),
        "NN - 10 Schichten, 200 Neuronen, relu": MLPClassifier(hidden_layer_sizes=(200, 200, 200, 200, 200, 200, 200, 200, 200, 200), activation='relu', solver='sgd', learning_rate_init=0.01, max_iter=1000)
    }

    # Neuronale Netz-Modelle trainieren und bewerten
    accuracy_scores_50 = []
    for name, model in nn_models_50.items():
        accuracies = []
        for _ in range(10):
            X_train_scaled, X_test_scaled, y_train, y_test = test_data()
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)
            accuracies.append(accuracy)
        average_accuracy = np.mean(accuracies)
        accuracy_scores_50.append(average_accuracy)
        print(f'{name} Durchschnittliche Genauigkeit: {average_accuracy:.4f}')

    accuracy_scores_100 = []
    for name, model in nn_models_100.items():
        accuracies = []
        for _ in range(10):
            X_train_scaled, X_test_scaled, y_train, y_test = test_data()
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)
            accuracies.append(accuracy)
        average_accuracy = np.mean(accuracies)
        accuracy_scores_100.append(average_accuracy)
        print(f'{name} Durchschnittliche Genauigkeit: {average_accuracy:.4f}')

    accuracy_scores_150 = []
    for name, model in nn_models_150.items():
        accuracies = []
        for _ in range(10):
            X_train_scaled, X_test_scaled, y_train, y_test = test_data()
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)
            accuracies.append(accuracy)
        average_accuracy = np.mean(accuracies)
        accuracy_scores_150.append(average_accuracy)
        print(f'{name} Durchschnittliche Genauigkeit: {average_accuracy:.4f}')

    accuracy_scores_200 = []
    for name, model in nn_models_200.items():
        accuracies = []
        for _ in range(10):
            X_train_scaled, X_test_scaled, y_train, y_test = test_data()
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)
            accuracies.append(accuracy)
        average_accuracy = np.mean(accuracies)
        accuracy_scores_200.append(average_accuracy)
        print(f'{name} Durchschnittliche Genauigkeit: {average_accuracy:.4f}')

    # Ergebnisse visualisieren
    plt.figure(figsize=(15, 6))
    plt.plot(range(1, 11), accuracy_scores_50, color='skyblue')
    plt.plot(range(1, 11), accuracy_scores_100, color='orange')
    plt.plot(range(1, 11), accuracy_scores_150, color='green')
    plt.plot(range(1, 11), accuracy_scores_200, color='purple')
    plt.xticks(rotation=0)
    plt.xlabel('Anzahl der Schichten')
    plt.ylabel('Genauigkeit')
    plt.legend(['50 Neuronen', '100 Neuronen', '150 Neuronen', '200 Neuronen'])
    plt.title('Vergleich von Neuronalen Netz-Modellen mit variierender Anzahl an Schichten (x10)')
    # plt.show()

    # Generate a random filename
    filename = f"plot_{int(time.time())}.png"
    desktop_path = os.path.join(os.path.expanduser("~"), "Desktop/Test_NN_Anzahl_Schichten")
    if not os.path.exists(desktop_path):
        os.makedirs(desktop_path)
    plt.savefig(os.path.join(desktop_path, filename))





def get_accuracy_svm():
    # SVM-Modelle mit verschiedenen Hyperparametern definieren
    svm_models = {
        "SVM - C=0.1, linear": SVC(kernel='linear', C=0.1),
        "SVM - C=1, linear": SVC(kernel='linear', C=1),
        "SVM - C=10, linear": SVC(kernel='linear', C=10),
        "SVM - C=1, rbf": SVC(kernel='rbf', C=1),
        "SVM - C=1, rbf, gamma=0.01": SVC(kernel='rbf', C=1, gamma=0.01),
        "SVM - C=1, rbf, gamma=1": SVC(kernel='rbf', C=1, gamma=1),
        "SVM - C=1, poly": SVC(kernel='poly', C=1),
        "SVM - C=1, poly, degree=3": SVC(kernel='poly', C=1, degree=3),
        "SVM - C=1, poly, degree=5": SVC(kernel='poly', C=1, degree=5)
    }

    # SVM-Modelle trainieren und bewerten
    accuracy_scores = []
    X_train_scaled, X_test_scaled, y_train, y_test = test_data()
    
    for name, model in svm_models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        accuracy_scores.append(accuracy)
        print(f'{name} Genauigkeit: {accuracy:.4f}')

    # Ergebnisse visualisieren
    plt.figure(figsize=(15, 6))
    plt.bar(svm_models.keys(), accuracy_scores, color='skyblue')
    plt.xticks(rotation=45)
    plt.ylabel('Genauigkeit')
    plt.title('SVM-Modellvergleich mit verschiedenen Hyperparametern')
    plt.show()

# SVM (balanced/unbalanced)
def get_accuracy_svm2():
        
    svm_models_standart = {
        "SVM - C=0.1, linear": SVC(kernel='linear', C=0.1),
        "SVM - C=1, linear": SVC(kernel='linear', C=1),
        "SVM - C=10, linear": SVC(kernel='linear', C=10),
        "SVM - C=0.1, rbf": SVC(kernel='rbf', C=0.1),
        "SVM - C=1, rbf": SVC(kernel='rbf', C=1),
        "SVM - C=10, rbf": SVC(kernel='rbf', C=10),
        "SVM - C=0.1, poly": SVC(kernel='poly', C=0.1),
        "SVM - C=1, poly": SVC(kernel='poly', C=1),
        "SVM - C=10, poly": SVC(kernel='poly', C=10),
        "SVM - C=0.1, sigmoid": SVC(kernel='sigmoid', C=0.1),
        "SVM - C=1, sigmoid": SVC(kernel='sigmoid', C=1),
        "SVM - C=10, sigmoid": SVC(kernel='sigmoid', C=10)
        }
    
    svm_models_balanced = {
        "SVM - C=0.1, linear": SVC(kernel='linear', C=0.1, class_weight='balanced'),
        "SVM - C=1, linear": SVC(kernel='linear', C=1, class_weight='balanced'),
        "SVM - C=10, linear": SVC(kernel='linear', C=10, class_weight='balanced'),
        "SVM - C=0.1, rbf": SVC(kernel='rbf', C=0.1, class_weight='balanced'),
        "SVM - C=1, rbf": SVC(kernel='rbf', C=1, class_weight='balanced'),
        "SVM - C=10, rbf": SVC(kernel='rbf', C=10, class_weight='balanced'),
        "SVM - C=0.1, poly": SVC(kernel='poly', C=0.1, class_weight='balanced'),
        "SVM - C=1, poly": SVC(kernel='poly', C=1, class_weight='balanced'),
        "SVM - C=10, poly": SVC(kernel='poly', C=10, class_weight='balanced'),
        "SVM - C=0.1, sigmoid": SVC(kernel='sigmoid', C=0.1, class_weight='balanced'),
        "SVM - C=1, sigmoid": SVC(kernel='sigmoid', C=1, class_weight='balanced'),
        "SVM - C=10, sigmoid": SVC(kernel='sigmoid', C=10, class_weight='balanced')
    }

    # Neuronale Netz-Modelle trainieren und bewerten

    accuracy_scores = [[],[]]
    X_train_scaled, X_test_scaled, y_train, y_test = test_data()
    
    for name, model in svm_models_standart.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        accuracy_scores[0].append(accuracy)
        print(f'{name} Genauigkeit: {accuracy:.4f}')

    for name, model in svm_models_balanced.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        accuracy_scores[1].append(accuracy)
        print(f'{name} Genauigkeit: {accuracy:.4f}')


    # Ergebnisse visualisieren
    plt.figure(figsize=(15, 6))
    plt.plot(svm_models_standart.keys(), accuracy_scores[0], color='skyblue')
    plt.plot(svm_models_balanced.keys(), accuracy_scores[1], color='orange')
    plt.xticks(rotation=45)
    plt.xlabel('Modelle')
    plt.ylabel('Genauigkeit')
    plt.legend(['unbalanced', 'balanced'])
    plt.title('Vergleich von SVM-Modellen balanced/unbalanced')
    #plt.show()

    # Generate a random filename

    filename = f"plot_{int(time.time())}.png"
    desktop_path = os.path.join(os.path.expanduser("~"), "Desktop/Test_SVM_balanced_unbalanced")
    # Überprüfen, ob das Verzeichnis vorhanden ist, andernfalls erstellen
    if not os.path.exists(desktop_path):
        os.makedirs(desktop_path)
    plt.savefig(os.path.join(desktop_path, filename))
 
# linear SVM (C)
def get_accuracy_svm3():
    
    svm_models_balanced = {
        "SVM - C=0.1, linear": SVC(kernel='linear', C=0.1, class_weight='balanced'),
        "SVM - C=0.5, linear": SVC(kernel='linear', C=0.5, class_weight='balanced'),
        "SVM - C=1, linear": SVC(kernel='linear', C=1, class_weight='balanced'),
        "SVM - C=5, linear": SVC(kernel='linear', C=5, class_weight='balanced'),
        "SVM - C=10, linear": SVC(kernel='linear', C=10, class_weight='balanced'),
        "SVM - C=50, linear": SVC(kernel='linear', C=50, class_weight='balanced'),
    }

    # Neuronale Netz-Modelle trainieren und bewerten

    accuracy_scores = []
    X_train_scaled, X_test_scaled, y_train, y_test = test_data()
    
    for name, model in svm_models_balanced.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        accuracy_scores.append(accuracy)
        print(f'{name} Genauigkeit: {accuracy:.4f}')


    # Ergebnisse visualisieren
    plt.figure(figsize=(15, 6))
    plt.plot(svm_models_balanced.keys(), accuracy_scores, color='skyblue')
    plt.xticks(rotation=45)
    plt.xlabel('Modelle')
    plt.ylabel('Genauigkeit')
    plt.title('Vergleich von linear SVM-Modellen (C)')
    plt.show()

    # Generate a random filename

    filename = f"plot_{int(time.time())}.png"
    desktop_path = os.path.join(os.path.expanduser("~"), "Desktop/Test_linear_SVM_C") 
    # Überprüfen, ob das Verzeichnis vorhanden ist, andernfalls erstellen
    if not os.path.exists(desktop_path):
        os.makedirs(desktop_path)
    plt.savefig(os.path.join(desktop_path, filename))
 
# rbf SVM (gamma)
def get_accuracy_svm4():
    
    svm_models_rbf_c1 = {
        "SVM - C=1, rbf": SVC(kernel='rbf', C=1),
        "SVM - C=1, rbf, gamma=0.01": SVC(kernel='rbf', C=1, gamma=0.01),
        "SVM - C=1, rbf, gamma=0.1": SVC(kernel='rbf', C=1, gamma=0.1),
        "SVM - C=1, rbf, gamma=1": SVC(kernel='rbf', C=1, gamma=1),
        "SVM - C=1, rbf, gamma=10": SVC(kernel='rbf', C=1, gamma=10),
    }
    
    svm_models_rbf_c10 = {
        "SVM - C=1, rbf": SVC(kernel='rbf', C=10),
        "SVM - C=1, rbf, gamma=0.01": SVC(kernel='rbf', C=10, gamma=0.01),
        "SVM - C=1, rbf, gamma=0.1": SVC(kernel='rbf', C=10, gamma=0.1),
        "SVM - C=1, rbf, gamma=1": SVC(kernel='rbf', C=10, gamma=1),
        "SVM - C=1, rbf, gamma=10": SVC(kernel='rbf', C=10, gamma=10),
    }

    # Neuronale Netz-Modelle trainieren und bewerten

    accuracy_scores = [[],[]]
    X_train_scaled, X_test_scaled, y_train, y_test = test_data()
    
    for name, model in svm_models_rbf_c1.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        accuracy_scores[0].append(accuracy)
        print(f'{name} Genauigkeit: {accuracy:.4f}')
        
    for name, model in svm_models_rbf_c10.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        accuracy_scores[1].append(accuracy)
        print(f'{name} Genauigkeit: {accuracy:.4f}')


    # Ergebnisse visualisieren
    plt.figure(figsize=(15, 6))
    plt.plot(["-","0,01","0,1","1","10"], accuracy_scores[0], color='skyblue')
    plt.plot(["-","0,01","0,1","1","10"], accuracy_scores[1], color='orange')
    plt.xticks(rotation=45)
    plt.legend(['C=1', 'C=10'])
    plt.xlabel('Modelle')
    plt.ylabel('Genauigkeit')
    plt.title('Vergleich von rbf SVM-Modellen (gamma & C)')
    plt.show()

    # Generate a random filename

    filename = f"plot_{int(time.time())}.png"
    desktop_path = os.path.join(os.path.expanduser("~"), "Desktop/Test_rbf_SVM_gamma_C") 
    # Überprüfen, ob das Verzeichnis vorhanden ist, andernfalls erstellen
    if not os.path.exists(desktop_path):
        os.makedirs(desktop_path)
    plt.savefig(os.path.join(desktop_path, filename))
 



# versch. Hyperparameter
def get_accuracy_knn():
    # KNN-Modelle mit verschiedenen Hyperparametern definieren
    knn_models = {
        "KNN - n_neighbors=3, uniform": KNeighborsClassifier(n_neighbors=3, weights='uniform'),
        "KNN - n_neighbors=5, uniform": KNeighborsClassifier(n_neighbors=5, weights='uniform'),
        "KNN - n_neighbors=7, uniform": KNeighborsClassifier(n_neighbors=7, weights='uniform'),
        "KNN - n_neighbors=3, distance": KNeighborsClassifier(n_neighbors=3, weights='distance'),
        "KNN - n_neighbors=5, distance": KNeighborsClassifier(n_neighbors=5, weights='distance'),
        "KNN - n_neighbors=7, distance": KNeighborsClassifier(n_neighbors=7, weights='distance'),
        "KNN - n_neighbors=5, ball_tree": KNeighborsClassifier(n_neighbors=5, algorithm='ball_tree'),
        "KNN - n_neighbors=5, kd_tree": KNeighborsClassifier(n_neighbors=5, algorithm='kd_tree'),
        "KNN - n_neighbors=5, brute": KNeighborsClassifier(n_neighbors=5, algorithm='brute')
    }

    # KNN-Modelle trainieren und bewerten
    accuracy_scores = []
    for name, model in knn_models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        accuracy_scores.append(accuracy)
        print(f'{name} Genauigkeit: {accuracy:.4f}')

    # Ergebnisse visualisieren
    plt.figure(figsize=(15, 6))
    plt.bar(knn_models.keys(), accuracy_scores, color='skyblue')
    plt.xticks(rotation=45)
    plt.ylabel('Genauigkeit')
    plt.title('KNN-Modellvergleich mit verschiedenen Hyperparametern')
    plt.show()
 
# versch. Hyperparameter (x100)
def get_accuracy_knn2():
    # KNN-Modelle mit verschiedenen Hyperparametern definieren
    knn_models = {
        "KNN - n_neighbors=3, uniform": KNeighborsClassifier(n_neighbors=3, weights='uniform'),
        "KNN - n_neighbors=5, uniform": KNeighborsClassifier(n_neighbors=5, weights='uniform'),
        "KNN - n_neighbors=7, uniform": KNeighborsClassifier(n_neighbors=7, weights='uniform'),
        "KNN - n_neighbors=3, distance": KNeighborsClassifier(n_neighbors=3, weights='distance'),
        "KNN - n_neighbors=5, distance": KNeighborsClassifier(n_neighbors=5, weights='distance'),
        "KNN - n_neighbors=7, distance": KNeighborsClassifier(n_neighbors=7, weights='distance'),
        "KNN - n_neighbors=5, ball_tree": KNeighborsClassifier(n_neighbors=5, algorithm='ball_tree'),
        "KNN - n_neighbors=5, kd_tree": KNeighborsClassifier(n_neighbors=5, algorithm='kd_tree'),
        "KNN - n_neighbors=5, brute": KNeighborsClassifier(n_neighbors=5, algorithm='brute')
    }

    # KNN-Modelle trainieren und bewerten
    accuracy_scores = []
    
    for name, model in knn_models.items():
        accuracies = []
        for _ in range(100):
            X_train_scaled, X_test_scaled, y_train, y_test = test_data()
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)
            accuracies.append(accuracy)
        average_accuracy = sum(accuracies) / len(accuracies)
        accuracy_scores.append(average_accuracy)
        print(f'{name} Durchschnittliche Genauigkeit: {average_accuracy:.4f}')

    # Ergebnisse visualisieren
    plt.figure(figsize=(15, 6))
    plt.plot(knn_models.keys(), accuracy_scores, marker='o', linestyle='-', color='skyblue')
    plt.xticks(rotation=45)
    plt.ylabel('Durchschnittliche Genauigkeit')
    plt.title('KNN-Modellvergleich mit verschiedenen Hyperparametern (x100)')
    plt.show()



get_accuracy_nn9()
    '''