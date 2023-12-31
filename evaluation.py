import numpy as np
import pandas as pd
import os
import time
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
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


# Bewertung der Genauigkeit
def generate_accuracy_scores(models, mean_from=10):
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

def generate_cross_val_scores(models, mean_from=10):
    index = 0
    accuracy_scores = [[], []]
    for name, model in models.items():
        accuracies = []
        for _ in range(mean_from):
            accuracy = cross_val_score(model, X, y, cv=5, n_jobs=-1)
            accuracies.append(accuracy)

        average_accuracy = np.mean(accuracies)

        accuracy_scores[0].append(name)
        accuracy_scores[1].append(average_accuracy)

        index += 1
        print(f'{name.ljust(50)} Genauigkeit: {average_accuracy:.4f} - {index}/{len(models)} Modelle berechnet')
        
    return accuracy_scores

def cv_score_ez(model, mean_from=10):
    accuracies = []
    for _ in range(mean_from):
        accuracy = cross_val_score(model, X, y, cv=5, n_jobs=-1)
        accuracies.append(accuracy)

    average_accuracy = np.mean(accuracies)
    return average_accuracy

def acc_score_ez(model, mean_from=10):
    accuracies = []
    for _ in range(mean_from):
        X_train_scaled, X_test_scaled, y_train, y_test = test_data()
        model.fit(X_train_scaled, y_train)
        accuracy = model.score(X_test_scaled, y_test)
        accuracies.append(accuracy)

    average_accuracy = np.mean(accuracies)
    return average_accuracy

# Plot Generierung
def plot_accuracy_scores(*accuracy_scores, xlabel, ylabel, title,  savefig=False, filename=None):
    plt.figure(figsize=(15, 10), layout='tight', dpi=100) 
    
    for i in range(len(accuracy_scores)):
        plt.plot(accuracy_scores[i][0], accuracy_scores[i][1], color=distinct_colors[i])
        
    plt.xticks(rotation=45)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    # plt.legend()
    plt.title(title)
    plt.grid()
    
    if savefig:
        file_path = os.path.join(os.path.expanduser("~"), "Desktop", filename)
        plt.savefig(file_path)
    else:
        plt.show()
        
def bar_accuracy_scores(*accuracy_scores, xlabel, ylabel, title,  savefig=False, filename=None):
    plt.figure(figsize=(15, 10), layout='tight', dpi=100) 
    
    for i in range(len(accuracy_scores)):
        plt.bar(accuracy_scores[i][0], accuracy_scores[i][1], color=distinct_colors[i])
        
    plt.xticks(rotation=45)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    # plt.legend()
    plt.title(title)
    plt.grid()
    
    if savefig:
        file_path = os.path.join(os.path.expanduser("~"), "Desktop", filename)
        plt.savefig(file_path)
    else:
        plt.show()


# berechnende Funktionen für jeweilige Modelle
def logitstic_regression(c_values=[0.01, 0.05, 0.1, 0.5, 1, 2, 5], max_iter=5000, mean_from=10):
    # Modelle mit verschiedenen Parametern definieren
    models = {}

    for c in c_values:
        model_name = f"Lineare Regression - C={c}"
        models[model_name] = LogisticRegression(C=c, max_iter=max_iter, class_weight='balanced', n_jobs=-1)

    accuracy_scores = generate_cross_val_scores(models, mean_from=mean_from)
    return accuracy_scores

def linear_regression(fit_intercept=True, copy_X=False, mean_from=10):
    # Modelle mit verschiedenen Parametern definieren
    models = {}

    # Verschiedene Kombinationen von Parametern, da bool nicht iterierbar ist
    if (fit_intercept != [True, False]) and (copy_X != [True, False]):
        model_name = f"Lineare Regression - fit_intercept=True, copy_X=True"
        models[model_name] = LinearRegression(fit_intercept=fit_intercept, copy_X=copy_X, n_jobs=-1)
    elif fit_intercept != [True, False]:
        for copy in copy_X:
            model_name = f"Lineare Regression - fit_intercept=True, copy_X={copy}"
            models[model_name] = LinearRegression(fit_intercept=fit_intercept, copy_X=copy, n_jobs=-1)
    elif copy_X != [True, False]:
        for fit in fit_intercept:
            model_name = f"Lineare Regression - fit_intercept={fit}, copy_X=True"
            models[model_name] = LinearRegression(fit_intercept=fit, copy_X=copy_X, n_jobs=-1)
    else:
        pass

    accuracy_scores = generate_accuracy_scores(models, mean_from=mean_from)
    return accuracy_scores

def svm(c_values=[50], kernel_values=['rbf'], gamma_values=[0.001], mean_from=10):
    # Modelle mit verschiedenen Parametern definieren
    models = {}

    for c in c_values:
        for kernel in kernel_values:
            for gamma in gamma_values:
                model_name = f"SVM - C={c}, Kernel={kernel}, Gamma={gamma}"
                models[model_name] = SVC(kernel=kernel, C=c, gamma=gamma, class_weight='balanced')

    accuracy_scores = generate_cross_val_scores(models, mean_from=mean_from)
    return accuracy_scores

def knn(n_neighbors=[3], weights=['uniform'], algorithm=['auto'], leaf_size=[10], p=[2], mean_from=10):
    # Modelle mit verschiedenen Parametern definieren
    models = {}

    for n in n_neighbors:
        for weight in weights:
            for algo in algorithm:
                for size in leaf_size:
                    for norm in p:
                        model_name = f"KNN - n_neighbors={n}, weights={weight}, algorithm={algo}, leaf_size={size}, p={norm}"
                        models[model_name] = KNeighborsClassifier(n_neighbors=n, weights=weight, algorithm=algo, leaf_size=size, p=norm, n_jobs=-1)

    accuracy_scores = generate_cross_val_scores(models, mean_from=mean_from)
    return accuracy_scores

def nn(hidden_layers=[(511,)], activation=['logistic'], solver=['adam'], alpha=[0.001], learning_rate_init=[0.0001], mean_from=10):
    models = {}

    for layers in hidden_layers:
        for act in activation:
            for solve in solver:
                for a in alpha:
                    for learning_rate in learning_rate_init:
                        model_name = f"Neural Network - hidden_layers={layers}, activation={act}, solver={solve}, alpha={a}, init_l_rate={learning_rate}"
                        models[model_name] = MLPClassifier(hidden_layer_sizes=layers, activation=act, solver=solve, alpha=a,max_iter=5000, learning_rate_init=learning_rate)

    accuracy_scores = generate_cross_val_scores(models, mean_from=mean_from)
    return accuracy_scores




# ausführende Funktionen für jeweilige Modelle
def test_svm(*args, mean_from=10):
    if "c" in args:
        # Test svm mit verschiedenen "C" Werten
        svm_scores = svm(c_values=[ 20, 30, 40, 50, 100, 250, 500, 1000], mean_from=mean_from)
        plot_accuracy_scores(svm_scores, xlabel="C", ylabel="Genauigkeit",
                            title="Genauigkeit der SVM mit verschiedenen C-Werten",
                            savefig=True, filename="svm_c")

    if "kernel" in args:
        # Test svm mit verschiedenen "Kernel" Werten
        svm_scores = svm(kernel_values=['linear', 'poly', 'rbf', 'sigmoid'], mean_from=mean_from)
        plot_accuracy_scores(svm_scores, xlabel="Kernel", ylabel="Genauigkeit",
                            title="Genauigkeit der SVM mit verschiedenen Kernel-Werten",
                            savefig=True, filename="svm_kernel")
        
    if "gamma" in args:    
        # Test svm mit verschiedenen "Gamma" Werten
        svm_scores = svm(gamma_values=[0.0001, 0.0005, 0.001, 0.01, 0.1, 1], mean_from=mean_from)
        plot_accuracy_scores(svm_scores, xlabel="Gamma", ylabel="Genauigkeit",
                            title="Genauigkeit der SVM mit verschiedenen Gamma-Werten",
                            savefig=True, filename="svm_gamma")

def test_logitstic_regression(mean_from=10):
    # Test linear regression mit verschiedenen "C" Werten
    logitstic_regression_scores = logitstic_regression(c_values=[0.01, 0.05, 0.1, 0.5, 1, 2, 5], mean_from=mean_from)
    plot_accuracy_scores(logitstic_regression_scores, xlabel="C", ylabel="Genauigkeit",
                        title="Genauigkeit der logische Regression mit verschiedenen C-Werten",
                        savefig=True, filename="logitstic_regression_c")

def test_linear_regression(*args, mean_from=10):
    if "fit" in args:
        # Test linear regression mit verschiedenen "fit_intercept" Werten
        linear_regression_scores = linear_regression(fit_intercept=[True, False], mean_from=mean_from)
        plot_accuracy_scores(linear_regression_scores, xlabel="fit_intercept", ylabel="Genauigkeit",
                            title="Genauigkeit der linearen Regression mit verschiedenen fit_intercept-Werten",
                            savefig=True, filename="linear_regression_fit_intercept")
    
    if "copy" in args:
        # Test linear regression mit verschiedenen "copy_X" Werten
        linear_regression_scores = linear_regression(copy_X=[True, False], mean_from=mean_from)
        plot_accuracy_scores(linear_regression_scores, xlabel="copy_X", ylabel="Genauigkeit",
                            title="Genauigkeit der linearen Regression mit verschiedenen copy_X-Werten",
                            savefig=True, filename="linear_regression_copy_X")

def test_knn(*args, mean_from=10):
    if "n" in args:
        # Test knn mit verschiedenen "n_neighbors" Werten
        knn_scores = knn(n_neighbors=[1, 2, 3, 5, 7, 9, 11, 19], mean_from=mean_from)
        plot_accuracy_scores(knn_scores, xlabel="n_neighbors", ylabel="Genauigkeit",
                            title="Genauigkeit des KNN mit verschiedenen n_neighbors-Werten",
                            savefig=True, filename="knn_n_neighbors")

    if "weights" in args:
        # Test knn mit verschiedenen "weights" Werten
        knn_scores = knn(weights=['uniform', 'distance'], mean_from=mean_from)
        plot_accuracy_scores(knn_scores, xlabel="weights", ylabel="Genauigkeit",
                        title="Genauigkeit des KNN mit verschiedenen weights-Werten",
                        savefig=True, filename="knn_weights")

    if "algo" in args:
        # Test knn mit verschiedenen "algorithm" Werten
        knn_scores = knn(algorithm=['auto', 'ball_tree', 'kd_tree', 'brute'], mean_from=mean_from)
        plot_accuracy_scores(knn_scores, xlabel="algorithm", ylabel="Genauigkeit",
                            title="Genauigkeit des KNN mit verschiedenen algorithm-Werten",
                            savefig=True, filename="knn_algorithm")

    if "leaf" in args:
        # Test knn mit verschiedenen "leaf_size" Werten
        knn_scores = knn(leaf_size=[1, 2, 5, 7, 10, 15, 20, 30, 40, 50, 100], mean_from=mean_from)
        plot_accuracy_scores(knn_scores, xlabel="leaf_size", ylabel="Genauigkeit",
                            title="Genauigkeit des KNN mit verschiedenen leaf_size-Werten",
                            savefig=True, filename="knn_leaf_size")

    if "p" in args:
        # Test knn mit verschiedenen "p" Werten
        knn_scores = knn(p=[1, 2], mean_from=mean_from)
        plot_accuracy_scores(knn_scores, xlabel="p", ylabel="Genauigkeit",
                            title="Genauigkeit des KNN mit verschiedenen p-Werten",
                            savefig=True, filename="knn_p")

def test_nn(*args, mean_from=10):
    if "layouts" in args:
        # Test neural network mit verschiedenen "hidden_layers" Werten
        neural_network_scores = nn(hidden_layers=[(511, ), (511, 200), (511, 200, 100), (511, 200, 100, 50), (511, 256), (511, 256, 127), (511, 256, 127, 64), (511, 256, 128), (511, 256, 128, 64)], mean_from=mean_from)
        plot_accuracy_scores(neural_network_scores, xlabel="Hidden Layers", ylabel="Genauigkeit",
                            title="Genauigkeit des Neural Network mit verschiedenen Hidden Layers",
                            savefig=True, filename="nn_layouts")

    if "layouts_512" in args:
        # Test neural network mit verschiedenen "hidden_layers" Werten
        neural_network_scores = nn(hidden_layers=[(512, ), (512, 200), (512, 200, 100), (512, 200, 100, 50), (512, 256), (512, 256, 127), (512, 256, 127, 64), (512, 256, 128), (512, 256, 128, 64)], mean_from=mean_from)
        plot_accuracy_scores(neural_network_scores, xlabel="Hidden Layers", ylabel="Genauigkeit",
                            title="Genauigkeit des Neural Network mit verschiedenen Hidden Layers",
                            savefig=True, filename="nn_layouts")

    if "neurons" in args:
        # Test neural network mit verschiedenen "hidden_layers" Werten
        neural_network_scores = nn(hidden_layers=[(10,), (25, ), (50, ), (75, ), (100, ), (150,), (200, ), (500, ), (512, ), (1000, )], mean_from=mean_from)
        plot_accuracy_scores(neural_network_scores, xlabel="Hidden Layers", ylabel="Genauigkeit",
                            title="Genauigkeit des Neural Network mit verschiedenen Hidden Layers",
                            savefig=True, filename="nn_neurons")
        
    if "neurons512_" in args:
        # Test neural network mit verschiedenen "hidden_layers" Werten 
        neural_network_scores = nn(hidden_layers=[(500,), (505,), (510,), (511,), (512, ), (513,), (515,), (520,), (550, )], mean_from=mean_from)
        plot_accuracy_scores(neural_network_scores, xlabel="Hidden Layers", ylabel="Genauigkeit",
                            title="Genauigkeit des Neural Network mit verschiedenen Hidden Layers",
                            savefig=True, filename="nn_neurons")

    if "layers" in args:
        # Test neural network mit verschiedenen "hidden_layers" Werten
        neural_network_scores = nn(hidden_layers=[(100,), (100, 100), (100, 100, 100), (100, 100, 100, 100), (100, 100, 100, 100, 100), (100, 100, 100, 100, 100, 100), (100, 100, 100, 100, 100, 100, 100)], mean_from=mean_from)
        plot_accuracy_scores(neural_network_scores, xlabel="Hidden Layers", ylabel="Genauigkeit",
                            title="Genauigkeit des Neural Network mit verschiedenen Hidden Layers",
                            savefig=True, filename="nn_layers")

    if "activation" in args:
        # Test neural network mit verschiedenen "activation" Werten
        neural_network_scores = nn(activation=['logistic', 'identity', 'relu', 'tanh'], mean_from=mean_from)
        plot_accuracy_scores(neural_network_scores, xlabel="Activation", ylabel="Genauigkeit",
                            title="Genauigkeit des Neural Network mit verschiedenen Activation-Werten",
                            savefig=True, filename="nn_activation")

    if "solver" in args:
        # Test neural network mit verschiedenen "solver" Werten
        neural_network_scores = nn(solver=['adam', 'sgd'], mean_from=mean_from)
        plot_accuracy_scores(neural_network_scores, xlabel="Solver", ylabel="Genauigkeit",
                            title="Genauigkeit des Neural Network mit verschiedenen Solver-Werten",
                            savefig=True, filename="nn_solver")

    if "alpha" in args:
        # Test neural network mit verschiedenen "alpha" Werten
        neural_network_scores = nn(alpha=[0.0001, 0.0005, 0.001, 0.025, 0.05, 0.075, 0.01, 0.1, 1], mean_from=mean_from)
        plot_accuracy_scores(neural_network_scores, xlabel="Alpha", ylabel="Genauigkeit",
                            title="Genauigkeit des Neural Network mit verschiedenen Alpha-Werten",
                            savefig=True, filename="nn_alpha)")
        
    if "learn" in args:
        # Test neural network mit verschiedenen "learning_rate_init" Werten
        neural_network_scores = nn(learning_rate_init=[0.0001, 0.001, 0.01, 0.1, 1, 10, 100], mean_from=mean_from)
        plot_accuracy_scores(neural_network_scores, xlabel="Learning Rate", ylabel="Genauigkeit",
                            title="Genauigkeit des Neural Network mit verschiedenen Learning Rate-Werten",
                            savefig=True, filename="nn_learning_rate")    
        

def compare_all():
    # Vergleich aller Modelle

    l_reg_standart = acc_score_ez(LinearRegression(fit_intercept=True, copy_X=False, n_jobs=-1))
    print(f"l_reg_standart   - 1/8 - Genauigkeit: {l_reg_standart:.4f}")
    svm_standart = cv_score_ez(SVC())
    print(f"svm_standart     - 2/8 - Genauigkeit: {svm_standart:.4f}")
    knn_standart = cv_score_ez(KNeighborsClassifier(n_jobs=-1))
    print(f"knn_standart     - 3/8 - Genauigkeit: {knn_standart:.4f}")
    nn_standart = cv_score_ez(MLPClassifier())
    print(f"nn_standart      - 4/8 - Genauigkeit: {nn_standart:.4f}")

    l_reg_optimal = acc_score_ez(LinearRegression(fit_intercept=True, copy_X=True, n_jobs=-1))
    print(f"l_reg_optimal    - 5/8 - Genauigkeit: {l_reg_optimal:.4f}")
    svm_optimal = cv_score_ez(SVC(C=50, kernel='rbf', gamma=0.001))
    print(f"svm_optimal      - 6/8 - Genauigkeit: {svm_optimal:.4f}")
    knn_optimal = cv_score_ez(KNeighborsClassifier(n_neighbors=3, weights='uniform', algorithm='auto', leaf_size=10, p=2, n_jobs=-1))
    print(f"knn_optimal      - 7/8 - Genauigkeit: {knn_optimal:.4f}")
    nn_optimal = cv_score_ez(MLPClassifier(hidden_layer_sizes=(512, ), activation='logistic', solver='adam', alpha=0.001, max_iter=5000, learning_rate_init=0.0001), mean_from=5)
    print(f"nn_optimal       - 8/8 - Genauigkeit: {nn_optimal:.4f}")
    

    # Plot generieren
    plt.figure(figsize=(15, 10), layout='tight', dpi=100) 
    
    models = ["lineare Regression", "SVM", "KNN", "Neural Network"]
    accuracy_standart = [l_reg_standart, svm_standart, knn_standart, nn_standart]
    accuracy_optimal = [0, svm_optimal-svm_standart, knn_optimal-knn_standart, nn_optimal-nn_standart]

    plt.stackplot(models, accuracy_standart, accuracy_optimal, labels=["standart", "optimiert"], colors=distinct_colors)
    
    # Matplotlib Einstellungen
    plt.xticks(rotation=45)
    plt.ylim(0.4, 1.0)
    plt.yticks(np.arange(0.4, 1.0, 0.05))
    plt.xlabel("Modelle")
    plt.ylabel("Genauigkeit")
    plt.legend(["standart", "optimiert"])
    plt.title("ML Modelle im Vergleich")
    plt.grid()
    
    # abspeichern
    file_path = os.path.join(os.path.expanduser("~"), "Desktop", "Vergleich_Modelle.png")
    plt.savefig(file_path)



#test_linear_regression("fit", "copy")

#test_nn("layouts", "neurons", "layers", "activation", "solver", "alpha")
#test_nn("learn")
#test_nn("layouts")
#test_nn("layouts_512")
#test_nn("alpha")
#test_nn("neurons")
#test_nn("neurons_512")
#test_nn("layers")

#test_svm("c", "kernel", "gamma")
#test_svm("c")
#test_svm("gamma")

#test_knn("p")
#test_knn("n", "weights", "algo", "leaf", "p")


compare_all()
