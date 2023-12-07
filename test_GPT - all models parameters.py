import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Daten einlesen
data_path = 'data/Sample-Database V1 (Sphere).csv'  # Setzen Sie hier Ihren Dateipfad
data = pd.read_csv(data_path, sep=';')

# Daten vorbereiten
X = data.drop(columns=["#", "Material"]).values # Features
y = data['Material'].values                     # Zielvariable

# Daten aufteilen
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Daten skalieren
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

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
