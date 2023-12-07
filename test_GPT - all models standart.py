import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LinearRegression  # Import für Lineare Regression
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
