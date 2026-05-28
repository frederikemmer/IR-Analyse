# >>> Imports [Start] <<< ----------------------------------------------------------------------------------------------
import PySimpleGUI as sg
from pathlib import Path
import shutil

# >>> Visualisierung ---------------------------------------------------------------------------------------------------
from matplotlib import pyplot as plt
from matplotlib import use as mpl_use
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# >>> SciKit learn -----------------------------------------------------------------------------------------------------
from sklearn.pipeline import Pipeline                   #
from sklearn.preprocessing import StandardScaler        #   Preprocessing
from sklearn.linear_model import LogisticRegression     #
from sklearn.neighbors import KNeighborsClassifier      #
# from sklearn.svm import SVR                           #
from sklearn.neural_network import MLPClassifier        #   ML_Modelle
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, balanced_accuracy_score

# >>> Datenverarbeitung, Hilfsmittel, etc. --------------------------------------------------------------------------
import pandas as pd
import numpy as np
import joblib
import random
import threading
import time
import os
import json
import paho.mqtt.client as mqtt

try:
    import seabreeze.spectrometers as sb
except Exception:
    sb = None

# Für Tests ohne GUI-Loop kann ein headless Backend erzwungen werden.
mpl_use("Agg" if os.environ.get("IR_ANALYSE_TEST_MODE") == "1" else "TKAgg")
# Imports [Ende] -------------------------------------------------------------------------------------------------------
#
#
# >>> Notizen
# Übersicht zu Funktionen von PySimpleGUI: https://pysimplegui.trinket.io/demo-programs
# Übersicht zu Funktionen von Matplotlib: https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.html
#
#
# ----------------------------------------------------------------------------------------------------------------------
# >>> Globale Variablen [START] <<< ------------------------------------------------------------------------------------
# Auflösung: 899,119 - 1712,067nm >>> in 513 Schritten (gegeben durch Spektroskop) - 1 Nachkommastelle
Auflösung = [899.119, 1712.067, 513, 1]

# generieren der Werte für die X-Achse (Wellenlängen & Wellenzahlen)
def get_parameters(called_parameter):
    output = []
    if called_parameter == "wavelengths":
        i = 0
        while i < Auflösung[2]:
            output.append(round(Auflösung[0] + i * ((Auflösung[1] - Auflösung[0]) / Auflösung[2]), Auflösung[3]))
            i += 1
        return output
    elif called_parameter == "wavenumbers":
        x = 10 ** 7  # Umrechungsfaktor Wellenlänge → Wellenzahl
        i = Auflösung[2]  # umgekehrt Auffüllen da Wellenzahl von "unten nach oben geht"
        while i > 0:
            output.append(round(x / (Auflösung[0] + i * ((Auflösung[1] - Auflösung[0]) / Auflösung[2])), Auflösung[3]))
            i -= 1
        return output

# >>> Variablen für Daten-Pfade
compare_path = "data/Vergleichs_Spektren.txt"           # Pfad zu Vergleichs-Spektren
realtime_path = "data/RealtimeData_Relative__0__1.txt"  # Pfad zu Realtime-Daten
database_path = "data/Sample-Database V1 (Sphere).csv"  # Pfad zu Datenbank von "Sphere" (für ML)

# >>> Graphen erstellen & Dimensionierung ("fig" mit Subplots verwendet, falls zusätzliche Graphen dazukommen)
fig = plt.figure(figsize=(20, 10))                      # Werte über (18,8) füllen das Fenster passend
# rect=[left, bottom, width, height] - Ausgangspunkt unten, Links (Verhältnis zu Canvas des Layouts) & Höhe, Breite
live = fig.add_axes(rect=[0.05, 0.1, 0.93, 0.89])       # Anordnung des Live-Graphen

# >>> Optische Einstellungen
# auswählbare Themen                                    # Standard -> 1. Eintrag (hell) / 2. Eintrag (dunkel)
selectable_themes = ["Reddit", "DarkBlue", "Black", "Topanga"]
# auswählbare Schriftarten                              # Standard -> 1. Eintrag
selectable_fonts = ["Arial", "Helvetica", "Courier", "Times New Roman", "Verdana", "Calibri", "Comic Sans MS"]
# auswählbare Schriftgrößen                             # Standard -> 3. Eintrag
selectable_font_sizes = [10, 11, 12, 13, 14]
live_color = "red"                                      # Farbe für Live-Graph
compare_color = "blue"                                  # Farbe für Vergleichs-Graph
hold_color = "green"                                    # Farbe für HOLD-Graph

# >>> nicht zu ändern / Variablen für Programm "under the hood" etc. (abschaffen wenn möglich - evtl. "Settings-Datei")
Materialien = ["PS", "PP", "PET", "PE", "HDPE", "LDPE", "PVC", "ABS", "PA", "PA6", "Acrylglas", "None"]
mat_col = {
    "PS": "#80ff00",            # "lime"
    "PP": "#0000ff",            # "blue"
    "PET": "#00ff00",           # "green"
    "PE": "#ffff00",            # "yellow"
    "HDPE": "#ffff00",          # "yellow"
    "LDPE": "#ffff00",          # "yellow"
    "PVC": "#ff8000",           # "orange"
    "ABS": "#8000ff",           # "purple"
    "PA": "#00ffff",            # "cyan"
    "PA6": "#ff00ff",           # "magenta"
    "Acrylglas": "#000000",     # "black"
    "None": "808080"            # "grey"
}                                         # Farben für Materialien
mat_num = {
    "1": "PS",
    "2": "PET",
    "3": "PP",
    "4": "HDPE",                                        # exisitiert im Trainingssatz nicht mehr -> Zusammenführung mit PE
    "5": "PE",
    "6": "PVC",
    "7": "Acrylglas",
    "8": "PA6",
}                                         # Nummern für Materialien
Wellenlängen = get_parameters("wavelengths")            # Beschriftung für x-Achse (evtl. nur bei Änderung ...)
Wellenzahlen = get_parameters("wavenumbers")            # Beschriftung für x-Achse (evtl. nur bei Änderung ...)
x_axe_label = "Wellenlänge"                             # Variable für aktuelle x-Achsen-Beschriftung
hold = False                                            # Variable für HOLD-Button
simulate_on = False                                     # Variable für Echtzeit-Simulation
ml_output_on = False                                    # Variable für ML-Ausgabe
relative_view = False                                   # Variable für relative Graphen
debug = False                                           # Variable für Debug-Modus
sim_speed = 0.3                                         # Variable für Simulations-Geschwindigkeit
hold_spectrum = []                                      # Variable für HOLD-Spektrum


def create_mqtt_client():
    # paho-mqtt >=2: explizit neue Callback-API nutzen (entfernt DeprecationWarning)
    try:
        return mqtt.Client(callback_api_version=mqtt.CallbackAPIVersion.VERSION2)
    except Exception:
        # Fallback für ältere paho-Versionen
        return mqtt.Client()


client = create_mqtt_client()                           # Initialisierung MQTT-Client
realtime_source_mode = "spectrometer"                   # "spectrometer" (Standard) | "file" (alt)
last_live_values = []                                   # zuletzt empfangenes Spektrum (512 Werte)
dark_spectrum = None                                    # aufgenommenes Dunkel-Spektrum (Rohwerte)
source_spectrum = None                                  # aufgenommenes Quell-Spektrum (Rohwerte)
dark_spectrum_path = "data/dark_spectrum.json"          # Pfad zum Dunkel-Spektrum
source_spectrum_path = "data/source_spectrum.json"      # Pfad zum Quell-Spektrum
display_mode = "raw"                                    # "raw" = Rohwerte | "percent" = kalibriert in %
normalize_view = False                                  # True = jedes Spektrum auf [0,100] strecken
main_window = None
figure_canvas_agg = None
values = {"-VERGLEICH_LIST-": []}

# Globale Variablen [ENDE] ---------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
#
#
#
#
# ----------------------------------------------------------------------------------------------------------------------
# >>> Funktionen [START] <<< -------------------------------------------------------------------------------------------

# >>> multithreaded Funktionen
class mt:
    def ml_output_loop(self=None):
        while ml_output_on:
            mt.ml_output_single()
            time.sleep(1)

    def ml_output_single(self=None):
        prediction_input = ML.prepare_prediction_input(hold_spectrum[1:])

        Vorhersage_neural_key = ML.prediction_to_key(ML.load_model_pred(prediction_input, "models/neural_5000.model"))
        Vorhersage_neural = mat_num.get(Vorhersage_neural_key, "None")

        Vorhersage_knn_key = ML.prediction_to_key(ML.load_model_pred(prediction_input, "models/knn_10.model"))
        Vorhersage_knn = mat_num.get(Vorhersage_knn_key, "None")

        Vorhersage_l_reg_key = ML.prediction_to_key(ML.load_model_pred(prediction_input, "models/l_reg.model"))
        Vorhersage_l_reg = mat_num.get(Vorhersage_l_reg_key, "None")

        main_window["-KI_OUTPUT_LIST-"].update([["Neurales Netz: ", Vorhersage_neural],
                                                ["KNN: ", Vorhersage_knn],
                                                ["Logistische Regression: ", Vorhersage_l_reg]
                                                ])

    # Echtzeitdaten und Vergleichs_Spektren.txt anzeigen, wenn ausgewählt
    # noinspection PyTypeChecker
    def Realtime_Loop(self=None):
        while True:
            global last_live_values

            # Daten aus gewählter Quelle holen
            try:
                if realtime_source_mode == "spectrometer":
                    spectrum = spectrometer_connection.read_live_spectrum()
                    if spectrum is not None and len(spectrum) > 0:
                        last_live_values = list(spectrum)
                else:
                    data_input = io.import_data(realtime_path)
                    if data_input and len(data_input) > 0:
                        last_live_values = list(data_input[-1][2:])
            except Exception:
                if debug:
                    print("Realtime_Loop: Fehler beim Importieren der Echtzeitdaten")

            # Nur im Legacy-Dateimodus trimmen
            if realtime_source_mode == "file":
                try:
                    io.trim_data(path=realtime_path)
                except Exception:
                    if debug:
                        print("Realtime_Loop: Fehler bei trim_data()")

            time.sleep(0.05)

    # Simuliert Echtzeitdaten
    def simulate(self=None):
        while simulate_on:
            path = "data/Realtime-Sample-Run-ORIGINAL.txt"
            lines = Path(path).read_text(encoding="utf-8").splitlines()
            # dataset passend dimensionieren (sonst müsste mit "append" gearbeitet werden)
            dataset = [[0] * len(lines[0].split("\t")) for _ in range(len(lines))]
            # Variablen für Schleifen
            i1 = 0
            i2 = 0
            # Zeilen/Spaltenweise dataset befüllen
            while i1 < len(lines):
                while i2 < len(lines[i1].split("\t")):
                    try:
                        dataset[i1][i2] = float(lines[i1].split("\t")[i2].replace(",", "."))
                    except:
                        dataset[i1][i2] = lines[i1].split("\t")[i2].replace(",", ".")
                    i2 += 1
                i2 = 0
                i1 += 1

            # dataset in einem String zusammenfassen
            line_to_write = str(dataset[random.randint(0, len(dataset) - 1)]). \
                replace(",", "\t").replace(" ", "").replace("'", "").replace("[", "").replace("]", "")
            # String in "Echtzeitdaten-Datei" schreiben
            Path(realtime_path).write_text(line_to_write + "\n", encoding="utf-8")
            time.sleep(sim_speed)

# Ki Berechnungen, etc.
class ML:
    model_paths = {
        "neural": "models/neural_5000.model",
        "knn": "models/knn_10.model",
        "l_reg": "models/l_reg.model",
    }
    archive_dir = Path("models/archiv")
    report_path = Path("models/training_report.json")
    _model_cache = {}

    def prediction_to_key(prediction):
        arr = np.asarray(prediction).reshape(-1)
        if arr.size == 0:
            return ""

        value = arr[0]
        try:
            return str(int(round(float(value))))
        except Exception:
            return str(value).replace("[", "").replace("]", "").strip()

    def prepare_prediction_input(spectrum_data, source_mode=None, dark=None, source=None):
        values = list(spectrum_data)
        mode = source_mode if source_mode is not None else realtime_source_mode

        if mode == "spectrometer":
            active_dark = dark if dark is not None else dark_spectrum
            active_source = source if source is not None else source_spectrum

            if active_dark is not None and active_source is not None:
                return compute.apply_calibration(values, dark=active_dark, source=active_source)

        return values

    def load_training_data(database_path="data/Sample-Database V1 (Sphere).csv"):
        data = pd.read_csv(database_path, sep=";")
        X = data.drop(columns=["#", "Material"]).astype(float).values
        y = data["Material"].astype(int).values
        return X, y

    def prep_data(database_path="data/Sample-Database V1 (Sphere).csv", test_size=0.2, random_state=random.randint(1, 1000000)):
        X, y = ML.load_training_data(database_path)

        # Stratified Split hält die Klassenverteilung stabil und verhindert Verzerrung seltener Materialien.
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=y
        )
        return X_train, X_test, y_train.reshape(-1, 1), y_test.reshape(-1, 1)

    def _cv_splitter(y, random_state=42):
        _, counts = np.unique(y, return_counts=True)
        min_count = int(counts.min()) if len(counts) > 0 else 2
        n_splits = max(2, min(5, min_count))
        return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    def _make_json_safe(value):
        if isinstance(value, dict):
            return {str(k): ML._make_json_safe(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [ML._make_json_safe(v) for v in value]
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, Path):
            return str(value)
        return value

    def _backup_model(model_path, version, archive_dir=None):
        src = Path(model_path)
        if not src.exists():
            return None

        archive_root = Path(archive_dir) if archive_dir is not None else ML.archive_dir
        archive_root.mkdir(parents=True, exist_ok=True)

        backup_name = f"{src.stem}_v{version}{src.suffix}"
        same_dir_path = src.with_name(backup_name)
        archive_path = archive_root / backup_name

        shutil.copy2(src, same_dir_path)
        shutil.copy2(src, archive_path)

        return {
            "source": str(src),
            "same_dir": str(same_dir_path),
            "archive": str(archive_path),
        }

    def backup_all_models(version=None, archive_dir=None):
        if version is None:
            version = time.strftime("%Y%m%d_%H%M%S")

        backups = []
        for model_path in ML.model_paths.values():
            backup_info = ML._backup_model(model_path, version, archive_dir=archive_dir)
            if backup_info is not None:
                backups.append(backup_info)

        return version, backups

    def _search_config(model_key, iterrations=5000, randome_state=1, neighbors=10):
        if model_key == "neural":
            pipeline = Pipeline([
                ("scale", StandardScaler()),
                ("model", MLPClassifier(solver="lbfgs", max_iter=iterrations, random_state=randome_state))
            ])
            param_grid = {
                "model__hidden_layer_sizes": [(100,)],
                "model__activation": ["relu"],
                "model__alpha": [0.001],
            }
            save_path = ML.model_paths["neural"]
            label = "Neurales Netz"
        elif model_key == "knn":
            neighbors_to_try = []
            for value in [neighbors - 3, neighbors - 1, neighbors + 1, neighbors + 3, 5, 7, 9, 11]:
                if value < 1:
                    continue
                odd_value = value if value % 2 == 1 else value + 1
                if odd_value not in neighbors_to_try:
                    neighbors_to_try.append(odd_value)

            pipeline = Pipeline([
                ("scale", StandardScaler()),
                ("model", KNeighborsClassifier())
            ])
            param_grid = {
                "model__n_neighbors": [value for value in neighbors_to_try if value in [7, 9]] or [7],
                "model__weights": ["distance"],
                "model__p": [2, 1],
            }
            save_path = ML.model_paths["knn"]
            label = "KNN"
        elif model_key == "l_reg":
            pipeline = Pipeline([
                ("scale", StandardScaler()),
                ("model", LogisticRegression(max_iter=5000, solver="lbfgs", random_state=randome_state))
            ])
            param_grid = {
                "model__C": [1.0, 2.0, 5.0],
                "model__class_weight": ["balanced"],
            }
            save_path = ML.model_paths["l_reg"]
            label = "Logistische Regression"
        else:
            raise ValueError(f"Unbekannter Modelltyp: {model_key}")

        return pipeline, param_grid, save_path, label

    def _train_best_model(model_key, database_path="data/Sample-Database V1 (Sphere).csv", iterrations=5000,
                          randome_state=42, neighbors=10, validation_size=0.2):
        X, y = ML.load_training_data(database_path)

        X_train, X_valid, y_train, y_valid = train_test_split(
            X,
            y,
            test_size=validation_size,
            random_state=randome_state,
            stratify=y
        )

        pipeline, param_grid, save_path, label = ML._search_config(
            model_key,
            iterrations=iterrations,
            randome_state=randome_state,
            neighbors=neighbors
        )

        fixed_candidate = all(len(values) == 1 for values in param_grid.values())
        if fixed_candidate:
            best_params = {key: values[0] for key, values in param_grid.items()}
            final_model = pipeline.set_params(**best_params)
            final_model.fit(X_train, y_train)
            validation_pred = final_model.predict(X_valid)
            validation_accuracy = accuracy_score(y_valid, validation_pred)
            validation_balanced_accuracy = balanced_accuracy_score(y_valid, validation_pred)
            search_candidates = 1
            cv_balanced_accuracy = float(validation_balanced_accuracy)
            cv_accuracy = float(validation_accuracy)
        else:
            search = GridSearchCV(
                estimator=pipeline,
                param_grid=param_grid,
                scoring={
                    "balanced_accuracy": "balanced_accuracy",
                    "accuracy": "accuracy",
                },
                refit="balanced_accuracy",
                cv=ML._cv_splitter(y_train, random_state=randome_state),
                n_jobs=-1,
                error_score="raise"
            )
            search.fit(X_train, y_train)

            final_model = search.best_estimator_
            best_params = search.best_params_
            search_candidates = len(search.cv_results_["params"])
            cv_balanced_accuracy = float(search.best_score_)
            cv_accuracy = float(search.cv_results_["mean_test_accuracy"][search.best_index_])

            validation_pred = final_model.predict(X_valid)
            validation_accuracy = accuracy_score(y_valid, validation_pred)
            validation_balanced_accuracy = balanced_accuracy_score(y_valid, validation_pred)

        final_model.fit(X, y)
        joblib.dump(final_model, save_path)

        cache_key = str(Path(save_path).resolve())
        ML._model_cache.pop(cache_key, None)

        class_values, class_counts = np.unique(y, return_counts=True)
        report = {
            "label": label,
            "path": save_path,
            "search_candidates": search_candidates,
            "best_params": best_params,
            "cv_balanced_accuracy": cv_balanced_accuracy,
            "cv_accuracy": cv_accuracy,
            "validation_accuracy": float(validation_accuracy),
            "validation_balanced_accuracy": float(validation_balanced_accuracy),
            "samples": int(len(y)),
            "features": int(X.shape[1]),
            "class_distribution": {
                str(int(cls)): int(count)
                for cls, count in zip(class_values, class_counts)
            },
        }

        return report

    def _write_training_report(report, version, archive_dir=None):
        archive_root = Path(archive_dir) if archive_dir is not None else ML.archive_dir
        archive_root.mkdir(parents=True, exist_ok=True)

        report_json = json.dumps(ML._make_json_safe(report), ensure_ascii=False, indent=2)
        ML.report_path.write_text(report_json, encoding="utf-8")
        (archive_root / f"training_report_v{version}.json").write_text(report_json, encoding="utf-8")

    def retrain_all_models(sample_spectrum=None, backup=True, version=None, archive_dir=None,
                           database_path="data/Sample-Database V1 (Sphere).csv", iterrations=5000,
                           randome_state=42, neighbors=10):
        if version is None:
            version = time.strftime("%Y%m%d_%H%M%S")

        if sample_spectrum is None:
            X, _ = ML.load_training_data(database_path)
            sample_spectrum = X[0].tolist()

        backups = []
        if backup:
            _, backups = ML.backup_all_models(version=version, archive_dir=archive_dir)

        training_reports = {}
        for model_key in ["neural", "knn", "l_reg"]:
            print(f"Trainiere {model_key}...")
            training_reports[model_key] = ML._train_best_model(
                model_key,
                database_path=database_path,
                iterrations=iterrations,
                randome_state=randome_state,
                neighbors=neighbors,
            )
            print(
                f"Fertig: {model_key} | CV bal={training_reports[model_key]['cv_balanced_accuracy']:.4f} | "
                f"Valid bal={training_reports[model_key]['validation_balanced_accuracy']:.4f}"
            )

        for model_key, model_path in ML.model_paths.items():
            training_reports[model_key]["sample_prediction"] = ML.prediction_to_key(
                ML.load_model_pred(sample_spectrum, model_path)
            )

        report = {
            "version": version,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "database_path": database_path,
            "archive_dir": str(Path(archive_dir) if archive_dir is not None else ML.archive_dir),
            "backups": backups,
            "models": training_reports,
        }
        ML._write_training_report(report, version, archive_dir=archive_dir)
        return report

    # Statistische Werte ausgeben
    def get_stat_value(predicted, actual, digits, Ausgabe=False):
        Varianz = round(np.var(predicted - actual), digits)
        Durchschnitt = round(np.mean(predicted - actual), digits)
        Standartabweichung = round(np.std(predicted - actual), digits)

        if Ausgabe:
            print("Varianz: " + str(Varianz))
            print("Durschnitt: " + str(Durchschnitt))
            print("Standartabweichung: " + str(Standartabweichung) + "\n")

        return Varianz, Durchschnitt, Standartabweichung

    # Neurales Netzwerk trainieren und Prediction ausgeben
    def neural_network(spectrum_data, iterrations=5000, randome_state=1):
        ML._train_best_model("neural", iterrations=iterrations, randome_state=randome_state)
        return ML.load_model_pred(spectrum_data, ML.model_paths["neural"])

    # KNN trainieren und Prediction ausgeben
    def knn(spectrum_data, neighbors=10):
        ML._train_best_model("knn", neighbors=neighbors)
        return ML.load_model_pred(spectrum_data, ML.model_paths["knn"])

    # Logistische Regression trainieren und Prediction ausgeben
    def l_reg(spectrum_data):
        ML._train_best_model("l_reg")
        return ML.load_model_pred(spectrum_data, ML.model_paths["l_reg"])

    def load_model(path="models/neural_5000.model"):
        path_obj = Path(path)
        cache_key = str(path_obj.resolve())
        cache_mtime = path_obj.stat().st_mtime_ns
        cached = ML._model_cache.get(cache_key)

        if cached is not None and cached["mtime"] == cache_mtime:
            return cached["model"]

        loaded_model = joblib.load(path_obj)
        ML._model_cache[cache_key] = {
            "mtime": cache_mtime,
            "model": loaded_model,
        }
        return loaded_model

    # Model laden und Prediction ausgeben
    def load_model_pred(spectrum_data, path = "models/neural_5000.model"):
        loaded_model = ML.load_model(path)
        spectrum_data = np.array(spectrum_data).reshape(1, -1)
        prediction = loaded_model.predict(spectrum_data)

        return prediction

# allgemeine Berechnungen
class compute:
    # relative Werte aus absoluten Werten berechnen
    def relative_values(values=None, begin=0, end=100):
        if values is None:
            values = []
        minValue = min(values)
        maxValue = max(values)
        factor = (end - begin) / (maxValue - minValue)

        for i in range(len(values)):
            values[i] = (values[i] - minValue) * factor + begin

        return values

    # Kalibrierte Prozentwerte: (raw - dark) / (source - dark) * 100
    def apply_calibration(raw_values, dark=None, source=None):
        d = dark if dark is not None else dark_spectrum
        s = source if source is not None else source_spectrum
        if d is None or s is None:
            return list(raw_values)
        n = min(len(raw_values), len(d), len(s))
        return [
            ((raw_values[i] - d[i]) / (s[i] - d[i]) * 100.0) if s[i] != d[i] else 0.0
            for i in range(n)
        ]

    # Einzelnes Spektrum auf [0, 100] strecken (unabhängige Normalisierung)
    def normalize_single(values):
        if not values:
            return list(values)
        mn, mx = min(values), max(values)
        if mx == mn:
            return [50.0] * len(values)
        return [(v - mn) / (mx - mn) * 100.0 for v in values]

# Interaktion mit Dateien
class io:
    # >>> Interaktion mit Echtzeitdaten
    # Daten aus Realtime-Datei lesen
    def import_data(path=realtime_path, split_char="\t"):
        try:
            lines = Path(path).read_text(encoding="utf-8").splitlines()
            # dataset passend zum Input Dimensionieren (vorsichtshalber "0" → kein Crash aber erkennbar)
            dataset = [[0] * len(lines[0].split(split_char)) for _ in range(len(lines))]

            # vereinfacht mit Index statt "i in lines" um direkt in [][] verwenden zu können
            i1 = i2 = 0
            while i1 < len(lines):
                while i2 < len(lines[i1].split(split_char)):
                    try:
                        dataset[i1][i2] = float(lines[i1].split(split_char)[i2].replace(",", "."))
                    except:
                        dataset[i1][i2] = lines[i1].split(split_char)[i2]
                    i2 += 1
                i2 = 0
                i1 += 1

            return dataset

        except:
            if debug:
                print("io.input_data: Fehler")

    # Trimmt die Realtime-Datei auf die letzte Zeile
    def trim_data(path=realtime_path):
        try:
            lines = Path(path).read_text(encoding="utf-8").splitlines()
            lines_trimmed = lines[len(lines) - 1] + "\n"
            Path(path).write_text(lines_trimmed, encoding="utf-8")
            time.sleep(0.1)
        except:
            if debug:
                print("io.trim_data: Trimmen übersprungen")

    # >>> Interaktion mit Vergleichs-Spektren
    # Anzeige-Namen aus compare-spectrum.txt lesen
    def get_compare(self=None):
        output = []
        lines = Path(compare_path).read_text(encoding="utf-8").splitlines()
        for i in lines[1:]:
            output.append(lines[lines.index(i)].split(';')[0])
        return output

    # Spektrum aus compare-spectrums.txt lesen
    def get_compare_spektrum(anzeige_name):
        lines = Path(compare_path).read_text(encoding="utf-8").splitlines()
        output = []
        for i1 in lines[1:]:
            if lines[lines.index(i1)].split(";")[0] == anzeige_name:
                for i2 in lines[lines.index(i1)].split(";")[3:]:
                    output.append(
                        float(lines[lines.index(i1)].split(";")[lines[lines.index(i1)].split(";").index(i2)]))
        return output

    # Farbe aus compare-spectrums.txt lesen
    def get_compare_color(anzeige_name):
        lines = Path(compare_path).read_text(encoding="utf-8").splitlines()
        output = []
        for i1 in lines[1:]:
            if lines[lines.index(i1)].split(";")[0] == anzeige_name:
                output = str(lines[lines.index(i1)].split(";")[2].replace(" ", ""))
        return output

    # Spektrum dem Vergleichs-Katalog HINZUFÜGEN
    def add_compare(spectrum_to_add):

        if values["-COMPARE_INPUT-"] in io.get_compare():
            sg.popup("Name bereits vergeben")
        else:
            try:
                lines = Path(compare_path).read_text(encoding="utf-8").splitlines()

                # Farbe auswählen (falls Color-Picker nicht verwendet → Aus Liste)
                try:
                    color = values["-COMPARE_COLOR-"]
                except:
                    color = mat_col[values["-COMPARE_COMBO-"]]

                # neue Zeilen erstellen
                new_lines = [values["-COMPARE_INPUT-"], values["-COMPARE_COMBO-"], color]
                # neue Zeile anhängen
                lines.append(new_lines + spectrum_to_add)
                complete_document = ""

                # lines bereinigen und in complete_document schreiben
                for i in lines:
                    line_to_write = (str(i).replace(",", ";")
                                     .replace("'", "")
                                     .replace("[", "")
                                     .replace("]", ""))
                    complete_document = complete_document + line_to_write + "\n"

                # neue Spektren in Datei schreiben
                Path(compare_path).write_text(complete_document, encoding="utf-8")

                # Anzeigen im Fenster aktualisieren
                hold = False
                main_window["-VERGLEICH_LIST-"].update(io.get_compare())
                main_window["-ADD_COMPARE-"].update(disabled=True)
                main_window["-ADD_COMPARE-"].set_tooltip('zuerst "HOLD" aktivieren')
                main_window["-HOLD-"].set_tooltip("friert Live-Vorschau ein")
                main_window["-HOLD-"].update(text="HOLD", button_color=sg.theme_button_color())
                main_window["-COMPARE_INPUT-"].update("")
            except:
                if debug:
                    print("compare_spectrum.add: Fehler beim Hinzufügen")

    # Spektrum aus Vergleichs-Katalog ENTFERNEN
    def remove_compare(self=None):
        try:
            lines = Path(compare_path).read_text(encoding="utf-8").splitlines()

            # Spektren einlesen und ausgewählte entfernen
            keep = io.get_compare()
            for i in values["-VERGLEICH_LIST-"]:
                keep.remove(i)

            # IDs der zu erhaltenden Spektren ermitteln und in keepIDs schreiben
            keepIDs = [None] * len(keep)
            for i1 in keep:
                keepIDs[keep.index(i1)] = io.get_compare().index(i1)

            # erste Zeile mit Header befüllen und mit +1 Versatz den Rest über "keepIDs"
            output = [None] * (len(keepIDs) + 1)  # +1 für den Header
            output[0] = lines[0]  # erste Zeile mit Header füllen
            for i2 in keepIDs:
                output[keepIDs.index(i2) + 1] = lines[int(i2) + 1]

            # ausgabe String mit \n in "eine" Zeile schreiben, da Array nicht ausgegeben werden kann
            new_compare = ""
            for i3 in output:
                new_compare = new_compare + str(i3) + "\n"

            # Datei mit neuen Spektren schreiben
            Path(compare_path).write_text(new_compare, encoding="utf-8")

            # Anzeigen im Fenster aktualisieren
            main_window["-VERGLEICH_LIST-"].update(io.get_compare())
            main_window["-ADD_COMPARE-"].update(disabled=True)
            main_window["-ADD_COMPARE-"].set_tooltip('zuerst "HOLD" aktivieren')
            main_window["-HOLD-"].set_tooltip("friert Live-Vorschau ein")
            main_window["-HOLD-"].update(text="HOLD", button_color=sg.theme_button_color())
        except:
            if debug:
                print("compare_spectrum.remove: Fehler beim Entfernen")

# Dunkel- und Quell-Spektren speichern / laden
class calibration_io:
    def _last_capture_text(path, empty_text="Nicht aufgenommen"):
        p = Path(path)
        if not p.exists():
            return empty_text
        ts = time.localtime(p.stat().st_mtime)
        return time.strftime("%d.%m.%Y %H:%M:%S", ts)

    def save_dark(spectrum):
        Path(dark_spectrum_path).write_text(json.dumps([round(v, 4) for v in spectrum]), encoding="utf-8")

    def save_source(spectrum):
        Path(source_spectrum_path).write_text(json.dumps([round(v, 4) for v in spectrum]), encoding="utf-8")

    def load_dark():
        global dark_spectrum
        if Path(dark_spectrum_path).exists():
            dark_spectrum = json.loads(Path(dark_spectrum_path).read_text(encoding="utf-8"))
        return dark_spectrum

    def load_source():
        global source_spectrum
        if Path(source_spectrum_path).exists():
            source_spectrum = json.loads(Path(source_spectrum_path).read_text(encoding="utf-8"))
        return source_spectrum

    def load_all():
        calibration_io.load_dark()
        calibration_io.load_source()

    def dark_status():
        return calibration_io._last_capture_text(dark_spectrum_path)

    def source_status():
        return calibration_io._last_capture_text(source_spectrum_path)

# aktualisieren der Graphen
class update_graphs:
    def axe_label(self=None):
        live.cla()
        match x_axe_label:
            case "Wellenlänge":
                x_values = Wellenlängen
                live.set_xlabel("Wellenlänge [nm]")
            case "Wellenzahl":
                x_values = Wellenzahlen
                live.set_xlabel("Wellenzahl [cm^-1]")

        live.grid()
        live.set_title("Live-View")
        if display_mode == "percent" and dark_spectrum is not None and source_spectrum is not None:
            ylabel = "Intensität [%]"
        else:
            ylabel = "Intensität (Rohwerte)"
        if normalize_view:
            ylabel += "  (norm.)"
        live.set_ylabel(ylabel)
        return x_values

    def live_absolute(y_values=None):
        if y_values is None:
            y_values = []
        x_values = update_graphs.axe_label()

        def _calib(vals):
            if display_mode == "percent" and dark_spectrum is not None and source_spectrum is not None:
                return compute.apply_calibration(list(vals))
            return list(vals)

        def _norm(vals):
            return compute.normalize_single(vals) if normalize_view else vals

        if not hold:
            live.plot(x_values[1:], _norm(_calib(y_values)), color=live_color)
        else:
            live.plot(x_values[1:], _norm(_calib(hold_spectrum[1:])), color=hold_color)

        for i in values["-VERGLEICH_LIST-"]:
            # [1:] da erster Spektrum-Wert immer 0 ist
            # Bereits gespeicherte Vergleichsspektren unverändert darstellen.
            live.plot(x_values[1:], _norm(io.get_compare_spektrum(i)[1:]), color=io.get_compare_color(i))

    def live_relative(y_values=None):
        if y_values is None:
            y_values = []
        x_values = update_graphs.axe_label()

        def _prep_live(vals):
            result = list(vals)
            if display_mode == "percent" and dark_spectrum is not None and source_spectrum is not None:
                result = compute.apply_calibration(result)
            result = compute.relative_values(values=list(result))
            return compute.normalize_single(result) if normalize_view else result

        def _prep_compare(vals):
            # Vergleichsspektren nicht erneut mit Dark/Source korrigieren.
            result = compute.relative_values(values=list(vals))
            return compute.normalize_single(result) if normalize_view else result

        if not hold:
            live.plot(x_values[1:], _prep_live(y_values), color=live_color)
        else:
            live.plot(x_values[1:], _prep_live(hold_spectrum[1:]), color=hold_color)

        for i in values["-VERGLEICH_LIST-"]:
            # [1:] da erster Spektrum-Wert immer 0 ist
            live.plot(x_values[1:], _prep_compare(io.get_compare_spektrum(i)[1:]), color=io.get_compare_color(i))

# Funktionen zur Verbindung & Kommunikation durch MQTT
class mqtt_connection:

    # function to connect to 192.168.178.147
    def start_client(self=None):
        try:
            client.on_connect = mqtt_connection.on_connected
            client.on_message = mqtt_connection.on_message
        except:
            if debug:
                print("Client konnte nicht erstellt werden")

        try:
            client.connect("192.168.178.147", 1883, 60)
        except:
            if debug:
                print("Verbindung zum Server fehlgeschlagen")

        client.loop_start()

    # Funktion wird bei Verbindung mit dem MQTT-Server ausgeführt
    def on_connected(client, userdata, flags, reason_code, properties=None):
        print("Mit MQTT-Server verbunden. Ergebnis: " + str(reason_code))
        client.subscribe("/measure")

    # Funktion wird bei Erhalten einer Nachricht ausgeführt
    def on_message(client, userdata, message):
        print(f"{message.topic}: {str(message.payload)}")


class spectrometer_connection:
    spectrometer = None
    connected = False
    status_text = "Nicht verbunden"
    last_error = ""
    lock = threading.RLock()

    def connect(self=None):
        with spectrometer_connection.lock:
            if sb is None:
                spectrometer_connection.connected = False
                spectrometer_connection.status_text = "Seabreeze nicht installiert"
                spectrometer_connection.last_error = "python-seabreeze fehlt"
                return False

            try:
                devices = sb.list_devices()
                if not devices:
                    spectrometer_connection.connected = False
                    spectrometer_connection.status_text = "Kein Spektrometer gefunden"
                    spectrometer_connection.last_error = ""
                    return False

                spectrometer_connection.spectrometer = sb.Spectrometer(devices[0])
                spectrometer_connection.connected = True
                spectrometer_connection.status_text = f"Verbunden: {spectrometer_connection.spectrometer.model}"
                spectrometer_connection.last_error = ""
                return True
            except Exception as e:
                spectrometer_connection.connected = False
                spectrometer_connection.status_text = "Verbindung fehlgeschlagen"
                spectrometer_connection.last_error = str(e)
                return False

    def disconnect(self=None):
        with spectrometer_connection.lock:
            try:
                if spectrometer_connection.spectrometer is not None:
                    spectrometer_connection.spectrometer.close()
            except Exception:
                pass
            spectrometer_connection.spectrometer = None
            spectrometer_connection.connected = False
            spectrometer_connection.status_text = "Nicht verbunden"

    def read_live_spectrum(self=None):
        with spectrometer_connection.lock:
            if not spectrometer_connection.connected or spectrometer_connection.spectrometer is None:
                return None

            try:
                values = spectrometer_connection.spectrometer.intensities(correct_nonlinearity=False)
                if values is None:
                    return None

                return [float(v) for v in values]
            except Exception as e:
                spectrometer_connection.connected = False
                spectrometer_connection.status_text = "Verbindung unterbrochen"
                spectrometer_connection.last_error = str(e)
                return None

    def get_status_line(self=None):
        if spectrometer_connection.last_error:
            return spectrometer_connection.status_text + "\n" + spectrometer_connection.last_error
        return spectrometer_connection.status_text

    def get_status_ui(self=None):
        if spectrometer_connection.connected:
            return {
                "label": "Verbunden",
                "color": "#2e7d32",
                "detail": spectrometer_connection.get_status_line(),
            }

        if spectrometer_connection.status_text in ("Kein Spektrometer gefunden", "Nicht verbunden"):
            return {
                "label": "Getrennt",
                "color": "#f9a825",
                "detail": spectrometer_connection.get_status_line(),
            }

        return {
            "label": "Fehler",
            "color": "#c62828",
            "detail": spectrometer_connection.get_status_line(),
        }

# Funktionen [ENDE] ----------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
#
#
#
#
# ----------------------------------------------------------------------------------------------------------------------
# >>> GUI [START] <<< --------------------------------------------------------------------------------------------------
def create_window(theme=selectable_themes[0], font=selectable_fonts[0], font_size=selectable_font_sizes[2]):
    sg.theme(theme)
    sg.set_options(font=(font, font_size))
    # Live-View-Tab
    live_view_tab = [
        [
            sg.pin(sg.Column([[
                sg.Frame("Spektrometer-Status", [
                    [sg.Text("Direkte Verbindung aktiv", text_color="green")],
                    [sg.Text("●", key="-SPEC_LED-", text_color="#f9a825", font=(font, max(font_size + 8, 14))),
                     sg.Text("Getrennt", key="-SPEC_STATE-", font=(font, font_size + 1))],
                    [sg.Multiline(spectrometer_connection.get_status_line(), key="-SPEC_STATUS-", size=(28, 1),
                                  disabled=True, no_scrollbar=True)],
                    [sg.Button("Neu verbinden", key="-SPEC_RECONNECT-", enable_events=True)]
                ], size=(280, 150))
            ]], key="-LIVE_SPEC_STATUS_COL-", visible=(realtime_source_mode == "spectrometer"))),
            sg.pin(sg.Column([[
                sg.Frame("Realtime-Einstellungen", [
                    [sg.Text("Realtime-Datei auswählen:")],
                    [sg.Button("..." + realtime_path[-40:] + "  •  ändern", key="-REALTIME_CHOOSE-", enable_events=True)],
                    [sg.Checkbox("Echtzeitdaten simulieren", key="-SIMULATION-", enable_events=True, default=simulate_on),
                     sg.Push(), sg.InputText("300", key="-SIM_DELAY-", size=(5, 1), enable_events=True),
                     sg.Text("ms"), sg.Push()],
                    [sg.Button(">1<", key="-TEST_BUTTON1-", enable_events=True),
                     sg.Button(">2<", key="-TEST_BUTTON2-", enable_events=True),
                     sg.Button(">3<", key="-TEST_BUTTON3-", enable_events=True)],
                ], size=(400, 200))
            ]], key="-LIVE_FILE_SETTINGS_COL-", visible=(realtime_source_mode == "file"))),
            sg.Push(),
            sg.Frame("ML-Analyse", [
                [sg.Text("Vorhersagen:")],
                [sg.Listbox({"Test", "test", "tesT"}, key="-KI_OUTPUT_LIST-",
                            expand_x=True, expand_y=True, no_scrollbar=True)],
                [sg.Checkbox("Echtzeit-Analyse", key="-REALTIME_ML-", enable_events=True, default=ml_output_on)],
                [sg.Button("Analyse", key="-ML_START-", enable_events=True, expand_x=True)],
            ], size=(200, 200), expand_y=True),
            sg.Push(),
            sg.Frame("Spektren-Vergleich", [
                [sg.Listbox(io.get_compare(),
                            key="-VERGLEICH_LIST-", select_mode="multiple", expand_x=True, expand_y=True,
                            enable_events=True)],
                # hinzufügen/entfernen von Vergleichs-Spektren
                [sg.Input(key="-COMPARE_INPUT-", enable_events=True, size=(30, 1), expand_x=True, pad=(5, 0)),
                 sg.Combo(Materialien, key="-COMPARE_COMBO-", enable_events=True, default_value="auswählen",
                          size=(20, 1), expand_x=True),
                 ],
                [sg.Button("HOLD", key="-HOLD-", enable_events=True, size=(10,), tooltip="friert Live-Vorschau ein"),
                 sg.Button("hinzufügen", disabled=True, disabled_button_color=("white", "grey"),
                           key="-ADD_COMPARE-", enable_events=True, expand_x=True, tooltip='zuerst "HOLD" aktivieren'),
                 sg.Button("entfernen", disabled=True, disabled_button_color=("white", "grey"),
                           key="-REMOVE_COMPARE-", enable_events=True, expand_x=True,
                           tooltip="zuerst Spektrum auswählen")],

            ], size=(400, 200)),
        ],
        [
            sg.Frame("Ansicht & Kalibrierung (Live-View)", [
                [
                    sg.Text("Referenz-Spektren:"),
                    sg.Button("Dark aufnehmen", key="-CAL_DARK-", enable_events=True, size=(14, 1)),
                    sg.Text(calibration_io.dark_status(), key="-CAL_DARK_STATUS-", size=(19, 1)),
                    sg.Button("Source aufnehmen", key="-CAL_SOURCE-", enable_events=True, size=(14, 1)),
                    sg.Text(calibration_io.source_status(), key="-CAL_SOURCE_STATUS-", size=(19, 1)),
                ],
                [
                    sg.Text("Anzeige:"),
                    sg.Radio("Rohwerte", "-DISPLAY_MODE-", default=(display_mode == "raw"),
                             key="-DISPLAY_RAW-", enable_events=True),
                    sg.Radio("Prozent (kalibriert)", "-DISPLAY_MODE-", default=(display_mode == "percent"),
                             key="-DISPLAY_PCT-", enable_events=True,
                             tooltip="Benötigt Dark- und Source-Spektrum"),
                    sg.VerticalSeparator(),
                    sg.Checkbox("relative Ansicht", key="-RELATIVE_VIEW-", enable_events=True, default=relative_view),
                    sg.Checkbox("Spektren normalisieren", key="-NORMALIZE_VIEW-", enable_events=True,
                                default=normalize_view,
                                tooltip="Streckt jedes Spektrum einzeln auf die volle vertikale Ausdehnung"),
                    sg.VerticalSeparator(),
                    sg.Text("Speichern:", text_color="gray40"),
                    sg.Text("", key="-SAVE_MODE_HINT-", text_color="gray40", tooltip=""),
                ]
            ], expand_x=True)
        ],
        [sg.Canvas(key="-CANVAS-", expand_x=True, expand_y=True)],
    ]
    # Einstellungen-Tab
    settings_tab = [
        [
            sg.Frame("Echtzeitquelle", [
                [sg.Radio("Direkte Spektrometer-Verbindung (Standard)", "-SOURCE_MODE-", default=True,
                          key="-SOURCE_SPEC-", enable_events=True)],
                [sg.Radio("Datei-basiert (alt)", "-SOURCE_MODE-", default=False,
                          key="-SOURCE_FILE-", enable_events=True)],
            ], expand_x=True, expand_y=True),
            sg.Frame("X-Achse", [
                [sg.Text("Beschriftung"), sg.Push(), sg.Combo(["Wellenlänge", "Wellenzahl"],
                                                              default_value="Wellenlänge",
                                                              key="-X-AXE-NAME-",
                                                              enable_events=True)],
                [sg.Text("Orientierung"), sg.Push(), sg.Combo([">>>", "<<<"],
                                                              default_value=">>>",
                                                              key="-X-AXE-ORIENTATION-",
                                                              enable_events=True)],
            ], expand_x=True, expand_y=True),
            # Frame zum Einstellen des Themes und der Schriftart
            sg.Frame("Aussehen", [
                [sg.Text("Schriftart"), sg.Push(),
                 sg.Combo(selectable_fonts, default_value=selectable_fonts[0],
                          key="-FONT-", enable_events=True),
                 sg.Combo(selectable_font_sizes, default_value=selectable_font_sizes[2],
                          key="-FONT_SIZES-", enable_events=True), ],
                [sg.Text("Theme"), sg.Push(),
                 sg.Combo(selectable_themes, default_value=selectable_themes[0], key="-THEME-", enable_events=True),
                 sg.Button("Übernehmen", key="-APPLY-THEME-")],
            ], expand_x=True, expand_y=True),
            sg.Push(),
        ],
        [
            sg.Frame("Entwickler-Einstellungen", [
                [sg.Checkbox("Debug-Modus", key="-DEBUG-", enable_events=True, default=False)],
            ], size=(300, 100)),
        ],
        [sg.VPush()],
    ]
    # Tab-Group Organisation
    main_layout = [
        [
            sg.TabGroup([[
                sg.Tab("Live View", live_view_tab),
                sg.Tab("Einstellungen", settings_tab)]],
                border_width=0, expand_x=True, expand_y=True,
                key="-TAB_GROUP-", enable_events=True, ),
        ]
    ]
    main_window = sg.Window("IR-Analyse - V2.0", main_layout, size=(1500, 800), resizable=True, finalize=True)
    # Graphen in Canvas einfügen
    figure_canvas_agg = FigureCanvasTkAgg(fig, main_window["-CANVAS-"].TKCanvas)
    figure_canvas_agg.get_tk_widget().pack()

    if "-SPEC_LED-" in main_window.AllKeysDict:
        ui = spectrometer_connection.get_status_ui()
        main_window["-SPEC_LED-"].update(text_color=ui["color"])
        main_window["-SPEC_STATE-"].update(ui["label"])
        main_window["-SPEC_STATUS-"].update(ui["detail"])

    return main_window, figure_canvas_agg


def run_app():
    global main_window
    global figure_canvas_agg
    global values
    global hold
    global realtime_path
    global realtime_source_mode
    global ml_output_on
    global relative_view
    global x_axe_label
    global Wellenlängen
    global Wellenzahlen
    global debug
    global hold_spectrum
    global simulate_on
    global sim_speed
    global dark_spectrum
    global source_spectrum
    global display_mode
    global normalize_view
    global last_live_values

    ml_realtime_thread = None
    simulation_thread = None

    def update_save_mode_hint():
        if realtime_source_mode == "spectrometer":
            if dark_spectrum is not None and source_spectrum is not None:
                hint = "neu: korrigiert (%)"
                tip = "Neue Vergleichsspektren werden kalibriert gespeichert: (raw-dark)/(source-dark)*100"
            else:
                hint = "neu: roh (Dark/Source fehlt)"
                tip = "Für korrigiertes Speichern zuerst Dark und Source aufnehmen"
        else:
            hint = "neu: wie Datei"
            tip = "Dateibasierter Modus: Speichern unverändert wie bisher"

        if "-SAVE_MODE_HINT-" in main_window.AllKeysDict:
            main_window["-SAVE_MODE_HINT-"].update(hint)
            main_window["-SAVE_MODE_HINT-"].set_tooltip(tip)

    # Realtime-Datei bei Start leeren (ursprüngliches Verhalten beibehalten)
    Path(realtime_path).write_text("", encoding="utf-8")

    # Initialisierung der GUI
    main_window, figure_canvas_agg = create_window()

    # Kalibrierungsspektren laden und Status anzeigen
    calibration_io.load_all()
    main_window["-CAL_DARK_STATUS-"].update(calibration_io.dark_status())
    main_window["-CAL_SOURCE_STATUS-"].update(calibration_io.source_status())
    update_save_mode_hint()

    # Erstverbindung vor Start des Realtime-Threads (verhindert Race in nativer Seabreeze-Lib)
    spectrometer_connection.connect()

    # Initalisierung MQTT-Client
    mqtt_connection.start_client()

    # >>> Echtzeit-Loop [START] <<< ------------------------------------------------------------------------------------
    while True:
        event, values = main_window.read(timeout=5)
        if event == sg.WIN_CLOSED:
            break

        # Realtime-Daten im UI-Thread aktualisieren (stabiler mit Tk/Seabreeze)
        try:
            if realtime_source_mode == "spectrometer":
                spectrum = spectrometer_connection.read_live_spectrum()
                if spectrum is not None and len(spectrum) > 0:
                    last_live_values = list(spectrum)
            else:
                data_input = io.import_data(realtime_path)
                if data_input and len(data_input) > 0:
                    last_live_values = list(data_input[-1][2:])
        except Exception:
            if debug:
                print("run_app: Fehler beim Importieren der Echtzeitdaten")

        if realtime_source_mode == "file":
            try:
                io.trim_data(path=realtime_path)
            except Exception:
                if debug:
                    print("run_app: Fehler bei trim_data()")

        # >>> Live-View [START] <<<-------------------------------------------------------------------------------------
        # Events bei Auswahl der Vergleichs-Spektren
        if event == "-VERGLEICH_LIST-":
            # Status des "entfernen"-Buttons überprüfen
            if values["-VERGLEICH_LIST-"] == []:
                main_window["-REMOVE_COMPARE-"].update(disabled=True)
                main_window["-REMOVE_COMPARE-"].set_tooltip("zu entfernendes Vergleichs-Spektrum auswählen")
            else:
                main_window["-REMOVE_COMPARE-"].update(disabled=False)
                main_window["-REMOVE_COMPARE-"].set_tooltip("entfernt ausgewählte Vergleichs-Spektren")

        # Events bei Drücken des Hold-Buttons
        if event == "-HOLD-":
            # Hold-Button aktivieren und Anzeigen anpassen
            if not hold:
                main_window["-ADD_COMPARE-"].update(disabled=False)
                main_window["-ADD_COMPARE-"].set_tooltip("aktuelles Spektrum zu\nVergleichs-Katalog hinzufügen")
                main_window["-HOLD-"].set_tooltip("Live-Vorschau wieder aktivieren")
                main_window["-HOLD-"].update(text="RELEASE", button_color=("white", "green"))
                hold = True
            # Hold-Button deaktivieren und Anzeigen anpassen
            else:
                main_window["-ADD_COMPARE-"].update(disabled=True)
                main_window["-ADD_COMPARE-"].set_tooltip('zuerst "HOLD" aktivieren')
                main_window["-HOLD-"].set_tooltip("friert Live-Vorschau ein")
                main_window["-HOLD-"].update(text="HOLD", button_color=sg.theme_button_color())
                hold = False

        # Button zum Entfernen von Vergleichs-Spektren
        if event == "-REMOVE_COMPARE-":
            io.remove_compare()

        # Button zum Hinzufügen von Vergleichs-Spektren
        if event == "-ADD_COMPARE-":
            # Variable aus der HOLD-Abfrage - steht zur Verfügung, da zuerst HOLD-aktiviert werden muss
            spectrum_to_save = list(hold_spectrum)

            if realtime_source_mode == "spectrometer":
                # Neue Spektren aus Live-Messung nur als korrigierte Variante speichern,
                # damit sie zum bestehenden Vergleichskatalog passen.
                if dark_spectrum is None or source_spectrum is None:
                    sg.popup_no_buttons("Bitte zuerst Dark und Source aufnehmen", auto_close=True, auto_close_duration=2)
                else:
                    if spectrum_to_save and spectrum_to_save[0] == 0.0:
                        corrected = compute.apply_calibration(spectrum_to_save[1:])
                        spectrum_to_save = [0.0] + corrected
                    else:
                        spectrum_to_save = compute.apply_calibration(spectrum_to_save)
                    io.add_compare(spectrum_to_add=spectrum_to_save)
            else:
                # Dateibasierter Modus: Daten bleiben wie zuvor gespeichert.
                io.add_compare(spectrum_to_add=spectrum_to_save)

        # Events bei Änderung der X-Achsen-Beschriftung
        if event == "-X-AXE-NAME-" or event == "-X-AXE-ORIENTATION-":
            # X-Achsen-Beschriftung anpassen
            if values["-X-AXE-NAME-"] == "Wellenlänge":
                x_axe_label = "Wellenlänge"
            elif values["-X-AXE-NAME-"] == "Wellenzahl":
                x_axe_label = "Wellenzahl"

        # Realtime-Datei auswählen
        if event == "-REALTIME_CHOOSE-":
            try:
                filename = sg.popup_get_file('Datei auswählen', no_window=True)
                if filename is not None:
                    main_window["-REALTIME_CHOOSE-"].update("..." + filename[-40:] + "  •  ändern")
                    Path(filename).write_text("", encoding="utf-8")
                    realtime_path = filename
            except:
                print("keine Datei ausgewählt")

        if event == "-SPEC_RECONNECT-":
            spectrometer_connection.disconnect()
            spectrometer_connection.connect()

        # Dark-Spektrum aufnehmen
        if event == "-CAL_DARK-":
            if last_live_values:
                dark_spectrum = list(last_live_values)
                calibration_io.save_dark(dark_spectrum)
                main_window["-CAL_DARK_STATUS-"].update(calibration_io.dark_status())
                update_save_mode_hint()
            else:
                sg.popup_no_buttons("Kein Spektrum verfügbar", auto_close=True, auto_close_duration=2)

        # Source-Spektrum aufnehmen
        if event == "-CAL_SOURCE-":
            if last_live_values:
                source_spectrum = list(last_live_values)
                calibration_io.save_source(source_spectrum)
                main_window["-CAL_SOURCE_STATUS-"].update(calibration_io.source_status())
                update_save_mode_hint()
            else:
                sg.popup_no_buttons("Kein Spektrum verfügbar", auto_close=True, auto_close_duration=2)

        # Anzeigemodus: Rohwerte / Prozent
        if event == "-DISPLAY_RAW-":
            display_mode = "raw"
        if event == "-DISPLAY_PCT-":
            if dark_spectrum is None or source_spectrum is None:
                display_mode = "raw"
                main_window["-DISPLAY_RAW-"].update(value=True)
                main_window["-DISPLAY_PCT-"].update(value=False)
                sg.popup_no_buttons("Bitte zuerst Dark und Source aufnehmen", auto_close=True, auto_close_duration=2)
            else:
                display_mode = "percent"

        # Normalisierung aller Spektren
        if event == "-NORMALIZE_VIEW-":
            normalize_view = values["-NORMALIZE_VIEW-"]

        if event == "-SOURCE_SPEC-":
            realtime_source_mode = "spectrometer"
            main_window["-LIVE_SPEC_STATUS_COL-"].update(visible=True)
            main_window["-LIVE_FILE_SETTINGS_COL-"].update(visible=False)
            update_save_mode_hint()

        if event == "-SOURCE_FILE-":
            realtime_source_mode = "file"
            main_window["-LIVE_SPEC_STATUS_COL-"].update(visible=False)
            main_window["-LIVE_FILE_SETTINGS_COL-"].update(visible=True)
            update_save_mode_hint()

        # Test-Knöpfe (zum Testen von Funktionen)
        if event == "-TEST_BUTTON1-":
            client.publish("/result", "1")
        if event == "-TEST_BUTTON2-":
            client.publish("/result", "2")
        if event == "-TEST_BUTTON3-":
            client.publish("/result", "3")

        # ML-Output Knopf
        if event == "-ML_START-":
            mt.ml_output_single()

        # Realtime-Analyse Checkbox
        if event == "-REALTIME_ML-":
            if values["-REALTIME_ML-"]:
                ml_output_on = True
                if ml_realtime_thread is None or not ml_realtime_thread.is_alive():
                    ml_realtime_thread = threading.Thread(target=mt.ml_output_loop, daemon=True)
                    ml_realtime_thread.start()
            else:
                ml_output_on = False
                try:
                    if ml_realtime_thread is not None and ml_realtime_thread.is_alive():
                        ml_realtime_thread.join(timeout=1)
                except:
                    if debug:
                        print("Realtime-ML: Thread nicht aktiv")

        # Relative Ansicht aktivieren/deaktivieren
        if event == "-RELATIVE_VIEW-":
            if values["-RELATIVE_VIEW-"]:
                relative_view = True
            else:
                relative_view = False


    # Live-View [ENDE] -------------------------------------------------------------------------------------------------
    #
    #
    #
    #
    #
        # >>> Einstellungen [START] <<<---------------------------------------------------------------------------------
        # Thema, Schriftart und Schriftgröße werden übernommen
        if event == "-APPLY-THEME-":
            main_new_window, figure_canvas_agg = create_window(theme=main_window["-THEME-"].get(),
                                                               font=main_window["-FONT-"].get(),
                                                               font_size=main_window["-FONT_SIZES-"].get())
            main_window.close()
            main_window = main_new_window

        if event == "-X-AXE-NAME-":
            if values["-X-AXE-NAME-"] == "Wellenlänge":
                x_axe_label = "Wellenlänge"
            elif values["-X-AXE-NAME-"] == "Wellenzahl":
                x_axe_label = "Wellenzahl"

        if event == "-X-AXE-ORIENTATION-":
            if values["-X-AXE-ORIENTATION-"] == ">>>":
                # Wellenlöngen und Wellennummern aufsteigend sortieren
                Wellenlängen = sorted(Wellenlängen)
                Wellenzahlen = sorted(Wellenzahlen, reverse=True)
            if values["-X-AXE-ORIENTATION-"] == "<<<":
                # Wellenlängen und Wellennummern absteigend sortieren
                Wellenlängen = sorted(Wellenlängen, reverse=True)
                Wellenzahlen = sorted(Wellenzahlen)

        if event == "-DEBUG-":
            debug = values["-DEBUG-"]
    # Events zu Einstellungen [ENDE] -----------------------------------------------------------------------------------
    #
    #
    #
    #
    #
        # Hold-Spectrum setzten
        if not hold:
            try:
                if realtime_source_mode == "spectrometer":
                    hold_spectrum = [0.0] + list(last_live_values) if last_live_values else []
                else:
                    hold_spectrum = io.import_data(realtime_path)[-1][1:]
            except:
                if debug:
                    print("hold_spektrum: Realtime Daten leer")

        # >>> Simulation von Echtzeitdaten
        if event == "-SIMULATION-":
            if values["-SIMULATION-"]:
                simulate_on = True
                if simulation_thread is None or not simulation_thread.is_alive():
                    simulation_thread = threading.Thread(target=mt.simulate, daemon=True)
                    simulation_thread.start()
            else:
                simulate_on = False
                if simulation_thread is not None and simulation_thread.is_alive():
                    simulation_thread.join(timeout=1)

        if event == "-SIM_DELAY-":
            sim_speed = float(values["-SIM_DELAY-"]) / 1000

        if "-SPEC_STATUS-" in main_window.AllKeysDict:
            try:
                ui = spectrometer_connection.get_status_ui()
                main_window["-SPEC_LED-"].update(text_color=ui["color"])
                main_window["-SPEC_STATE-"].update(ui["label"])
                main_window["-SPEC_STATUS-"].update(ui["detail"])
            except Exception:
                pass

        # Live-Graph immer im UI-Thread zeichnen (Tk/Matplotlib ist nicht thread-safe)
        try:
            if last_live_values:
                if relative_view:
                    update_graphs.live_relative(y_values=last_live_values)
                else:
                    update_graphs.live_absolute(y_values=last_live_values)
                if figure_canvas_agg is not None:
                    figure_canvas_agg.draw()
        except Exception:
            if debug:
                print("UI-Loop: Fehler beim Zeichnen des Live-Graphen")

    # Echtzeit-Loop [ENDE] ---------------------------------------------------------------------------------------------
    spectrometer_connection.disconnect()
    main_window.close()


# GUI [ENDE] -----------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    run_app()
