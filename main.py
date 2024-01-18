# >>> Imports [Start] <<< ----------------------------------------------------------------------------------------------
import PySimpleGUI as sg
from pathlib import Path

# >>> Visualisierung ---------------------------------------------------------------------------------------------------
from matplotlib import pyplot as plt
from matplotlib import use as mpl_use
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# >>> SciKit learn -----------------------------------------------------------------------------------------------------
from sklearn.pipeline import Pipeline                   #
from sklearn.preprocessing import StandardScaler        #   Preprocessing
from sklearn.linear_model import LinearRegression       #
from sklearn.neighbors import KNeighborsRegressor       #
# from sklearn.svm import SVR                             #
from sklearn.neural_network import MLPClassifier        #   ML_Modelle
from sklearn.model_selection import train_test_split    #
# from sklearn.model_selection import GridSearchCV        #   Validierung, Tests, etc.

# >>> Datenverarbeitung, Hilfsmittel, etc. --------------------------------------------------------------------------
import pandas as pd
import numpy as np
import joblib
import random
import threading
import time
import paho.mqtt.client as mqtt

mpl_use("TKAgg")  # Modus für Matplotlib → Verwendung von TKinter
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
Path(realtime_path).write_text("", encoding="utf-8")    # leeren der Realtime-Daten (teils MBs zu lesen...)

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
    "4": "HDPE",
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
client = mqtt.Client()                                  # Initialisierung MQTT-Client

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
        Vorhersage_neural = str(ML.load_model_pred(hold_spectrum[1:], "models/neural_5000.model"))
        Vorhersage_neural = Vorhersage_neural.replace("[", "").replace("]", "")
        Vorhersage_neural = mat_num[Vorhersage_neural]

        Vorhersage_knn = str(int(ML.load_model_pred(hold_spectrum[1:], "models/knn_10.model")))
        Vorhersage_knn = Vorhersage_knn.replace("[", "").replace("]", "")
        Vorhersage_knn = mat_num[Vorhersage_knn]

        Vorhersage_l_reg = str(int(ML.load_model_pred(hold_spectrum[1:], "models/l_reg.model")))
        Vorhersage_l_reg = Vorhersage_l_reg.replace("[", "").replace("]", "")
        Vorhersage_l_reg = mat_num[Vorhersage_l_reg]

        main_window["-KI_OUTPUT_LIST-"].update([["Neurales Netz: ", Vorhersage_neural],
                                                ["KNN: ", Vorhersage_knn],
                                                ["Lineare Regression: ", Vorhersage_l_reg]
                                                ])

    # Echtzeitdaten und Vergleichs_Spektren.txt anzeigen, wenn ausgewählt
    # noinspection PyTypeChecker
    def Realtime_Loop(self=None):
        while True:
            # Daten importieren
            try:
                data_input = io.import_data(realtime_path)
            except:
                if debug:
                    print("Realtime_Loop: Fehler beim Importieren der Echtzeitdaten")

            # abfragen ob relativ/absolut - Dann entsprechend Graphen updaten
            try:
                if relative_view:
                    update_graphs.live_relative(y_values=data_input[-1][2:])
                else:
                    update_graphs.live_absolute(y_values=data_input[-1][2:])
                figure_canvas_agg.draw()
            except:
                if debug:
                        print(f'Realtime_Loop: Fehler bei update_graphs.live_absolute() / ...compare()')

            # Daten zum Schluss auf letzten Wert trimmen
            try:
                io.trim_data(path=realtime_path)
            except:
                if debug:
                    print("Realtime_Loop: Fehler bei trim_data()")

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
    def prep_data(database_path="data/Sample-Database V1 (Sphere).csv", test_size=0.2, random_state=random.randint(1, 1000000)):
        data = pd.read_csv(database_path, sep=";")

        X = data.drop(columns=["#", "Material"]).values     # alles bis auf "Material"
        y = data[["Material"]].values                       # nur "Material"

        X = np.array(X)                                     # in numpy-Array umwandeln
        y = np.array(y)                                     # in numpy-Array umwandeln

        # Daten in Trainings- und Testdaten aufteilen
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        return X_train, X_test, y_train, y_test

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
        X_train, X_test, y_train, y_test = ML.prep_data(random_state=randome_state)

        pipe_neural = Pipeline([
            ("scale", StandardScaler()),
            ("model", MLPClassifier(random_state=randome_state, max_iter=iterrations))
        ])
        pipe_neural.fit(X_train, y_train.ravel())
        joblib.dump(pipe_neural, "models/neural_5000.model")

        # Statistische Werte ausgeben
        # Varianz, Durchschnitt, Standartabweichung = ML.get_stat_value(pipe_neural.predict(X_test), y_test, 3)

        spectrum_data = np.array(spectrum_data).reshape(1, -1)
        prediction = pipe_neural.predict(spectrum_data)
        return prediction

    # KNN trainieren und Prediction ausgeben
    def knn(spectrum_data, neighbors=10):
        X_train, X_test, y_train, y_test = ML.prep_data()

        pipe_knn = Pipeline([
            ("scale", StandardScaler()),
            ("model", KNeighborsRegressor(n_neighbors=neighbors))
        ])
        pipe_knn.fit(X_train, y_train.ravel())
        joblib.dump(pipe_knn, "models/knn_10.model")

        # Statistische Werte ausgeben
        # Varianz, Durchschnitt, Standartabweichung = ML.get_stat_value(pipe_knn.predict(X_test), y_test, 3)

        spectrum_data = np.array(spectrum_data).reshape(1, -1)
        prediction = pipe_knn.predict(spectrum_data)
        return prediction

    # Lineare Regression trainieren und Prediction ausgeben
    def l_reg(spectrum_data):
        X_train, X_test, y_train, y_test = ML.prep_data()

        pipe_l_reg = Pipeline([
            ("scale", StandardScaler()),
            ("model", LinearRegression())
        ])
        pipe_l_reg.fit(X_train, y_train.ravel())
        joblib.dump(pipe_l_reg, "models/l_reg.model")

        # Statistische Werte ausgeben
        # Varianz, Durchschnitt, Standartabweichung = ML.get_stat_value(pipe_l_reg.predict(X_test), y_test, 3)

        spectrum_data = np.array(spectrum_data).reshape(1, -1)
        prediction = pipe_l_reg.predict(spectrum_data)
        return prediction

    # Model laden und Prediction ausgeben
    def load_model_pred(spectrum_data, path = "models/neural_5000.model"):
        loaded_model = joblib.load(path)
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
        live.set_ylabel("Intensität [%]")
        return x_values

    def live_absolute(y_values=None):
        if y_values is None:
            y_values = []
        x_values = update_graphs.axe_label()

        if not hold:
            live.plot(x_values[1:], y_values, color=live_color)
        else:
            live.plot(x_values[1:], hold_spectrum[1:], color=hold_color)

        for i in values["-VERGLEICH_LIST-"]:
            # [1:] da erster Spektrum-Wert immer 0 ist
            live.plot(x_values[1:], io.get_compare_spektrum(i)[1:], color=io.get_compare_color(i))

    def live_relative(y_values=None):
        if y_values is None:
            y_values = []
        x_values = update_graphs.axe_label()

        if not hold:
            live.plot(x_values[1:], compute.relative_values(values=y_values), color=live_color)
        else:
            live.plot(x_values[1:], compute.relative_values(values=hold_spectrum[1:]), color=hold_color)

        for i in values["-VERGLEICH_LIST-"]:
            # [1:] da erster Spektrum-Wert immer 0 ist
            live.plot(x_values[1:], compute.relative_values(values=io.get_compare_spektrum(i)[1:]), color=io.get_compare_color(i))

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
    def on_connected(client, userdata, flags, returncode):
        print("Mit MQTT-Server verbunden. Ergebnis: " + str(returncode))
        client.subscribe("/measure")

    # Funktion wird bei Erhalten einer Nachricht ausgeführt
    def on_message(client, userdata, message):
        print(f"{message.topic}: {str(message.payload)}")

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
            sg.Frame("Realtime-Einstellungen", [
                [sg.Text("Realtime-Datei auswählen:")],
                [sg.Button("..." + realtime_path[-40:] + "  •  ändern", key="-REALTIME_CHOOSE-", enable_events=True)],
                [sg.Checkbox("Echtzeitdaten simulieren", key="-SIMULATION-", enable_events=True, default=simulate_on),
                 sg.Push(), sg.InputText("300", key="-SIM_DELAY-", size=(5, 1), enable_events=True),
                 sg.Text("ms"), sg.Push()],
                [sg.Checkbox("relative Daten", key="-RELATIVE_VIEW-", enable_events=True, default=relative_view)],
                [sg.Button(">1<", key="-TEST_BUTTON1-", enable_events=True),
                 sg.Button(">2<", key="-TEST_BUTTON2-", enable_events=True),
                 sg.Button(">3<", key="-TEST_BUTTON3-", enable_events=True)],

            ], size=(400, 200)),
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
                [sg.Input(key="-COMPARE_INPUT-", enable_events=True, size=(40, 1), expand_x=True, pad=(5, 0)),
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
        [sg.Canvas(key="-CANVAS-", expand_x=True, expand_y=True)],
    ]
    # Einstellungen-Tab
    settings_tab = [
        [
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
    main_window = sg.Window("IR-Analyse - V1.0", main_layout, size=(1500, 800), resizable=True, finalize=True)
    # Graphen in Canvas einfügen
    figure_canvas_agg = FigureCanvasTkAgg(fig, main_window["-CANVAS-"].TKCanvas)
    figure_canvas_agg.get_tk_widget().pack()
    return main_window, figure_canvas_agg


# Initialisierung der GUI
main_window, figure_canvas_agg = create_window()
threading.Thread(target=mt.Realtime_Loop, daemon=True).start()

# Initalisierung MQTT-Client
mqtt_connection.start_client()

# GUI [ENDE] -----------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
#
#
#
#
# ----------------------------------------------------------------------------------------------------------------------
# >>> Echtzeit-Loop [START] <<< ----------------------------------------------------------------------------------------
while True:
    event, values = main_window.read(timeout=5)
    if event == sg.WIN_CLOSED:
        break

    # >>> Live-View [START] <<<-----------------------------------------------------------------------------------------
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
        io.add_compare(spectrum_to_add=hold_spectrum)

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
            ml_realtime = threading.Thread(target=mt.ml_output_loop(), daemon=True)
            ml_realtime.start()
        else:
            ml_output_on = False
            try:
                ml_realtime.join()
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
    # >>> Einstellungen [START] <<<-------------------------------------------------------------------------------------
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
            hold_spectrum = io.import_data(realtime_path)[-1][1:]
        except:
            if debug:
                print("hold_spektrum: Realtime Daten leer")

    # >>> Simulation von Echtzeitdaten
    if event == "-SIMULATION-":
        if values["-SIMULATION-"]:
            simulate_on = True
            simulation = threading.Thread(target=mt.simulate, daemon=True)
            simulation.start()
        else:
            simulate_on = False
            simulation.join()

    if event == "-SIM_DELAY-":
        sim_speed = float(values["-SIM_DELAY-"]) / 1000

# Echtzeit-Loop [ENDE] -------------------------------------------------------------------------------------------------
main_window.close()
