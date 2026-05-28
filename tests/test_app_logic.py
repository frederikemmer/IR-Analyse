import os

import pandas as pd

os.environ.setdefault("IR_ANALYSE_TEST_MODE", "1")

import main as app


class _DummyElement:
    def __init__(self):
        self.updated = []
        self.tooltips = []

    def update(self, *args, **kwargs):
        self.updated.append((args, kwargs))

    def set_tooltip(self, value):
        self.tooltips.append(value)


class _DummyWindow(dict):
    def __init__(self):
        super().__init__()
        self.AllKeysDict = {"-SPEC_STATUS-": True}

    def __getitem__(self, item):
        if item not in self:
            self[item] = _DummyElement()
        return dict.__getitem__(self, item)

    def read(self, timeout=5):
        return app.sg.WIN_CLOSED, {}

    def close(self):
        return None


class _DummyMQTTClient:
    def __init__(self):
        self.on_connect = None
        self.on_message = None
        self.connect_args = None
        self.loop_started = False
        self.subscribed = None

    def connect(self, host, port, keepalive):
        self.connect_args = (host, port, keepalive)

    def loop_start(self):
        self.loop_started = True

    def subscribe(self, topic):
        self.subscribed = topic


class _DummySpectrometer:
    def __init__(self, values):
        self.values = values
        self.closed = False

    def intensities(self, correct_nonlinearity=False):
        return self.values

    def close(self):
        self.closed = True


def test_io_add_compare_and_remove_compare(tmp_path, monkeypatch):
    compare_file = tmp_path / "compare.txt"
    compare_file.write_text("Name;Material;Farbe;S0;S1\nA;PET;#ff0000;0;1\n", encoding="utf-8")

    monkeypatch.setattr(app, "compare_path", str(compare_file), raising=True)
    monkeypatch.setattr(app, "main_window", _DummyWindow(), raising=True)
    monkeypatch.setattr(
        app,
        "values",
        {
            "-COMPARE_INPUT-": "B",
            "-COMPARE_COMBO-": "PE",
            "-COMPARE_COLOR-": "#00ff00",
            "-VERGLEICH_LIST-": ["A"],
        },
        raising=True,
    )

    app.io.add_compare([2.0, 3.0])

    content_after_add = compare_file.read_text(encoding="utf-8").replace(" ", "")
    assert "B;PE;#00ff00;2.0;3.0" in content_after_add

    app.io.remove_compare()

    content_after_remove = compare_file.read_text(encoding="utf-8").replace(" ", "")
    assert "A;PET;#ff0000;0;1" not in content_after_remove
    assert "B;PE;#00ff00;2.0;3.0" in content_after_remove


def test_update_graphs_absolute_and_relative_without_compare(monkeypatch):
    monkeypatch.setattr(app, "values", {"-VERGLEICH_LIST-": []}, raising=True)

    app.x_axe_label = "Wellenlänge"
    x_values = app.update_graphs.axe_label()
    assert x_values == app.Wellenlängen
    assert "Wellenlänge" in app.live.get_xlabel()

    app.hold = False
    app.update_graphs.live_absolute(y_values=[1.0] * (len(x_values) - 1))
    assert len(app.live.lines) >= 1

    app.hold = True
    app.hold_spectrum = [0.0] + [float(i) for i in range(len(x_values) - 1)]
    app.update_graphs.live_relative(y_values=[float(i) for i in range(len(x_values) - 1)])
    assert len(app.live.lines) >= 1


def test_mqtt_connection_start_and_on_connected(monkeypatch):
    dummy = _DummyMQTTClient()
    monkeypatch.setattr(app, "client", dummy, raising=True)

    app.mqtt_connection.start_client()
    app.mqtt_connection.on_connected(dummy, None, None, 0)

    assert dummy.on_connect == app.mqtt_connection.on_connected
    assert dummy.on_message == app.mqtt_connection.on_message
    assert dummy.connect_args == ("192.168.178.147", 1883, 60)
    assert dummy.loop_started is True
    assert dummy.subscribed == "/measure"


def test_ml_output_single_updates_ui(monkeypatch):
    db = pd.read_csv("data/Sample-Database V1 (Sphere).csv", sep=";")
    sample = db.drop(columns=["#", "Material"]).iloc[0].to_list()

    monkeypatch.setattr(app, "hold_spectrum", [0.0] + sample, raising=True)
    monkeypatch.setattr(app, "main_window", _DummyWindow(), raising=True)

    call_idx = {"n": 0}

    def _fake_load_model_pred(_spectrum_data, _path):
        # Reihenfolge in ml_output_single: neural -> knn -> l_reg
        values = ["2", 3, 1]
        out = values[call_idx["n"]]
        call_idx["n"] += 1
        return out

    monkeypatch.setattr(app.ML, "load_model_pred", _fake_load_model_pred)

    app.mt.ml_output_single()

    updates = app.main_window["-KI_OUTPUT_LIST-"].updated
    assert len(updates) == 1
    payload = updates[0][0][0]
    assert len(payload) == 3


def test_spectrometer_read_live_spectrum_paths(monkeypatch):
    app.spectrometer_connection.spectrometer = _DummySpectrometer([1, 2, 3])
    app.spectrometer_connection.connected = True

    values = app.spectrometer_connection.read_live_spectrum()
    assert values == [1.0, 2.0, 3.0]

    monkeypatch.setattr(app.spectrometer_connection, "connect", lambda self=None: False, raising=True)
    app.spectrometer_connection.connected = False
    app.spectrometer_connection.spectrometer = None

    assert app.spectrometer_connection.read_live_spectrum() is None


def test_run_app_smoke(monkeypatch, tmp_path):
    realtime_file = tmp_path / "rt.txt"

    class _DummyThread:
        def __init__(self, *args, **kwargs):
            pass

        def start(self):
            return None

        def join(self):
            return None

    dummy_window = _DummyWindow()

    monkeypatch.setattr(app, "create_window", lambda *args, **kwargs: (dummy_window, object()), raising=True)
    monkeypatch.setattr(app.threading, "Thread", _DummyThread, raising=True)
    monkeypatch.setattr(app.mqtt_connection, "start_client", lambda self=None: None, raising=True)
    monkeypatch.setattr(app.spectrometer_connection, "connect", lambda self=None: True, raising=True)
    monkeypatch.setattr(app.spectrometer_connection, "disconnect", lambda self=None: None, raising=True)
    monkeypatch.setattr(app, "realtime_path", str(realtime_file), raising=True)

    app.run_app()

    assert realtime_file.exists()
