import os

os.environ.setdefault("IR_ANALYSE_TEST_MODE", "1")

import main as app


class _DummySpec:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


def test_status_ui_connected():
    app.spectrometer_connection.connected = True
    app.spectrometer_connection.status_text = "Verbunden: Demo"
    app.spectrometer_connection.last_error = ""

    ui = app.spectrometer_connection.get_status_ui()

    assert ui["label"] == "Verbunden"
    assert ui["color"] == "#2e7d32"
    assert "Verbunden" in ui["detail"]


def test_status_ui_disconnected():
    app.spectrometer_connection.connected = False
    app.spectrometer_connection.status_text = "Kein Spektrometer gefunden"
    app.spectrometer_connection.last_error = ""

    ui = app.spectrometer_connection.get_status_ui()

    assert ui["label"] == "Getrennt"
    assert ui["color"] == "#f9a825"


def test_status_ui_error_and_line():
    app.spectrometer_connection.connected = False
    app.spectrometer_connection.status_text = "Verbindung fehlgeschlagen"
    app.spectrometer_connection.last_error = "USB timeout"

    ui = app.spectrometer_connection.get_status_ui()
    line = app.spectrometer_connection.get_status_line()

    assert ui["label"] == "Fehler"
    assert ui["color"] == "#c62828"
    assert "USB timeout" in line


def test_disconnect_closes_open_spectrometer():
    dummy = _DummySpec()
    app.spectrometer_connection.spectrometer = dummy
    app.spectrometer_connection.connected = True

    app.spectrometer_connection.disconnect()

    assert dummy.closed is True
    assert app.spectrometer_connection.connected is False
    assert app.spectrometer_connection.spectrometer is None
