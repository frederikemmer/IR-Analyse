import os

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


class _QueueWindow(dict):
    def __init__(self, events):
        super().__init__()
        self._events = list(events)
        self.AllKeysDict = {
            "-SPEC_STATUS-": True,
            "-SAVE_MODE_HINT-": True,
            "-DISPLAY_RAW-": True,
            "-DISPLAY_PCT-": True,
        }

    def __getitem__(self, item):
        if item not in self:
            self[item] = _DummyElement()
        return dict.__getitem__(self, item)

    def read(self, timeout=5):
        if self._events:
            return self._events.pop(0)
        return app.sg.WIN_CLOSED, {}

    def close(self):
        return None


class _DummyThread:
    def __init__(self, *args, **kwargs):
        pass

    def start(self):
        return None

    def join(self, timeout=None):
        return None


def _base_values(**extra):
    values = {
        "-VERGLEICH_LIST-": [],
        "-COMPARE_INPUT-": "Neu",
        "-COMPARE_COMBO-": "PE",
        "-COMPARE_COLOR-": "#00ff00",
        "-DISPLAY_RAW-": True,
        "-DISPLAY_PCT-": False,
        "-NORMALIZE_VIEW-": False,
        "-RELATIVE_VIEW-": False,
    }
    values.update(extra)
    return values


def _patch_common(monkeypatch, window, realtime_file):
    dark_file = realtime_file.parent / "dark.json"
    source_file = realtime_file.parent / "source.json"

    monkeypatch.setattr(app, "create_window", lambda *args, **kwargs: (window, object()), raising=True)
    monkeypatch.setattr(app.threading, "Thread", _DummyThread, raising=True)
    monkeypatch.setattr(app.mqtt_connection, "start_client", lambda self=None: None, raising=True)
    monkeypatch.setattr(app.spectrometer_connection, "connect", lambda self=None: True, raising=True)
    monkeypatch.setattr(app.spectrometer_connection, "disconnect", lambda self=None: None, raising=True)
    monkeypatch.setattr(app.spectrometer_connection, "read_live_spectrum", lambda self=None: None, raising=True)
    monkeypatch.setattr(app, "realtime_path", str(realtime_file), raising=True)
    monkeypatch.setattr(app, "dark_spectrum_path", str(dark_file), raising=True)
    monkeypatch.setattr(app, "source_spectrum_path", str(source_file), raising=True)


def test_run_app_hint_spectrometer_without_refs(monkeypatch, tmp_path):
    compare_file = tmp_path / "compare.txt"
    compare_file.write_text("Name;Material;Farbe;S0;S1\n", encoding="utf-8")
    realtime_file = tmp_path / "rt.txt"

    window = _QueueWindow([(app.sg.WIN_CLOSED, _base_values())])

    _patch_common(monkeypatch, window, realtime_file)
    monkeypatch.setattr(app, "compare_path", str(compare_file), raising=True)
    monkeypatch.setattr(app, "realtime_source_mode", "spectrometer", raising=True)
    monkeypatch.setattr(app, "dark_spectrum", None, raising=True)
    monkeypatch.setattr(app, "source_spectrum", None, raising=True)

    app.run_app()

    hint_updates = window["-SAVE_MODE_HINT-"].updated
    assert any("roh" in (u[0][0] if u[0] else "") for u in hint_updates)


def test_run_app_hint_file_mode(monkeypatch, tmp_path):
    compare_file = tmp_path / "compare.txt"
    compare_file.write_text("Name;Material;Farbe;S0;S1\n", encoding="utf-8")
    realtime_file = tmp_path / "rt.txt"

    window = _QueueWindow([(app.sg.WIN_CLOSED, _base_values())])

    _patch_common(monkeypatch, window, realtime_file)
    monkeypatch.setattr(app, "compare_path", str(compare_file), raising=True)
    monkeypatch.setattr(app, "realtime_source_mode", "file", raising=True)

    app.run_app()

    hint_updates = window["-SAVE_MODE_HINT-"].updated
    assert any("Datei" in (u[0][0] if u[0] else "") for u in hint_updates)


def test_add_compare_spectrometer_requires_dark_source(monkeypatch, tmp_path):
    compare_file = tmp_path / "compare.txt"
    compare_file.write_text("Name;Material;Farbe;S0;S1\n", encoding="utf-8")
    realtime_file = tmp_path / "rt.txt"

    events = [
        ("-ADD_COMPARE-", _base_values()),
        (app.sg.WIN_CLOSED, _base_values()),
    ]
    window = _QueueWindow(events)

    _patch_common(monkeypatch, window, realtime_file)
    monkeypatch.setattr(app, "compare_path", str(compare_file), raising=True)
    monkeypatch.setattr(app, "realtime_source_mode", "spectrometer", raising=True)
    monkeypatch.setattr(app, "dark_spectrum", None, raising=True)
    monkeypatch.setattr(app, "source_spectrum", None, raising=True)
    monkeypatch.setattr(app, "hold_spectrum", [0.0, 10.0, 20.0], raising=True)

    popups = {"n": 0}

    def _popup(*args, **kwargs):
        popups["n"] += 1

    monkeypatch.setattr(app.sg, "popup_no_buttons", _popup, raising=True)

    app.run_app()

    lines = compare_file.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert popups["n"] >= 1


def test_add_compare_spectrometer_saves_corrected(monkeypatch, tmp_path):
    compare_file = tmp_path / "compare.txt"
    compare_file.write_text("Name;Material;Farbe;S0;S1;S2\n", encoding="utf-8")
    realtime_file = tmp_path / "rt.txt"

    events = [
        ("-ADD_COMPARE-", _base_values()),
        (app.sg.WIN_CLOSED, _base_values()),
    ]
    window = _QueueWindow(events)

    _patch_common(monkeypatch, window, realtime_file)
    monkeypatch.setattr(app, "compare_path", str(compare_file), raising=True)
    monkeypatch.setattr(app, "realtime_source_mode", "spectrometer", raising=True)
    monkeypatch.setattr(app, "dark_spectrum", [0.0, 0.0], raising=True)
    monkeypatch.setattr(app, "source_spectrum", [20.0, 40.0], raising=True)
    monkeypatch.setattr(app, "hold_spectrum", [0.0, 10.0, 20.0], raising=True)

    app.run_app()

    content = compare_file.read_text(encoding="utf-8").replace(" ", "")
    assert "Neu;PE;#00ff00;0.0;50.0;50.0" in content


def test_add_compare_file_mode_saves_raw(monkeypatch, tmp_path):
    compare_file = tmp_path / "compare.txt"
    compare_file.write_text("Name;Material;Farbe;S0;S1;S2\n", encoding="utf-8")
    realtime_file = tmp_path / "rt.txt"

    events = [
        ("-ADD_COMPARE-", _base_values()),
        (app.sg.WIN_CLOSED, _base_values()),
    ]
    window = _QueueWindow(events)

    _patch_common(monkeypatch, window, realtime_file)
    monkeypatch.setattr(app, "compare_path", str(compare_file), raising=True)
    monkeypatch.setattr(app, "realtime_source_mode", "file", raising=True)
    monkeypatch.setattr(app, "hold_spectrum", [0.0, 10.0, 20.0], raising=True)

    app.run_app()

    content = compare_file.read_text(encoding="utf-8").replace(" ", "")
    assert "Neu;PE;#00ff00;0.0;10.0;20.0" in content


def test_display_pct_without_refs_falls_back_to_raw(monkeypatch, tmp_path):
    compare_file = tmp_path / "compare.txt"
    compare_file.write_text("Name;Material;Farbe;S0;S1\n", encoding="utf-8")
    realtime_file = tmp_path / "rt.txt"

    values = _base_values(**{"-DISPLAY_RAW-": False, "-DISPLAY_PCT-": True})
    events = [
        ("-DISPLAY_PCT-", values),
        (app.sg.WIN_CLOSED, values),
    ]
    window = _QueueWindow(events)

    _patch_common(monkeypatch, window, realtime_file)
    monkeypatch.setattr(app, "compare_path", str(compare_file), raising=True)
    monkeypatch.setattr(app, "dark_spectrum", None, raising=True)
    monkeypatch.setattr(app, "source_spectrum", None, raising=True)

    popups = {"n": 0}

    def _popup(*args, **kwargs):
        popups["n"] += 1

    monkeypatch.setattr(app.sg, "popup_no_buttons", _popup, raising=True)

    app.run_app()

    assert app.display_mode == "raw"
    assert popups["n"] >= 1
