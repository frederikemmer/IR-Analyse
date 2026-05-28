import os

os.environ.setdefault("IR_ANALYSE_TEST_MODE", "1")

import main as app


def test_apply_calibration_with_explicit_refs():
    raw = [10.0, 30.0]
    dark = [0.0, 10.0]
    source = [20.0, 50.0]

    corrected = app.compute.apply_calibration(raw_values=raw, dark=dark, source=source)

    assert corrected == [50.0, 50.0]


def test_apply_calibration_uses_global_refs(monkeypatch):
    monkeypatch.setattr(app, "dark_spectrum", [0.0, 10.0], raising=True)
    monkeypatch.setattr(app, "source_spectrum", [20.0, 50.0], raising=True)

    corrected = app.compute.apply_calibration(raw_values=[10.0, 30.0])

    assert corrected == [50.0, 50.0]


def test_apply_calibration_handles_zero_denominator():
    corrected = app.compute.apply_calibration(
        raw_values=[10.0, 20.0],
        dark=[0.0, 5.0],
        source=[20.0, 5.0],
    )

    assert corrected == [50.0, 0.0]


def test_normalize_single_range_and_flatline():
    assert app.compute.normalize_single([10.0, 20.0, 30.0]) == [0.0, 50.0, 100.0]
    assert app.compute.normalize_single([7.0, 7.0, 7.0]) == [50.0, 50.0, 50.0]


def test_calibration_io_status_strings(tmp_path, monkeypatch):
    dark_path = tmp_path / "dark.json"
    source_path = tmp_path / "source.json"
    source_path.write_text("[1.0]", encoding="utf-8")

    monkeypatch.setattr(app, "dark_spectrum_path", str(dark_path), raising=True)
    monkeypatch.setattr(app, "source_spectrum_path", str(source_path), raising=True)
    monkeypatch.setattr(app, "dark_spectrum", None, raising=True)
    monkeypatch.setattr(app, "source_spectrum", [1.0], raising=True)

    assert app.calibration_io.dark_status() == "Nicht aufgenommen"
    assert len(app.calibration_io.source_status()) == 19


def test_calibration_io_timestamp_text_no_file(tmp_path, monkeypatch):
    dark_path = tmp_path / "dark.json"
    source_path = tmp_path / "source.json"
    monkeypatch.setattr(app, "dark_spectrum_path", str(dark_path), raising=True)
    monkeypatch.setattr(app, "source_spectrum_path", str(source_path), raising=True)

    assert app.calibration_io._last_capture_text(str(dark_path), empty_text="") == ""
    assert app.calibration_io._last_capture_text(str(source_path)) == "Nicht aufgenommen"


def test_calibration_io_save_and_load_roundtrip(tmp_path, monkeypatch):
    dark_path = tmp_path / "dark.json"
    source_path = tmp_path / "source.json"
    monkeypatch.setattr(app, "dark_spectrum_path", str(dark_path), raising=True)
    monkeypatch.setattr(app, "source_spectrum_path", str(source_path), raising=True)
    monkeypatch.setattr(app, "dark_spectrum", None, raising=True)
    monkeypatch.setattr(app, "source_spectrum", None, raising=True)

    app.calibration_io.save_dark([1.11119, 2.22229])
    app.calibration_io.save_source([3.33339, 4.44449])
    app.calibration_io.load_all()

    assert app.dark_spectrum == [1.1112, 2.2223]
    assert app.source_spectrum == [3.3334, 4.4445]
    assert len(app.calibration_io.dark_status()) == 19


def test_live_absolute_does_not_recalibrate_compare(monkeypatch):
    n = len(getattr(app, "Wellenlängen")) - 1
    compare_spec = [0.0] + [2.0] * n

    monkeypatch.setattr(app, "values", {"-VERGLEICH_LIST-": ["cmp"]}, raising=True)
    monkeypatch.setattr(app.io, "get_compare_spektrum", lambda _name: compare_spec)
    monkeypatch.setattr(app.io, "get_compare_color", lambda _name: "#00ff00")
    monkeypatch.setattr(app, "display_mode", "percent", raising=True)
    monkeypatch.setattr(app, "normalize_view", False, raising=True)
    monkeypatch.setattr(app, "dark_spectrum", [0.0] * n, raising=True)
    monkeypatch.setattr(app, "source_spectrum", [10.0] * n, raising=True)
    monkeypatch.setattr(app, "hold", False, raising=True)

    calls = {"n": 0}

    def _fake_apply(vals, dark=None, source=None):
        calls["n"] += 1
        return list(vals)

    monkeypatch.setattr(app.compute, "apply_calibration", _fake_apply)

    app.update_graphs.live_absolute(y_values=[1.0] * n)

    assert calls["n"] == 1


def test_live_relative_does_not_recalibrate_compare(monkeypatch):
    n = len(getattr(app, "Wellenlängen")) - 1
    compare_spec = [0.0] + [float(i) for i in range(n)]

    monkeypatch.setattr(app, "values", {"-VERGLEICH_LIST-": ["cmp"]}, raising=True)
    monkeypatch.setattr(app.io, "get_compare_spektrum", lambda _name: compare_spec)
    monkeypatch.setattr(app.io, "get_compare_color", lambda _name: "#00ff00")
    monkeypatch.setattr(app, "display_mode", "percent", raising=True)
    monkeypatch.setattr(app, "normalize_view", False, raising=True)
    monkeypatch.setattr(app, "dark_spectrum", [0.0] * n, raising=True)
    monkeypatch.setattr(app, "source_spectrum", [10.0] * n, raising=True)
    monkeypatch.setattr(app, "hold", False, raising=True)

    calls = {"n": 0}

    def _fake_apply(vals, dark=None, source=None):
        calls["n"] += 1
        return list(vals)

    monkeypatch.setattr(app.compute, "apply_calibration", _fake_apply)

    app.update_graphs.live_relative(y_values=[float(i) for i in range(n)])

    assert calls["n"] == 1
