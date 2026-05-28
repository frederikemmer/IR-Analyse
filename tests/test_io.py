import os
from pathlib import Path

os.environ.setdefault("IR_ANALYSE_TEST_MODE", "1")

import main as app


def test_import_data_and_trim_data(tmp_path):
    realtime_file = tmp_path / "realtime.txt"
    realtime_file.write_text("0\t1\t10\t20\n0\t1\t30\t40\n", encoding="utf-8")

    data = app.io.import_data(path=str(realtime_file))

    assert len(data) == 2
    assert data[0][:2] == [0.0, 1.0]
    assert data[1][2:] == [30.0, 40.0]

    app.io.trim_data(path=str(realtime_file))

    remaining = realtime_file.read_text(encoding="utf-8").splitlines()
    assert remaining == ["0\t1\t30\t40"]


def test_compare_file_helpers(tmp_path, monkeypatch):
    compare_file = tmp_path / "vergleich.txt"
    compare_file.write_text(
        "Name;Material;Farbe;S0;S1;S2\n"
        "A;PET;#ff0000;0;1;2\n"
        "B;PE;#00ff00;3;4;5\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(app, "compare_path", str(compare_file), raising=True)

    names = app.io.get_compare()
    spec_a = app.io.get_compare_spektrum("A")
    color_b = app.io.get_compare_color("B")

    assert names == ["A", "B"]
    assert spec_a == [0.0, 1.0, 2.0]
    assert color_b == "#00ff00"
