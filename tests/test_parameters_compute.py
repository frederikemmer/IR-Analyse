import os

os.environ.setdefault("IR_ANALYSE_TEST_MODE", "1")

import main as app


def test_get_parameters_wavelengths_shape_and_range():
    values = app.get_parameters("wavelengths")

    assert len(values) == app.Auflösung[2]
    assert abs(values[0] - app.Auflösung[0]) < 0.1
    assert values[-1] > values[0]
    assert all(values[i] <= values[i + 1] for i in range(len(values) - 1))


def test_get_parameters_wavenumbers_monotonic():
    values = app.get_parameters("wavenumbers")

    assert len(values) == app.Auflösung[2]
    assert all(values[i] <= values[i + 1] for i in range(len(values) - 1))


def test_compute_relative_values_default_range():
    spectrum = [10.0, 20.0, 30.0]

    rel = app.compute.relative_values(spectrum)

    assert rel == [0.0, 50.0, 100.0]


def test_compute_relative_values_custom_range():
    spectrum = [2.0, 6.0, 10.0]

    rel = app.compute.relative_values(spectrum, begin=-1, end=1)

    assert rel[0] == -1
    assert rel[1] == 0
    assert rel[2] == 1
