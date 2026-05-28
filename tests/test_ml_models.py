import os
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("IR_ANALYSE_TEST_MODE", "1")

import main as app


def test_prep_data_shapes():
    x_train, x_test, y_train, y_test = app.ML.prep_data(test_size=0.25, random_state=42)

    assert len(x_train) > 0
    assert len(x_test) > 0
    assert len(y_train) == len(x_train)
    assert len(y_test) == len(x_test)


def test_prep_data_keeps_all_material_classes_in_both_splits():
    _, _, y_train, y_test = app.ML.prep_data(test_size=0.25, random_state=42)

    assert set(y_train.ravel()) == {1, 2, 3, 4, 5, 6, 7, 8}
    assert set(y_test.ravel()) == {1, 2, 3, 4, 5, 6, 7, 8}


def test_load_model_pred_with_existing_model():
    db = pd.read_csv("data/Sample-Database V1 (Sphere).csv", sep=";")
    sample = db.drop(columns=["#", "Material"]).iloc[0].to_list()

    model_path = Path("models/neural_5000.model")
    assert model_path.exists()

    pred = app.ML.load_model_pred(sample, str(model_path))

    assert len(pred) == 1


def test_prediction_to_key_handles_numpy_and_scalar_values():
    assert app.ML.prediction_to_key(np.array([2.0])) == "2"
    assert app.ML.prediction_to_key(3) == "3"


def test_prepare_prediction_input_calibrates_spectrometer_data_back_to_training_scale():
    db = pd.read_csv("data/Sample-Database V1 (Sphere).csv", sep=";")
    sample = db.drop(columns=["#", "Material"]).iloc[0].to_numpy(dtype=float)

    dark = np.asarray(app.calibration_io.load_dark(), dtype=float)
    source = np.asarray(app.calibration_io.load_source(), dtype=float)
    raw = dark + (sample / 100.0) * (source - dark)

    prepared = np.asarray(
        app.ML.prepare_prediction_input(raw.tolist(), source_mode="spectrometer", dark=dark.tolist(), source=source.tolist()),
        dtype=float,
    )

    np.testing.assert_allclose(prepared, sample, atol=1e-6)


def test_existing_model_predicts_same_class_after_spectrometer_calibration():
    db = pd.read_csv("data/Sample-Database V1 (Sphere).csv", sep=";")
    sample_row = db.iloc[0]
    sample = sample_row.drop(labels=["#", "Material"]).to_numpy(dtype=float)

    dark = np.asarray(app.calibration_io.load_dark(), dtype=float)
    source = np.asarray(app.calibration_io.load_source(), dtype=float)
    raw = dark + (sample / 100.0) * (source - dark)

    prepared = app.ML.prepare_prediction_input(raw.tolist(), source_mode="spectrometer", dark=dark.tolist(), source=source.tolist())
    prediction = app.ML.prediction_to_key(app.ML.load_model_pred(prepared, "models/neural_5000.model"))

    assert prediction == str(int(sample_row["Material"]))
