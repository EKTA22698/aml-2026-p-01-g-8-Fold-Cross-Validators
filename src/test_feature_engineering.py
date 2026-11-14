
import pandas as pd

from feature_engineering import (
    add_cyclic_features,
    prepare_features,
    build_preprocessor
)


def sample_data():
    return pd.DataFrame({
        "season": [1, 2, 3, 4],
        "yr": [0, 0, 1, 1],
        "mnth": [1, 4, 8, 12],
        "hr": [0, 6, 12, 23],
        "holiday": [0, 0, 1, 0],
        "weekday": [0, 2, 4, 6],
        "workingday": [1, 1, 0, 0],
        "weathersit": [1, 2, 3, 1],
        "temp": [0.2, 0.4, 0.6, 0.3],
        "atemp": [0.2, 0.4, 0.5, 0.3],
        "hum": [0.5, 0.6, 0.7, 0.4],
        "windspeed": [0.1, 0.2, 0.3, 0.2],
        "casual": [10, 20, 30, 40],
        "registered": [50, 60, 70, 80],
        "cnt": [60, 80, 100, 120]
    })


def test_cyclic_features():
    encoded = add_cyclic_features(sample_data())

    for column in [
        "hr_sin", "hr_cos",
        "mnth_sin", "mnth_cos",
        "weekday_sin", "weekday_cos"
    ]:
        assert column in encoded.columns


def test_target_leakage_prevention():
    X, y = prepare_features(sample_data())

    assert "cnt" not in X.columns
    assert "casual" not in X.columns
    assert "registered" not in X.columns
    assert len(y) == 4


def test_preprocessing_pipeline():
    X, y = prepare_features(sample_data())

    preprocessor = build_preprocessor(X)
    transformed = preprocessor.fit_transform(X)

    assert transformed.shape[0] == len(X)
    assert transformed.shape[1] > 0
