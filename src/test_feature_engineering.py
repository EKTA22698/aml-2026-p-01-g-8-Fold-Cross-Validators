import pandas as pd

from feature_engineering import (
    add_cyclic_features,
    prepare_features,
    build_preprocessor
)


def main():

    df = pd.DataFrame({
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

    encoded = add_cyclic_features(df)

    assert "hr_sin" in encoded.columns
    assert "hr_cos" in encoded.columns
    assert "mnth_sin" in encoded.columns
    assert "mnth_cos" in encoded.columns
    assert "weekday_sin" in encoded.columns
    assert "weekday_cos" in encoded.columns

    print("PASS: Cyclic encoding.")

   
    X, y = prepare_features(df)

    assert "cnt" not in X.columns
    assert "casual" not in X.columns
    assert "registered" not in X.columns

    print("PASS: Target leakage prevention.")


    preprocessor = build_preprocessor(X)

    transformed = preprocessor.fit_transform(X)

    assert transformed.shape[0] == len(df)

    print("PASS: Preprocessing pipeline.")
    print("Input shape:", X.shape)
    print("Output shape:", transformed.shape)

    print("\nAll tests completed successfully!")


if __name__ == "__main__":
    main()