
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer



TARGET_COLUMN = "cnt"

LEAKAGE_COLUMNS = [
    "casual",
    "registered"
]

CYCLIC_PERIODS = {
    "hr": 24,
    "mnth": 12,
    "weekday": 7
}

CATEGORICAL_COLUMNS = [
    "season",
    "holiday",
    "workingday",
    "weathersit"
]

NUMERIC_COLUMNS = [
    "yr",
    "temp",
    "atemp",
    "hum",
    "windspeed"
]



def add_cyclic_features(df):
    """
    Encode cyclical features using sine and cosine.

    Hour: 24-hour cycle
    Month: 12-month cycle
    Weekday: 7-day cycle

    The original DataFrame is not modified.
    """

    result = df.copy()

    for column, period in CYCLIC_PERIODS.items():

        if column not in result.columns:
            continue

        values = result[column].astype(float)

        result[f"{column}_sin"] = np.sin(
            2 * np.pi * values / period
        )

        result[f"{column}_cos"] = np.cos(
            2 * np.pi * values / period
        )

        result.drop(columns=[column], inplace=True)

    return result


def get_feature_groups(df):
    """
    Identify cyclic, categorical, and numeric features
    available in the supplied DataFrame.
    """

    cyclic = [
        col for col in CYCLIC_PERIODS
        if col in df.columns
    ]

    categorical = [
        col for col in CATEGORICAL_COLUMNS
        if col in df.columns
    ]

    numeric = [
        col for col in NUMERIC_COLUMNS
        if col in df.columns
    ]

    return {
        "cyclic": cyclic,
        "categorical": categorical,
        "numeric": numeric
    }



def prepare_features(df):
    """
    Separate predictors from the target.

    The target column and leakage columns are excluded
    from the predictor DataFrame.
    """

    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' not found."
        )

    excluded = [
        TARGET_COLUMN,
        *LEAKAGE_COLUMNS,
        "dteday",
        "instant"
    ]

    X = df.drop(
        columns=excluded,
        errors="ignore"
    ).copy()

    y = df[TARGET_COLUMN].copy()

    if X.empty:
        raise ValueError(
            "No predictor columns are available."
        )

    return X, y


def build_preprocessor(X):
    """
    Build a reusable preprocessing pipeline.

    Cyclic features:
        Sine/cosine transformation.

    Categorical features:
        Most-frequent imputation and one-hot encoding.

    Numeric features:
        Median imputation and standard scaling.

    Fit the returned pipeline on training data only.
    """

    groups = get_feature_groups(X)

    cyclic_columns = groups["cyclic"]
    categorical_columns = groups["categorical"]
    numeric_columns = groups["numeric"]

  
    assigned_columns = (
        cyclic_columns
        + categorical_columns
        + numeric_columns
    )

    extra_columns = [
        col for col in X.columns
        if col not in assigned_columns
    ]

    for column in extra_columns:

        if pd.api.types.is_numeric_dtype(X[column]):
            numeric_columns.append(column)

        else:
            categorical_columns.append(column)

    numeric_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "scaler",
            StandardScaler()
        )
    ])

    categorical_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "encoder",
            OneHotEncoder(handle_unknown="ignore")
        )
    ])

    transformers = []

    if cyclic_columns:

        transformers.append((
            "cyclic",
            Pipeline([
                (
                    "encoding",
                    CyclicEncoderTransformer()
                ),
                (
                    "imputer",
                    SimpleImputer(strategy="median")
                )
            ]),
            cyclic_columns
        ))

    if categorical_columns:

        transformers.append((
            "categorical",
            categorical_pipeline,
            categorical_columns
        ))

    if numeric_columns:

        transformers.append((
            "numeric",
            numeric_pipeline,
            numeric_columns
        ))

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop"
    )

    return preprocessor



from sklearn.base import BaseEstimator, TransformerMixin

class CyclicEncoderTransformer(
    BaseEstimator,
    TransformerMixin
):
    """Encode cyclical variables using sine and cosine."""

    def __init__(self):
        self.periods = {
            "hr": 24,
            "mnth": 12,
            "weekday": 7
        }

    def fit(self, X, y=None):
        self.n_features_in_ = X.shape[1]

        if isinstance(X, pd.DataFrame):
            self.columns_ = list(X.columns)
        else:
            self.columns_ = list(self.periods.keys())[
                :self.n_features_in_
            ]

        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            data = X
        else:
            data = pd.DataFrame(
                X,
                columns=self.columns_
            )

        result = []

        for column in self.columns_:
            period = self.periods[column]
            values = data[column].to_numpy(dtype=float)

            result.append(
                np.sin(2 * np.pi * values / period)
            )

            result.append(
                np.cos(2 * np.pi * values / period)
            )

        return np.column_stack(result)

    def get_feature_names_out(self, input_features=None):
        columns = (
            self.columns_
            if input_features is None
            else input_features
        )

        names = []

        for column in columns:
            names.extend([
                f"{column}_sin",
                f"{column}_cos"
            ])

        return np.asarray(names, dtype=object)


def engineer_features(X):
    """
    Apply cyclic encoding directly to a DataFrame.

    This function is useful for inspecting the generated
    features independently of model preprocessing.
    """

    return add_cyclic_features(X)