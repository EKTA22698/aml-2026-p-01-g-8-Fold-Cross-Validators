import pandas as pd


REQUIRED_COLUMNS = [
    "instant", "dteday", "season", "yr", "mnth", "hr",
    "holiday", "weekday", "workingday", "weathersit",
    "temp", "atemp", "hum", "windspeed",
    "casual", "registered", "cnt"
]


def validate_basic_structure(df):
    """
    Validate the basic structure and completeness of the hourly
    bike-sharing dataset.
    """
    print("=== BASIC DATA VALIDATION ===")

    # 1. Dataset shape
    print(f"Dataset shape: {df.shape}")

    if df.shape[1] == 17:
        print("PASS: Dataset contains the expected 17 columns.")
    else:
        print(
            f"WARNING: Expected 17 columns, but found {df.shape[1]}."
        )

    # 2. Required columns
    missing_columns = [
        col for col in REQUIRED_COLUMNS if col not in df.columns
    ]

    if not missing_columns:
        print("PASS: All required columns are present.")
    else:
        print("WARNING: Missing required columns:", missing_columns)

    # 3. Missing values
    total_missing = df.isnull().sum().sum()

    if total_missing == 0:
        print("PASS: No missing/null values found.")
    else:
        print(f"WARNING: Found {total_missing} missing values.")
        print(df.isnull().sum()[df.isnull().sum() > 0])

    # 4. Duplicate rows
    duplicate_count = df.duplicated().sum()

    if duplicate_count == 0:
        print("PASS: No duplicate rows found.")
    else:
        print(f"WARNING: Found {duplicate_count} duplicate rows.")

def build_hourly_timestamp(df):
    """
    Create a true hourly timestamp by combining dteday and hr.

    A copy is returned so that the original DataFrame is not modified.
    """
    validated_df = df.copy()

    validated_df["timestamp"] = (
        pd.to_datetime(validated_df["dteday"])
        + pd.to_timedelta(validated_df["hr"], unit="h")
    )

    return validated_df    
  
def find_missing_hours(df):
    """
    Find hourly timestamps that are absent from the dataset.

    Returns
    -------
    pd.DatetimeIndex
        All expected hourly timestamps that are missing.
    """
    timestamped_df = build_hourly_timestamp(df)

    expected_hours = pd.date_range(
        start=timestamped_df["timestamp"].min(),
        end=timestamped_df["timestamp"].max(),
        freq="h"
    )

    observed_hours = pd.DatetimeIndex(
        timestamped_df["timestamp"]
    )

    missing_hours = expected_hours.difference(observed_hours)

    return missing_hours

def summarize_missing_hours(df):
    """
    Summarize missing hourly timestamps and their distribution.

    The function reports missing hours but does not impute them,
    because an absent timestamp does not provide reliable values
    for weather or other predictor variables.
    """
    missing_hours = find_missing_hours(df)

    print("\n=== MISSING-HOUR VALIDATION ===")
    print(f"Number of missing hours: {len(missing_hours)}")

    if len(missing_hours) == 0:
        print("PASS: No missing hourly timestamps found.")
        return missing_hours

    print("WARNING: Missing hourly timestamps were detected.")

    missing_series = pd.Series(missing_hours)

    print("\nMissing hours by hour of day:")
    print(
        missing_series.dt.hour
        .value_counts()
        .sort_index()
    )

    print("\nMissing hours by year:")
    print(
        missing_series.dt.year
        .value_counts()
        .sort_index()
    )

    print("\nMissing hours by month:")
    print(
        missing_series.dt.month
        .value_counts()
        .sort_index()
    )

    return missing_hours

def find_missing_hour_blocks(df):
    """
    Group consecutive missing timestamps into continuous blocks.

    Returns a DataFrame containing the start time, end time,
    and number of missing hours in each block.
    """
    missing_hours = find_missing_hours(df)

    if len(missing_hours) == 0:
        return pd.DataFrame(
            columns=["start", "end", "hours_missing"]
        )

    missing_series = pd.Series(missing_hours)

    block_id = (
        missing_series.diff() != pd.Timedelta(hours=1)
    ).cumsum()

    blocks = (
        missing_series.groupby(block_id)
        .agg(["first", "last", "size"])
        .reset_index(drop=True)
    )

    blocks.columns = [
        "start",
        "end",
        "hours_missing"
    ]

    return blocks.sort_values(
        "hours_missing",
        ascending=False
    ).reset_index(drop=True)

def validate_target_identity(df):
    """
    Verify that cnt equals casual + registered for every row.

    casual and registered must not be used as predictors when
    modelling cnt because they directly reconstruct the target.
    """
    required = {"cnt", "casual", "registered"}

    if not required.issubset(df.columns):
        missing = sorted(required - set(df.columns))
        print("\n=== TARGET LEAKAGE VALIDATION ===")
        print("WARNING: Cannot perform target identity check.")
        print("Missing columns:", missing)
        return False

    identity_holds = (
        df["cnt"] == df["casual"] + df["registered"]
    )

    violations = (~identity_holds).sum()

    print("\n=== TARGET LEAKAGE VALIDATION ===")

    if violations == 0:
        print(
            "PASS: cnt equals casual + registered "
            "for every row."
        )
        print(
            "WARNING: casual and registered are target-leakage "
            "features and must be excluded when predicting cnt."
        )
        return True

    print(
        f"WARNING: Found {violations} rows where "
        "cnt != casual + registered."
    )

    return False

def validate_value_ranges(df):
    """
    Check whether important variables contain values within
    the ranges/categories defined for the Bike Sharing dataset.
    """
    print("\n=== VALUE-RANGE VALIDATION ===")

    allowed_values = {
        "season": {1, 2, 3, 4},
        "yr": {0, 1},
        "mnth": set(range(1, 13)),
        "hr": set(range(24)),
        "holiday": {0, 1},
        "weekday": set(range(7)),
        "workingday": {0, 1},
        "weathersit": {1, 2, 3, 4},
    }

    all_valid = True

    for column, allowed in allowed_values.items():
        if column not in df.columns:
            print(f"WARNING: Column '{column}' is missing.")
            all_valid = False
            continue

        invalid_values = set(df[column].dropna().unique()) - allowed

        if invalid_values:
            print(
                f"WARNING: {column} contains invalid values: "
                f"{sorted(invalid_values)}"
            )
            all_valid = False
        else:
            print(f"PASS: {column} contains valid values.")

    # Rental counts cannot be negative.
    count_columns = ["casual", "registered", "cnt"]

    for column in count_columns:
        if column in df.columns:
            negative_count = (df[column] < 0).sum()

            if negative_count == 0:
                print(f"PASS: {column} contains no negative values.")
            else:
                print(
                    f"WARNING: {column} contains "
                    f"{negative_count} negative values."
                )
                all_valid = False

    return all_valid

def validate_normalized_weather(df):
    """
    Inspect the weather variables that are already normalized
    in the UCI Bike Sharing dataset.

    These values must not be interpreted directly as physical
    units such as degrees Celsius or raw wind speed.
    """
    print("\n=== NORMALIZED WEATHER VALIDATION ===")

    weather_columns = [
        "temp",
        "atemp",
        "hum",
        "windspeed"
    ]

    all_valid = True

    for column in weather_columns:
        if column not in df.columns:
            print(f"WARNING: Column '{column}' is missing.")
            all_valid = False
            continue

        minimum = df[column].min()
        maximum = df[column].max()

        print(
            f"{column}: min={minimum:.4f}, "
            f"max={maximum:.4f}"
        )

        if minimum < 0 or maximum > 1:
            print(
                f"WARNING: {column} contains values "
                "outside the expected normalized range [0, 1]."
            )
            all_valid = False
        else:
            print(
                f"PASS: {column} lies within the "
                "normalized range [0, 1]."
            )

    print(
        "NOTE: These weather variables are already normalized "
        "and must not be interpreted directly as physical units."
    )

    return all_valid

def run_all_validations(df):
    """
    Run the complete set of validation checks for the raw
    hourly Bike Sharing dataset.

    This function reports data-quality issues but does not
    modify or impute the supplied DataFrame.
    """
    print("=" * 60)
    print("UCI BIKE SHARING - DATA VALIDATION REPORT")
    print("=" * 60)

    validate_basic_structure(df)

    summarize_missing_hours(df)

    blocks = find_missing_hour_blocks(df)

    print("\n=== CONSECUTIVE MISSING-HOUR BLOCKS ===")

    if len(blocks) == 0:
        print("PASS: No missing-hour blocks found.")
    else:
        print(f"Number of missing blocks: {len(blocks)}")
        print("\nLargest missing-hour blocks:")
        print(blocks.head(10).to_string(index=False))

    validate_target_identity(df)

    validate_value_ranges(df)

    validate_normalized_weather(df)

    print("\n=== MISSING-HOUR HANDLING DECISION ===")
    print(
        "Missing timestamps are reported but are not automatically "
        "inserted into the modelling dataset."
    )
    print(
        "Reason: some gaps form long consecutive blocks, and absent "
        "rows do not contain reliable weather/predictor values. "
        "Assigning zero demand and fabricated predictor values would "
        "introduce unsupported observations."
    )

    print("\n" + "=" * 60)
    print("VALIDATION COMPLETE")
    print("=" * 60)

    return {
        "missing_hours": find_missing_hours(df),
        "missing_blocks": blocks,
    }