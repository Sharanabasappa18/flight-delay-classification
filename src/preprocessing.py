import os
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

CONGESTED_HUBS = {"ORD", "EWR", "JFK", "LGA", "ATL", "SFO"}


def add_engineered_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Applies domain-specific feature engineering to raw flight data.
    """
    df = data.copy()

    # Time slot categorization
    def categorize_hour(hour):
        if 5 <= hour < 12:
            return "Morning"
        elif 12 <= hour < 17:
            return "Afternoon"
        elif 17 <= hour < 22:
            return "Evening"
        else:
            return "Night"

    df["DepTimeSlot"] = df["DepHour"].apply(categorize_hour)

    # Weekend flag (Day 6=Saturday, 7=Sunday)
    df["IsWeekend"] = df["DayOfWeek"].isin([6, 7]).astype(int)

    # Congested hub flag
    df["CongestedOrigin"] = df["Origin"].isin(CONGESTED_HUBS).astype(int)

    # Estimated flight speed (mph)
    # Safe division to prevent zero-division
    df["EstSpeed"] = np.where(
        df["AirTime"] > 0,
        df["Distance"] / (df["AirTime"] / 60.0),
        450.0
    )
    df["EstSpeed"] = np.clip(df["EstSpeed"], 200.0, 650.0)

    return df


def build_preprocessor(numeric_features, categorical_features) -> ColumnTransformer:
    """
    Creates a scikit-learn ColumnTransformer for numerical and categorical pipelines.
    """
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ],
        remainder="drop"
    )
    return preprocessor


def get_feature_columns():
    """
    Defines the feature subsets for modeling.
    """
    numeric_features = [
        "Month",
        "DayOfWeek",
        "DepHour",
        "Distance",
        "AirTime",
        "TaxiOut",
        "WindSpeed",
        "Visibility",
        "IsWeekend",
        "CongestedOrigin",
        "EstSpeed"
    ]

    categorical_features = [
        "Airline",
        "Origin",
        "Dest",
        "WeatherCondition",
        "DepTimeSlot"
    ]

    return numeric_features, categorical_features


def fit_and_save_preprocessor(
    X_train: pd.DataFrame,
    save_path: str = "models/preprocessor.joblib"
) -> ColumnTransformer:
    """
    Engineers features, fits the preprocessor, and saves it to disk.
    """
    X_fe = add_engineered_features(X_train)
    num_cols, cat_cols = get_feature_columns()

    preprocessor = build_preprocessor(num_cols, cat_cols)
    preprocessor.fit(X_fe)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    joblib.dump(preprocessor, save_path)
    print(f"Preprocessor successfully fitted and saved to '{save_path}'.")
    return preprocessor


def load_preprocessor(save_path: str = "models/preprocessor.joblib") -> ColumnTransformer:
    """
    Loads persisted preprocessor from disk.
    """
    if not os.path.exists(save_path):
        raise FileNotFoundError(f"Preprocessor file '{save_path}' does not exist.")
    return joblib.load(save_path)


def transform_data(preprocessor: ColumnTransformer, X: pd.DataFrame) -> np.ndarray:
    """
    Applies feature engineering and fitted preprocessor transformation.
    """
    X_fe = add_engineered_features(X)
    return preprocessor.transform(X_fe)


def get_feature_names_out(preprocessor: ColumnTransformer) -> list:
    """
    Retrieves transformed feature names from ColumnTransformer.
    """
    try:
        return list(preprocessor.get_feature_names_out())
    except Exception:
        num_cols, cat_cols = get_feature_columns()
        cat_encoder = preprocessor.named_transformers_["cat"].named_steps["onehot"]
        cat_names = cat_encoder.get_feature_names_out(cat_cols)
        return list(num_cols) + list(cat_names)
