import os
import pandas as pd
from sklearn.model_selection import train_test_split

DATA_PATH = "data/raw/flights.csv"

# Predefined reference categories for UI and model consistency
AIRLINES = [
    "Delta Air Lines",
    "American Airlines",
    "United Airlines",
    "Southwest Airlines",
    "JetBlue Airways",
    "Spirit Airlines",
    "SkyWest Airlines",
    "Alaska Airlines"
]

AIRPORTS = [
    "ATL", "ORD", "DFW", "DEN", "CLT",
    "LAX", "IAH", "PHX", "MCO", "SEA",
    "SFO", "BOS", "EWR", "LGA", "JFK"
]

WEATHER_CONDITIONS = ["Clear", "Cloudy", "Rain", "Fog", "Snow", "Thunderstorm"]


def load_flight_data(file_path: str = DATA_PATH) -> pd.DataFrame:
    """
    Imports and loads flight dataset directly from the CSV file.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Flight dataset not found at '{file_path}'. "
            f"Please ensure '{file_path}' exists in the data directory."
        )

    print(f"Importing flight dataset from '{file_path}'...")
    df = pd.read_csv(file_path)
    print(f"Loaded {len(df):,} flight records.")
    print(f"Delay rate: {df['IsDelayed'].mean() * 100:.2f}% ({df['IsDelayed'].sum():,} delayed flights)")
    return df


def get_train_test_data(
    file_path: str = DATA_PATH,
    test_size: float = 0.2,
    random_state: int = 42
):
    """
    Imports the CSV dataset and splits it into train and test sets.
    """
    df = load_flight_data(file_path=file_path)
    X = df.drop(columns=["IsDelayed"])
    y = df["IsDelayed"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    df = load_flight_data()
    print("\nDataset Preview:")
    print(df.head())
    print("\nSummary Statistics:")
    print(df.describe())
