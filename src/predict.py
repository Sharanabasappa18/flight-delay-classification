import os
import joblib
import pandas as pd
from src.preprocessing import transform_data

MODELS_DIR = "models"
PREPROCESSOR_PATH = os.path.join(MODELS_DIR, "preprocessor.joblib")


def get_model_path(model_name: str) -> str:
    slug = model_name.lower().replace(" ", "_")
    return os.path.join(MODELS_DIR, f"{slug}.joblib")


def load_artifacts(model_name: str = "Random Forest"):
    """
    Loads specified trained model and the fitted preprocessor.
    """
    if not os.path.exists(PREPROCESSOR_PATH):
        raise FileNotFoundError(f"Preprocessor not found at '{PREPROCESSOR_PATH}'. Run training first.")

    model_path = get_model_path(model_name)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at '{model_path}'. Run training first.")

    preprocessor = joblib.load(PREPROCESSOR_PATH)
    model = joblib.load(model_path)
    return model, preprocessor


def predict_flight(flight_data: dict, model_name: str = "Random Forest") -> dict:
    """
    Predicts flight delay for a single flight dictionary.
    
    Expected flight_data keys:
        - Month (1-12)
        - DayOfWeek (1-7)
        - DepHour (0-23)
        - Airline (e.g. 'Delta Air Lines')
        - Origin (e.g. 'JFK')
        - Dest (e.g. 'LAX')
        - Distance (miles, e.g. 2475)
        - AirTime (minutes, e.g. 330)
        - TaxiOut (minutes, e.g. 20)
        - WeatherCondition ('Clear', 'Cloudy', 'Rain', 'Fog', 'Snow', 'Thunderstorm')
        - WindSpeed (mph, e.g. 12.0)
        - Visibility (miles, e.g. 10.0)
    """
    model, preprocessor = load_artifacts(model_name)
    df_input = pd.DataFrame([flight_data])

    X_trans = transform_data(preprocessor, df_input)

    pred_label = int(model.predict(X_trans)[0])
    
    if hasattr(model, "predict_proba"):
        delay_prob = float(model.predict_proba(X_trans)[0][1])
    else:
        # Decision function fallback for models without predict_proba
        decision = float(model.decision_function(X_trans)[0])
        delay_prob = 1.0 / (1.0 + (2.71828 ** (-decision)))

    # Determine risk category
    if delay_prob >= 0.65:
        risk_level = "High Delay Risk"
        status_color = "red"
    elif delay_prob >= 0.40:
        risk_level = "Moderate Delay Risk"
        status_color = "orange"
    else:
        risk_level = "Low Delay Risk (Likely On-Time)"
        status_color = "green"

    # Identify primary risk contributors
    reasons = []
    if flight_data.get("WeatherCondition") in ["Snow", "Thunderstorm", "Rain"]:
        reasons.append(f"Adverse weather condition ({flight_data.get('WeatherCondition')})")
    if flight_data.get("DepHour", 0) >= 16 and flight_data.get("DepHour", 0) <= 21:
        reasons.append("Peak evening departure rush (higher cascading delays)")
    if flight_data.get("Origin") in ["ORD", "EWR", "JFK", "LGA", "ATL", "SFO"]:
        reasons.append(f"Origin is a high-congestion hub ({flight_data.get('Origin')})")
    if flight_data.get("TaxiOut", 0) > 22:
        reasons.append(f"Extended ground taxi-out time ({flight_data.get('TaxiOut')} mins)")
    if flight_data.get("WindSpeed", 0) > 20:
        reasons.append(f"Strong surface winds ({flight_data.get('WindSpeed')} mph)")

    return {
        "model_used": model_name,
        "is_delayed": pred_label,
        "delay_prediction": "Delayed (>= 15 mins)" if pred_label == 1 else "On-Time",
        "delay_probability": delay_prob,
        "risk_level": risk_level,
        "status_color": status_color,
        "contributing_factors": reasons
    }


if __name__ == "__main__":
    sample_flight = {
        "Month": 12,
        "DayOfWeek": 5,
        "DepHour": 18,
        "Airline": "United Airlines",
        "Origin": "ORD",
        "Dest": "SFO",
        "Distance": 1846,
        "AirTime": 270,
        "TaxiOut": 32,
        "WeatherCondition": "Snow",
        "WindSpeed": 26.5,
        "Visibility": 2.0
    }
    print("Testing sample flight prediction:")
    try:
        res = predict_flight(sample_flight, "Random Forest")
        print(res)
    except Exception as e:
        print(f"Artifacts not yet built: {e}")
