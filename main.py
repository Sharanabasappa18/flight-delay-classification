import sys
import pandas as pd
from src.data_loader import load_flight_data, get_train_test_data
from src.preprocessing import fit_and_save_preprocessor, transform_data
from src.train import train_all_models
from src.evaluate import evaluate_models
from src.predict import predict_flight


def main():
    print("=" * 75)
    print("      FLIGHT DELAY CLASSIFICATION MACHINE LEARNING PIPELINE      ")
    print("     (Logistic Regression | Random Forest | Gradient Boosting)    ")
    print("=" * 75)

    # 1. Train and save models
    trained_models, preprocessor, X_test_trans, y_test, X_test_raw = train_all_models()

    # 2. Evaluate all models
    summary_df = evaluate_models(
        models=trained_models,
        X_test=X_test_trans,
        y_test=y_test,
        preprocessor=preprocessor
    )

    print("\n" + "=" * 75)
    print("FINAL MODEL PERFORMANCE COMPARISON SUMMARY:")
    print("=" * 75)
    print(summary_df.to_string(index=False))

    # 3. Demonstration inference on realistic scenarios
    print("\n" + "=" * 75)
    print("DEMONSTRATION: PREDICTING FLIGHT DELAYS ON TEST SCENARIOS")
    print("=" * 75)

    scenarios = [
        {
            "name": "Scenario A: Low Risk Flight (Clear morning flight, favorable weather)",
            "data": {
                "Month": 5,
                "DayOfWeek": 2,
                "DepHour": 8,
                "Airline": "Delta Air Lines",
                "Origin": "SEA",
                "Dest": "DEN",
                "Distance": 1024,
                "AirTime": 140,
                "TaxiOut": 12,
                "WeatherCondition": "Clear",
                "WindSpeed": 6.5,
                "Visibility": 10.0
            }
        },
        {
            "name": "Scenario B: High Risk Flight (Snowy evening flight at congested hub)",
            "data": {
                "Month": 12,
                "DayOfWeek": 5,
                "DepHour": 19,
                "Airline": "Spirit Airlines",
                "Origin": "ORD",
                "Dest": "LGA",
                "Distance": 733,
                "AirTime": 120,
                "TaxiOut": 35,
                "WeatherCondition": "Snow",
                "WindSpeed": 28.0,
                "Visibility": 1.5
            }
        }
    ]

    for sc in scenarios:
        print(f"\n>>> {sc['name']}:")
        for model_name in ["Logistic Regression", "Random Forest", "Gradient Boosting"]:
            res = predict_flight(sc["data"], model_name=model_name)
            pct = res["delay_probability"] * 100
            print(f"   [{model_name:19s}] -> Result: {res['delay_prediction']:19s} | Delay Prob: {pct:5.1f}% | {res['risk_level']}")
        if res["contributing_factors"]:
            print(f"   Key Risk Factors: {', '.join(res['contributing_factors'])}")

    print("\n" + "=" * 75)
    print("Pipeline completed successfully! All models, reports, and plots saved.")
    print("=" * 75)


if __name__ == "__main__":
    main()
