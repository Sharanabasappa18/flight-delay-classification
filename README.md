# ✈️ Flight Delay Classification Project

An end-to-end Machine Learning system to predict flight delays based on operational, temporal, route, and meteorological factors.

This project implements, evaluates, and compares three core classification algorithms:
1. **Logistic Regression** (Linear baseline with balanced class weights)
2. **Random Forest Classifier** (Bagging ensemble with multi-tree voting)
3. **Gradient Boosting Classifier** (Sequential boosting ensemble minimizing pseudo-residual loss)

---

## 📌 Project Overview & Industry Standard

According to the **Federal Aviation Administration (FAA)** and the **U.S. Bureau of Transportation Statistics (BTS)**, a commercial flight is categorized as **delayed** if its actual departure or arrival time exceeds the scheduled time by **15 minutes or more** (`IsDelayed = 1`).

### Key Predictive Factors Modeled:
- **Temporal Cascading**: Cascading network delays accumulating throughout the day (morning flights vs. evening peak 16:00–21:00).
- **Weather Impedance**: Thunderstorms, snow, high wind speeds, and low visibility.
- **Airport Ground Congestion**: High-traffic hub airports (e.g., ORD, JFK, EWR, LGA, ATL, SFO) and taxi-out duration.
- **Carrier Operations**: Fleet turnaround times and airline-specific performance profiles.
- **Route Characteristics**: Distance, air time, and seasonal trends.

---

## 🗂️ Project Structure

```
Flight Delay Classification/
├── data/
│   ├── raw/
│   │   └── flights.csv              # Flight operational dataset (50,000 records)
│   └── processed/
├── models/
│   ├── logistic_regression.joblib   # Trained Logistic Regression model
│   ├── random_forest.joblib         # Trained Random Forest model
│   ├── gradient_boosting.joblib     # Trained Gradient Boosting model
│   └── preprocessor.joblib          # Scikit-learn ColumnTransformer pipeline
├── reports/
│   ├── figures/
│   │   ├── roc_curves.png           # Comparative ROC curves with AUC annotations
│   │   ├── confusion_matrices.png   # 3-panel confusion matrix comparison
│   │   ├── metrics_comparison.png   # Bar chart comparing Accuracy, Precision, Recall, F1, ROC-AUC
│   │   └── feature_importance.png   # Top 15 feature importances (RF & GB)
│   └── metrics_summary.csv          # Exported metrics comparison table
├── src/
│   ├── __init__.py
│   ├── data_loader.py               # Dataset loading & realistic benchmark data generator
│   ├── preprocessing.py             # Feature engineering & ColumnTransformer pipeline
│   ├── train.py                     # Model training & model persistence
│   ├── evaluate.py                  # Metric calculation & figure plotting
│   └── predict.py                   # Inference module for new flight queries
├── app.py                           # Interactive Streamlit Web UI dashboard
├── main.py                          # Full pipeline orchestration script
└── requirements.txt                 # Project dependencies
```

---

## ⚙️ Virtual Environment & Setup

A virtual environment (`venv`) is configured and installed with all dependencies.

### Activate Virtual Environment:
```powershell
# Windows PowerShell
.\venv\Scripts\Activate.ps1
```

*(Or use `.\venv\Scripts\python.exe` directly)*

---

## 🚀 How to Run

### Option 1: Run the End-to-End Pipeline
Imports `data/raw/flights.csv`, transforms features, trains all 3 models, evaluates performance, and generates visualization plots:
```powershell
.\venv\Scripts\python.exe main.py
```

### Option 2: Launch the Interactive Streamlit Web Dashboard
Launch an interactive UI to test flight delay predictions, view model comparisons, and explore dataset insights:
```powershell
.\venv\Scripts\streamlit.exe run app.py
```

---

## 🤖 Algorithms Implemented & Technical Details

| Algorithm | Strengths | Role in this Project |
| :--- | :--- | :--- |
| **Logistic Regression** | Interpretable linear model, fast execution, calibrated probabilistic output. | Acts as the foundational linear benchmark with class balancing. |
| **Random Forest** | Non-parametric bagging ensemble, robust against outliers, captures complex non-linear feature interactions without overfitting. | Provides strong classification accuracy and transparent Gini feature importance rankings. |
| **Gradient Boosting** | Sequential additive ensemble, iteratively optimizes decision trees on pseudo-residuals. | Delivers optimal discrimination power and highest ROC-AUC scores. |

---

## 📊 Preprocessing & Feature Engineering Pipeline

The preprocessing workflow uses `scikit-learn`'s `ColumnTransformer` to ensure reproducibility and prevent data leakage:

1. **Feature Engineering**:
   - `DepTimeSlot`: Categorizes departure hours into `['Morning', 'Afternoon', 'Evening', 'Night']`.
   - `IsWeekend`: Binary indicator for Saturday/Sunday travel.
   - `CongestedOrigin`: Flags flights originating from congested airspace hubs (`ORD`, `EWR`, `JFK`, `LGA`, `ATL`, `SFO`).
   - `EstSpeed`: Calculated cruising speed (`Distance / AirTime * 60`).
2. **Numerical Pipeline**:
   - Median imputation + `StandardScaler` normalization.
3. **Categorical Pipeline**:
   - Mode imputation + `OneHotEncoder(handle_unknown='ignore')`.

---

## 🔍 Evaluation Metrics Explained

- **Accuracy**: Overall proportion of correct predictions.
- **Precision**: Proportion of predicted delays that were truly delayed (avoids false alarms).
- **Recall (Sensitivity)**: Proportion of actual delays successfully flagged by the model (critical for passenger & airline planning).
- **F1-Score**: Harmonic mean of Precision and Recall.
- **ROC-AUC Score**: Area under the Receiver Operating Characteristic curve; measures the model's ability to discriminate between delayed and on-time flights across all probability thresholds.

---

## 💡 Using the Prediction API

You can use the prediction module in your own scripts:

```python
from src.predict import predict_flight

sample_flight = {
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

result = predict_flight(sample_flight, model_name="Gradient Boosting")
print(result)
# Output:
# {
#   'model_used': 'Gradient Boosting',
#   'is_delayed': 1,
#   'delay_prediction': 'Delayed (>= 15 mins)',
#   'delay_probability': 0.88,
#   'risk_level': 'High Delay Risk',
#   'contributing_factors': ['Adverse weather condition (Snow)', 'Peak evening departure rush', ...]
# }
```
