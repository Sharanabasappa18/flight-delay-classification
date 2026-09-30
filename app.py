import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src.data_loader import AIRLINES, AIRPORTS, WEATHER_CONDITIONS, load_flight_data
from src.predict import predict_flight, load_artifacts, MODELS_DIR

st.set_page_config(
    page_title="Flight Delay Classification System",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.3rem;
        font-weight: 700;
        color: #1e3a8a;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.1rem;
        color: #4b5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
    }
    .badge-ontime {
        background-color: #dcfce7;
        color: #166534;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: bold;
        display: inline-block;
    }
    .badge-delayed {
        background-color: #fee2e2;
        color: #991b1b;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: bold;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)


def check_models_exist():
    required = ["logistic_regression.joblib", "random_forest.joblib", "gradient_boosting.joblib", "preprocessor.joblib"]
    return all(os.path.exists(os.path.join(MODELS_DIR, f)) for f in required)


# Header
st.markdown('<div class="main-title">✈️ Flight Delay Classification & Risk System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Predict flight delays using <b>Logistic Regression</b>, <b>Random Forest</b>, and <b>Gradient Boosting</b>. FAA standard: delay ≥ 15 mins.</div>', unsafe_allow_html=True)

if not check_models_exist():
    st.warning("⚠️ Trained models were not found. Please click the button below to train models or run `python main.py` in your terminal.")
    if st.button("🚀 Train Models Now", type="primary"):
        with st.spinner("Training Logistic Regression, Random Forest, and Gradient Boosting... this may take ~20 seconds."):
            from src.train import train_all_models
            from src.evaluate import evaluate_models
            trained_models, preprocessor, X_test_trans, y_test, _ = train_all_models()
            evaluate_models(trained_models, X_test_trans, y_test, preprocessor)
            st.success("All models trained and persisted successfully!")
            st.rerun()
    st.stop()

tabs = st.tabs(["🔮 Flight Delay Predictor", "📊 Model Comparison & Metrics", "📈 Operational Data Insights"])

# ==========================================
# TAB 1: INTERACTIVE PREDICTOR
# ==========================================
with tabs[0]:
    st.subheader("Configure Flight Parameters")
    
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("#### 🛫 Flight & Route")
        airline = st.selectbox("Airline Carrier", AIRLINES, index=0)
        origin = st.selectbox("Origin Airport", AIRPORTS, index=1)  # Default ORD
        dest_options = [a for a in AIRPORTS if a != origin]
        dest = st.selectbox("Destination Airport", dest_options, index=0)
        distance = st.number_input("Flight Distance (miles)", min_value=150, max_value=3000, value=733, step=25)
        air_time = st.number_input("Estimated Air Time (minutes)", min_value=30, max_value=450, value=120, step=10)

    with col2:
        st.markdown("#### 🕒 Schedule & Timing")
        dep_hour = st.slider("Scheduled Departure Hour (24h)", min_value=0, max_value=23, value=18, format="%d:00")
        month_names = {
            1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
            7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"
        }
        month = st.selectbox("Month of Travel", options=list(month_names.keys()), format_func=lambda x: f"{x} - {month_names[x]}", index=11)
        day_names = {1: "Monday", 2: "Tuesday", 3: "Wednesday", 4: "Thursday", 5: "Friday", 6: "Saturday", 7: "Sunday"}
        day_of_week = st.selectbox("Day of Week", options=list(day_names.keys()), format_func=lambda x: day_names[x], index=4)
        taxi_out = st.slider("Expected Taxi-Out Time (mins)", min_value=5, max_value=60, value=28)

    with col3:
        st.markdown("#### ⛅ Weather Conditions")
        weather = st.selectbox("Weather Condition at Origin", WEATHER_CONDITIONS, index=4)  # Snow
        wind_speed = st.slider("Surface Wind Speed (mph)", min_value=0.0, max_value=50.0, value=24.0, step=1.0)
        visibility = st.slider("Visibility (miles)", min_value=0.5, max_value=10.0, value=2.0, step=0.5)

    st.markdown("---")
    
    st.markdown("#### 🤖 Select Machine Learning Model")
    m_col1, m_col2 = st.columns([2, 1])
    with m_col1:
        model_choice = st.radio(
            "Classification Algorithm:",
            ["Compare All Models", "Gradient Boosting", "Random Forest", "Logistic Regression"],
            horizontal=True
        )
    with m_col2:
        predict_btn = st.button("✈️ Predict Delay Probability", type="primary", use_container_width=True)

    flight_payload = {
        "Month": month,
        "DayOfWeek": day_of_week,
        "DepHour": dep_hour,
        "Airline": airline,
        "Origin": origin,
        "Dest": dest,
        "Distance": distance,
        "AirTime": air_time,
        "TaxiOut": taxi_out,
        "WeatherCondition": weather,
        "WindSpeed": wind_speed,
        "Visibility": visibility
    }

    if predict_btn or True:
        st.markdown("### Prediction Results")

        if model_choice == "Compare All Models":
            res_cols = st.columns(3)
            models_to_test = ["Logistic Regression", "Random Forest", "Gradient Boosting"]

            for idx, m_name in enumerate(models_to_test):
                result = predict_flight(flight_payload, model_name=m_name)
                prob_pct = result["delay_probability"] * 100
                is_del = result["is_delayed"] == 1

                with res_cols[idx]:
                    st.markdown(f"#### {m_name}")
                    badge = f'<div class="badge-delayed">⚠️ DELAY RISK (≥15m)</div>' if is_del else f'<div class="badge-ontime">✅ ON-TIME</div>'
                    st.markdown(badge, unsafe_allow_html=True)
                    st.metric(label="Delay Probability", value=f"{prob_pct:.1f}%")
                    st.progress(float(result["delay_probability"]))
                    st.caption(f"Risk Assessment: **{result['risk_level']}**")

            # Risk factors breakdown
            ref_result = predict_flight(flight_payload, model_name="Gradient Boosting")
            if ref_result["contributing_factors"]:
                st.info("🔍 **Key Delay Risk Factors Identified:**\n\n" + "\n".join([f"- {f}" for f in ref_result["contributing_factors"]]))
            else:
                st.success("🌟 Favorable operational conditions detected with minimal risk indicators.")

        else:
            result = predict_flight(flight_payload, model_name=model_choice)
            prob_pct = result["delay_probability"] * 100
            is_del = result["is_delayed"] == 1

            rcol1, rcol2, rcol3 = st.columns([1, 1, 1.5])
            with rcol1:
                badge = f'<div class="badge-delayed">⚠️ DELAYED (≥15 mins)</div>' if is_del else f'<div class="badge-ontime">✅ ON-TIME EXPECTED</div>'
                st.markdown(badge, unsafe_allow_html=True)
                st.metric(label="Predicted Outcome", value="Delayed" if is_del else "On-Time")

            with rcol2:
                st.metric(label="Delay Probability", value=f"{prob_pct:.1f}%")
                st.progress(float(result["delay_probability"]))

            with rcol3:
                st.markdown(f"**Classification Model:** `{model_choice}`")
                st.markdown(f"**Risk Level:** `{result['risk_level']}`")

            if result["contributing_factors"]:
                st.warning("⚠️ **Contributing Risk Factors:**\n\n" + "\n".join([f"- {f}" for f in result["contributing_factors"]]))
            else:
                st.success("✅ Flight parameters indicate stable on-time operations.")


# ==========================================
# TAB 2: MODEL COMPARISON & METRICS
# ==========================================
with tabs[1]:
    st.subheader("Model Performance Comparison")
    metrics_path = "reports/metrics_summary.csv"

    if os.path.exists(metrics_path):
        metrics_df = pd.read_csv(metrics_path)
        
        # Display styled table
        st.dataframe(
            metrics_df.style.highlight_max(axis=0, subset=["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"], color="#bbf7d0"),
            use_container_width=True
        )

        st.markdown("---")
        st.subheader("Comparative Visualizations")

        img_col1, img_col2 = st.columns(2)
        with img_col1:
            roc_path = "reports/figures/roc_curves.png"
            if os.path.exists(roc_path):
                st.image(roc_path, caption="ROC Curves & AUC Benchmark", use_container_width=True)
        with img_col2:
            comp_path = "reports/figures/metrics_comparison.png"
            if os.path.exists(comp_path):
                st.image(comp_path, caption="Key Classification Metrics Comparison", use_container_width=True)

        cm_path = "reports/figures/confusion_matrices.png"
        if os.path.exists(cm_path):
            st.image(cm_path, caption="Confusion Matrices Across All Models", use_container_width=True)

        feat_path = "reports/figures/feature_importance.png"
        if os.path.exists(feat_path):
            st.image(feat_path, caption="Feature Importance Ranking (Random Forest & Gradient Boosting)", use_container_width=True)
    else:
        st.info("Metrics summary not found yet. Run training to generate metrics.")


# ==========================================
# TAB 3: OPERATIONAL DATA INSIGHTS
# ==========================================
with tabs[2]:
    st.subheader("Exploratory Data Analysis: Key Delay Drivers")
    raw_path = "data/raw/flights.csv"

    if os.path.exists(raw_path):
        df = pd.read_csv(raw_path)
        
        c1, c2 = st.columns(2)
        with c1:
            fig, ax = plt.subplots(figsize=(8, 4))
            hour_delays = df.groupby("DepHour")["IsDelayed"].mean() * 100
            sns.barplot(x=hour_delays.index, y=hour_delays.values, ax=ax, hue=hour_delays.index, palette="Blues_r", legend=False)
            ax.set_title("Delay Rate by Scheduled Departure Hour", fontsize=11, fontweight="bold")
            ax.set_ylabel("Delay Rate (%)")
            ax.set_xlabel("Hour of Day (0-23)")
            ax.grid(axis="y", linestyle=":", alpha=0.6)
            st.pyplot(fig)
            plt.close()

        with c2:
            fig, ax = plt.subplots(figsize=(8, 4))
            weather_delays = df.groupby("WeatherCondition")["IsDelayed"].mean().sort_values(ascending=False) * 100
            sns.barplot(x=weather_delays.values, y=weather_delays.index, ax=ax, hue=weather_delays.index, palette="Reds_r", legend=False)
            ax.set_title("Delay Rate by Weather Condition", fontsize=11, fontweight="bold")
            ax.set_xlabel("Delay Rate (%)")
            ax.grid(axis="x", linestyle=":", alpha=0.6)
            st.pyplot(fig)
            plt.close()

        c3, c4 = st.columns(2)
        with c3:
            fig, ax = plt.subplots(figsize=(8, 4))
            origin_delays = df.groupby("Origin")["IsDelayed"].mean().sort_values(ascending=False) * 100
            sns.barplot(x=origin_delays.index, y=origin_delays.values, ax=ax, hue=origin_delays.index, palette="mako", legend=False)
            ax.set_title("Delay Rate by Origin Hub", fontsize=11, fontweight="bold")
            ax.set_ylabel("Delay Rate (%)")
            ax.tick_params(axis="x", rotation=45)
            ax.grid(axis="y", linestyle=":", alpha=0.6)
            st.pyplot(fig)
            plt.close()

        with c4:
            fig, ax = plt.subplots(figsize=(8, 4))
            carrier_delays = df.groupby("Airline")["IsDelayed"].mean().sort_values(ascending=False) * 100
            sns.barplot(x=carrier_delays.values, y=carrier_delays.index, ax=ax, hue=carrier_delays.index, palette="viridis", legend=False)
            ax.set_title("Delay Rate by Airline Carrier", fontsize=11, fontweight="bold")
            ax.set_xlabel("Delay Rate (%)")
            ax.grid(axis="x", linestyle=":", alpha=0.6)
            st.pyplot(fig)
            plt.close()
    else:
        st.info("Run `python main.py` to create dataset and generate data insights.")
