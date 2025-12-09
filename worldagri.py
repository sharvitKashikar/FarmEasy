import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import plotly.express as px

st.set_page_config(page_title="🌾 Soil Nutrient Deficiency Detector", layout="wide")

# ------------------------------------------
# Load Sample Dataset (You can replace with your own)
# ------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("soil_data.csv")  # must have N, P, K, pH, organic_carbon, moisture
    df = df.dropna()
    return df

# ------------------------------------------
# Train Model
# ------------------------------------------
@st.cache_resource
def train_models(df):
    features = ["pH", "Organic_Carbon", "Moisture"]
    
    models = {}
    scalers = {}

    for nutrient in ["N", "P", "K"]:
        X = df[features]
        y = df[nutrient + "_level"]  # categorical: Low/Medium/High

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )

        model = RandomForestClassifier(n_estimators=250, random_state=42)
        model.fit(X_train, y_train)

        models[nutrient] = model
        scalers[nutrient] = scaler

    return models, scalers


# ------------------------------------------
# Generate Fertilizer Recommendations
# ------------------------------------------
def recommend_fertilizer(n_level, p_level, k_level):
    rec = []

    level_map = {
        "Low": "Apply high-dose fertilizer",
        "Medium": "Apply moderate fertilizer dose",
        "High": "No need to add fertilizer"
    }

    fertilizer_dose = {
        "N": {"Low": "50–70 kg/ha", "Medium": "25–40 kg/ha", "High": "0 kg/ha"},
        "P": {"Low": "40–60 kg/ha", "Medium": "20–30 kg/ha", "High": "0 kg/ha"},
        "K": {"Low": "35–55 kg/ha", "Medium": "20–25 kg/ha", "High": "0 kg/ha"},
    }

    rec.append(f"🌱 **Nitrogen (N):** {level_map[n_level]} — Recommended: **{fertilizer_dose['N'][n_level]}**")
    rec.append(f"🌾 **Phosphorus (P):** {level_map[p_level]} — Recommended: **{fertilizer_dose['P'][p_level]}**")
    rec.append(f"🪨 **Potassium (K):** {level_map[k_level]} — Recommended: **{fertilizer_dose['K'][k_level]}**")

    return rec


# ------------------------------------------
# MAIN APP
# ------------------------------------------
def main():
    st.markdown("<h1 style='text-align:center;'>🌾 Soil Nutrient Deficiency Detector</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center;'>AI-based N–P–K deficiency prediction & fertilizer recommendations</p>", unsafe_allow_html=True)

    df = load_data()
    models, scalers = train_models(df)

    st.sidebar.header("🧪 Enter Soil Test Values")
    ph = st.sidebar.number_input("Soil pH", 3.0, 10.0, 6.5)
    organic = st.sidebar.number_input("Organic Carbon (%)", 0.0, 5.0, 1.2)
    moisture = st.sidebar.number_input("Soil Moisture (%)", 1.0, 60.0, 25.0)

    if st.sidebar.button("🔍 Detect Deficiency"):
        inputs = np.array([[ph, organic, moisture]])

        predictions = {}
        for nutrient in ["N", "P", "K"]:
            scaled = scalers[nutrient].transform(inputs)
            pred = models[nutrient].predict(scaled)[0]
            predictions[nutrient] = pred

        n_level = predictions["N"]
        p_level = predictions["P"]
        k_level = predictions["K"]

        st.success("Prediction Complete")

        # Display Results
        col1, col2, col3 = st.columns(3)

        col1.metric("🌱 Nitrogen (N)", n_level)
        col2.metric("🌾 Phosphorus (P)", p_level)
        col3.metric("🪨 Potassium (K)", k_level)

        st.markdown("### 🧪 Nutrient Deficiency Analysis")
        risk_score = {"Low": 10, "Medium": 5, "High": 1}
        total_risk = risk_score[n_level] + risk_score[p_level] + risk_score[k_level]

        st.info(f"**Soil Health Score:** {total_risk} / 30")

        # Fertilizer recommendations
        st.markdown("### 🌱 Fertilizer Recommendations")
        recs = recommend_fertilizer(n_level, p_level, k_level)

        for r in recs:
            st.markdown(f"""
            <div style='background:#eef7ff;padding:10px;margin-bottom:10px;border-left:5px solid #1A73E8;border-radius:8px'>
            {r}
            </div>
            """, unsafe_allow_html=True)

        # Download report
        report = f"""
Soil Nutrient Deficiency Report
-------------------------------

Soil Inputs:
- pH: {ph}
- Organic Carbon: {organic}
- Moisture: {moisture}

Nutrient Levels:
- Nitrogen: {n_level}
- Phosphorus: {p_level}
- Potassium: {k_level}

Soil Health Score: {total_risk}/30

Fertilizer Recommendations:
- N: {recs[0]}
- P: {recs[1]}
- K: {recs[2]}
"""

        st.download_button(
            label="📥 Download Report",
            data=report,
            file_name="soil_nutrient_report.txt",
            mime="text/plain"
        )

    # Dataset visualization
    st.markdown("### 📊 Dataset Visualization")

    fig = px.scatter(df, x="pH", y="Organic_Carbon", color="N_level",
                     title="Soil Characteristics by Nitrogen Level")
    st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()
