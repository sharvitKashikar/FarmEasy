import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="🌾 Crop Recommendation System", layout="wide")

# ----------------------------
# Load Dataset
# ----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("crop_recommendation.csv")
    return df


# ----------------------------
# Train Model
# ----------------------------
@st.cache_resource
def train_model(df):
    X = df.drop("label", axis=1)
    y = df["label"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    return model, scaler, acc


# ----------------------------
# Main App
# ----------------------------
def main():
    st.title("🌾 AI-Based Crop Recommendation System")
    st.write("Enter soil & climate conditions to get the best crop to grow.")

    df = load_data()
    model, scaler, acc = train_model(df)

    st.sidebar.header("🧪 Input Soil Values")

    N = st.sidebar.number_input("Nitrogen (N)", 0, 200, 50)
    P = st.sidebar.number_input("Phosphorus (P)", 0, 200, 40)
    K = st.sidebar.number_input("Potassium (K)", 0, 200, 40)
    ph = st.sidebar.number_input("Soil pH", 3.0, 10.0, 6.5)
    rainfall = st.sidebar.number_input("Rainfall (mm)", 0, 500, 120)
    temp = st.sidebar.number_input("Temperature (°C)", 10, 50, 25)
    humidity = st.sidebar.number_input("Humidity (%)", 10, 100, 60)

    if st.sidebar.button("🌾 Recommend Crop"):
        inputs = np.array([[N, P, K, temp, humidity, ph, rainfall]])
        scaled_inputs = scaler.transform(inputs)
        crop = model.predict(scaled_inputs)[0]

        st.success(f"Recommended Crop: **{crop}**")

        # ----------------------
        # Download Report
        # ----------------------
        report = f"""
Crop Recommendation Report
--------------------------
Input Conditions:
- Nitrogen: {N}
- Phosphorus: {P}
- Potassium: {K}
- Temperature: {temp}
- Humidity: {humidity}
- pH: {ph}
- Rainfall: {rainfall}

Recommended Crop: {crop}

Model Accuracy: {acc:.2%}
"""

        st.download_button(
            label="📥 Download Recommendation Report",
            data=report,
            file_name="crop_recommendation_report.txt",
            mime="text/plain",
        )

    # ----------------------------
    # Show Feature Importance
    # ----------------------------
    st.subheader("🌱 Feature Importance")
    importances = model.feature_importances_
    feature_names = df.drop("label", axis=1).columns

    fi_df = pd.DataFrame({"Feature": feature_names, "Importance": importances})
    fi_df = fi_df.sort_values(by="Importance", ascending=False)

    st.bar_chart(fi_df.set_index("Feature"))


if __name__ == "__main__":
    main()
