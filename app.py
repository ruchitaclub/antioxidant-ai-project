import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Load model
model = joblib.load("antioxidant_model.pkl")
scaler = joblib.load("scaler.pkl")

# Feature names
feature_names = ["MolWeight", "HDonors", "HAcceptors", "LogP"]

# Herbal data
herbal_data = {
    "Tulsi": [350, 2, 5, 3.0],
    "Neem": [420, 1, 7, 2.4],
    "Aloe Vera": [300, 3, 4, 2.8],
    "Ashwagandha": [380, 2, 6, 2.9],
    "Turmeric": [370, 1, 5, 3.2]
}

# Title
st.title("🌿 Antioxidant Activity Prediction System")

# Input
name = st.text_input("Enter Herbal Compound Name")

if st.button("Predict"):

    if name in herbal_data:
        sample_df = pd.DataFrame([herbal_data[name]], columns=feature_names)

        sample_scaled = scaler.transform(sample_df)
        prediction = model.predict(sample_scaled)

        st.success(f"Predicted Antioxidant Activity of {name}: {prediction[0]:.2f}")

        # Graph
        names = []
        values = []

        for herb, features in herbal_data.items():
            df = pd.DataFrame([features], columns=feature_names)
            scaled = scaler.transform(df)
            pred = model.predict(scaled)

            names.append(herb)
            values.append(pred[0])

        fig, ax = plt.subplots()
        ax.bar(names, values)
        ax.set_title("Antioxidant Activity")
        ax.set_ylabel("Activity")

        st.pyplot(fig)

    else:
        st.error("❌ Compound not found")
