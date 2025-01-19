import streamlit as st
from main import PredictiveMaintenanceModel  # Import backend model class
import joblib

# Load the trained predictive maintenance model
model = PredictiveMaintenanceModel('predictive_maintenance.csv')

# Set model accuracy stats (pre-calculated in backend)
binary_model_accuracy = 98.14  # Binary prediction accuracy
multiclass_model_accuracy = 99.40  # Multiclass prediction accuracy

# Configure the Streamlit app
st.set_page_config(
    page_title="Machine Maintenance Predictor",
    page_icon="🔧",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Main page: Dataset info and app purpose
st.title("🔧 Machine Maintenance Predictor")
st.markdown("""
### About the Dataset
The dataset has details about machine operations like:
- Air temperature, process temperature, speed, torque, and tool wear.

This app helps to:
1. **Predict if a machine will fail (Binary Classification).**
2. **Identify the failure type (Multiclass Classification).**

The system uses **Random Forest models** with high accuracy:
- **Binary Accuracy:** 98.14%
- **Multiclass Accuracy:** 99.40%
""")

# Section divider
st.markdown("---")

# Input section: Collect machine parameters
st.header("🛠️ Enter Machine Data")

col1, col2 = st.columns(2)  # Two columns for better layout

with col1:
    air_temp = st.number_input("🌡️ Air Temp (K)", min_value=200.0, max_value=400.0, value=305.0, step=1.0)
    process_temp = st.number_input("🔥 Process Temp (K)", min_value=200.0, max_value=400.0, value=315.0, step=1.0)

with col2:
    speed = st.number_input("⚙️ Speed (rpm)", min_value=500.0, max_value=5000.0, value=1200.0, step=100.0)
    torque = st.number_input("🔩 Torque (Nm)", min_value=10.0, max_value=100.0, value=40.0, step=1.0)
    wear_time = st.number_input("⏳ Tool Wear (min)", min_value=0.0, max_value=500.0, value=200.0, step=10.0)

# Prediction button
if st.button("🔍 Predict Failure"):
    # Get prediction results
    failure_type, failure_probs, failure_categories = model.predict_failure(
        air_temp, process_temp, speed, torque, wear_time
    )

    # Show prediction results
    st.markdown("---")
    st.header("🔧 Prediction Results")
    st.subheader(f"Predicted Failure: **{failure_type}**")

    st.markdown("#### Probabilities for Each Failure Type:")
    for category, prob in zip(failure_categories, failure_probs):
        st.write(f"- **{category}:** {prob:.4f}")

    import matplotlib.pyplot as plt

    # Confidence Levels Section
    st.subheader("Probability graph:")

    # Create a bar plot
    fig, ax = plt.subplots()
    ax.barh(failure_categories, failure_probs, color='skyblue')
    ax.set_xlim(0, 1)  # Set x-axis to represent probabilities (0 to 1)
    ax.set_xlabel("Probability", fontsize=12)
    ax.set_ylabel("Failure Categories", fontsize=12)
    ax.set_title("Predicted Probabilities for Each Failure Type", fontsize=14)

    # Display probability values on the bars
    for i, v in enumerate(failure_probs):
        ax.text(v + 0.02, i, f"{v:.2f}", color='black', va='center')

    # Show the graph in Streamlit
    st.pyplot(fig)

    # Display model performance
    st.markdown("---")
    st.header("📊 Model Accuracy")
    st.write(f"**Binary Accuracy:** {binary_model_accuracy:.2f}%")
    st.write(f"**Multiclass Accuracy:** {multiclass_model_accuracy:.2f}%")

# Footer
import streamlit as st

# Create a row with two columns
col1, col2 = st.columns([1, 5])  # Adjust column width ratio (1:5)

with col1:
    st.image("eyeofrah.jpg", width=50)  # Replace "profile.jpg" with your image file name

with col2:
    st.markdown("### Aarav Bling")  # Replace with your name
st.markdown("---")
st.markdown("""
#### About This App
Built using **Streamlit** to help predict and understand machine failures.
The model uses **Random Forest Classifier** and methods like **SMOTE** to manage unbalanced data.

---
*Created by aaravbling*
""")