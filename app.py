import streamlit as st
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Page config
st.set_page_config(page_title="🩺 Diabetes Predictor", page_icon="🩺", layout="centered")

# Custom CSS for styling
st.markdown("""
<style>
.big-title {
    font-size:40px !important;
    font-weight: bold;
    text-align:center;
    color:#FF4B4B;
}
.card {
    padding:20px;
    border-radius:15px;
    background-color:#f9f9f9;
    box-shadow:0px 4px 10px rgba(0,0,0,0.1);
}
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<p class="big-title">🩺 Diabetes Prediction System</p>', unsafe_allow_html=True)

# Disclaimer
st.warning("⚠️ This tool is for educational purposes only. Not a medical diagnosis.")

# Load dataset
df = pd.read_csv("diabetes.csv")

X = df.drop(columns='Outcome', axis=1)
Y = df['Outcome']

# Split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

# Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

# Model
model = SVC(kernel='linear')
model.fit(X_train, Y_train)

# Section
st.markdown("### 📋 Enter Patient Details")

col1, col2 = st.columns(2)

with col1:
    preg = st.number_input("🤰 Pregnancies", 0, 20, 1)
    glucose = st.number_input("🍬 Glucose Level", 0, 200, 100)
    bp = st.number_input("💓 Blood Pressure", 0, 150, 70)
    skin = st.number_input("🧬 Skin Thickness", 0, 100, 20)

with col2:
    insulin = st.number_input("💉 Insulin", 0, 900, 80)
    bmi = st.number_input("⚖️ BMI", 0.0, 70.0, 25.0)
    dpf = st.number_input("🧬 DPF", 0.0, 2.5, 0.5)
    age = st.number_input("🎂 Age", 1, 120, 30)

st.markdown("---")

# Button
if st.button("🔍 Predict Now", use_container_width=True):
    input_data = np.array([[preg, glucose, bp, skin, insulin, bmi, dpf, age]])
    input_data = scaler.transform(input_data)

    prediction = model.predict(input_data)

    st.markdown("## 🧾 Result")

    if prediction[0] == 1:
        st.error("⚠️ High Risk of Diabetes Detected")

        st.markdown("### 🧠 Types of Diabetes")
        st.info("""
        🔹 Type 1 – Body does not produce insulin  
        🔹 Type 2 – Body does not use insulin properly (common)  
        🔹 Gestational – During pregnancy  
        """)

        st.markdown("### 🩺 What You Should Do")
        st.success("""
        👨‍⚕️ Consult a doctor  
        🧪 Get blood sugar tests  
        🥗 Follow low-sugar diet  
        🏃 Exercise daily  
        💊 Take medicine only if prescribed  
        """)

    else:
        st.success("✅ Low Risk – No Diabetes Detected")

        st.markdown("### 💡 Stay Healthy Tips")
        st.info("""
        🥗 Eat balanced diet  
        🏃 Exercise regularly  
        ⚖️ Maintain weight  
        🚫 Avoid junk food  
        🩺 Regular checkups  
        """)

# Footer
st.markdown("---")
st.markdown("### 🌟 Stay Healthy, Stay Fit")
st.caption("Made with ❤️ using Machine Learning")