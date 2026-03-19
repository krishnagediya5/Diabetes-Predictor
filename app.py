import streamlit as st
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Page setup
st.set_page_config(page_title="Diabetes Health Dashboard", page_icon="🩺", layout="wide")

# ---------- RANGE FUNCTION ----------
def check_range(value, low, high):
    if value < low:
        return "🔵 Low"
    elif value > high:
        return "🔴 High"
    else:
        return "🟢 Normal"

# ---------- UI STYLE ----------
st.markdown("""
<style>
.main {
    background-color: #f5f7fa;
}
.title {
    font-size: 40px;
    font-weight: bold;
    color: #2c3e50;
}
</style>
""", unsafe_allow_html=True)

# Title
st.markdown("""
<div style='text-align: center; padding: 20px;'>
    <h1 style='color:#FF4B4B; font-size:42px; margin-bottom:5px;'>
        🩺 Diabetes Health Dashboard
    </h1>
    <p style='font-size:18px; color:gray;'>
        Smart Health Analysis & Risk Prediction System
    </p>
</div>
""", unsafe_allow_html=True)



# ---------- LOAD DATA ----------
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

# ---------- INPUT SECTION ----------
st.markdown("### 📋 Enter Patient Details")

col1, col2, col3 = st.columns(3)

with col1:
    preg = st.number_input("🤰 Pregnancies", 0, 20, 1)
    st.caption("Normal: 0–10")

    glucose = st.number_input("🍬 Glucose", 0, 200, 100)
    st.caption(f"Normal: 70–140 → {check_range(glucose,70,140)}")

with col2:
    bp = st.number_input("💓 Blood Pressure", 0, 150, 80)
    st.caption(f"Normal: 80–120 → {check_range(bp,80,120)}")

    bmi = st.number_input("⚖️ BMI", 10.0, 50.0, 22.0)
    st.caption(f"Normal: 18.5–24.9 → {check_range(bmi,18.5,24.9)}")

with col3:
    insulin = st.number_input("💉 Insulin", 0, 300, 80)
    st.caption(f"Normal: 16–166 → {check_range(insulin,16,166)}")

    age = st.number_input("🎂 Age", 1, 100, 30)
    st.caption("Depends on person")

# Extra sliders
skin = st.slider("🧬 Skin Thickness", 0, 100, 20)
st.caption("Typical: 10–40")

dpf = st.slider("🧬 Diabetes Pedigree Function", 0.0, 2.5, 0.5)
st.caption("Higher = more genetic risk")

st.markdown("---")

# ---------- PREDICTION ----------
if st.button("🔍 Analyze Health", use_container_width=True):

    input_data = np.array([[preg, glucose, bp, skin, insulin, bmi, dpf, age]])
    input_data = scaler.transform(input_data)

    prediction = model.predict(input_data)

    st.markdown("## 🧾 Health Report")

    if prediction[0] == 1:
        st.error("⚠️ High Risk of Diabetes Detected")

        colA, colB = st.columns(2)

        with colA:
            st.markdown("### 🧠 Types of Diabetes")
            st.write("""
            🔹 Type 1 – No insulin production  
            🔹 Type 2 – Insulin resistance (most common)  
            🔹 Gestational – During pregnancy  
            """)

        with colB:
            st.markdown("### 🩺 What You Should Do")
            st.write("""
            ✔ Consult a doctor  
            ✔ Blood sugar test (Fasting / HbA1c)  
            ✔ Follow low sugar diet  
            ✔ Daily exercise  
            ✔ Avoid junk food  
            """)

    else:
        st.success("✅ You are likely Healthy")

        st.markdown("### 💡 Maintain Your Health")
        st.write("""
        🥗 Balanced diet  
        🏃 Exercise regularly  
        ⚖ Maintain healthy weight  
        🩺 Regular health checkups  
        """)

# ---------- FOOTER ----------
st.markdown("---")
st.caption("🌟 Stay Healthy | Made with ❤️ using Machine Learning")
st.warning("""
⚠️ Disclaimer

This tool is designed to give a general idea about diabetes risk using machine learning techniques.  
It is not intended to provide medical diagnosis or treatment recommendations.

Consider the results as informative guidance only, and always seek advice from a healthcare professional for medical concerns.
""")
