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

# ---------- PREMIUM CSS ----------
st.markdown("""
<style>
.main {
    background: linear-gradient(135deg, #eef2f3, #ffffff);
}

/* Header */
.glass-header {
    background: linear-gradient(90deg,#ff4b4b,#ff7b7b);
    padding: 25px;
    border-radius: 18px;
    text-align: center;
    color: white;
    box-shadow: 0px 8px 25px rgba(0,0,0,0.15);
}

/* Card */
.card {
    background: rgba(255,255,255,0.85);
    padding: 20px;
    border-radius: 15px;
    box-shadow: 0px 6px 18px rgba(0,0,0,0.08);
    margin-bottom: 20px;
}

/* Section Title */
.section-title {
    font-size: 22px;
    font-weight: 600;
    color: #2c3e50;
    margin-bottom: 10px;
}
</style>
""", unsafe_allow_html=True)

# ---------- HEADER ----------
st.markdown("""
<div class="glass-header">
    <h1>🩺 Diabetes Health Dashboard</h1>
    <p>AI-Powered Smart Health Risk Analysis System</p>
</div>
""", unsafe_allow_html=True)

# ---------- LOAD DATA ----------
df = pd.read_csv("diabetes.csv")

X = df.drop(columns='Outcome', axis=1)
Y = df['Outcome']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

model = SVC(kernel='linear')
model.fit(X_train, Y_train)

# ---------- INPUT CARD ----------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">📋 Enter Patient Details</div>', unsafe_allow_html=True)

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

skin = st.slider("🧬 Skin Thickness", 0, 100, 20)
st.caption("Typical: 10–40")

dpf = st.slider("🧬 Diabetes Pedigree Function", 0.0, 2.5, 0.5)
st.caption("Higher = more genetic risk")

st.markdown('</div>', unsafe_allow_html=True)

# ---------- BUTTON ----------
if st.button("🔍 Analyze Health", use_container_width=True):

    input_data = np.array([[preg, glucose, bp, skin, insulin, bmi, dpf, age]])
    input_data = scaler.transform(input_data)

    prediction = model.predict(input_data)

    # ---------- RESULT CARD ----------
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🧾 Health Report</div>', unsafe_allow_html=True)

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
            ✔ Blood sugar test  
            ✔ Healthy diet  
            ✔ Daily exercise  
            """)

    else:
        st.success("✅ You are likely Healthy")

        st.markdown("### 💡 Maintain Your Health")
        st.write("""
        🥗 Balanced diet  
        🏃 Exercise regularly  
        ⚖ Maintain healthy weight  
        🩺 Regular checkups  
        """)

    st.markdown('</div>', unsafe_allow_html=True)

# ---------- FOOTER ----------
st.markdown("---")
st.caption("🌟 Stay Healthy | Made with ❤️ using Machine Learning")

# ---------- DISCLAIMER ----------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.warning("""
⚠️ Disclaimer

This application provides an estimated diabetes risk based on entered data.  
It is intended for awareness and learning purposes only.

The results should not be considered as medical advice or diagnosis.  
Always consult a qualified healthcare professional for proper evaluation.
""")
st.markdown('</div>', unsafe_allow_html=True)
