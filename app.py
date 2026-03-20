import streamlit as st
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# ---------- PAGE ----------
st.set_page_config(page_title="Diabetes Health Dashboard", page_icon="🩺", layout="wide")

# ---------- RANGE FUNCTION ----------
def check_range(value, low, high):
    if value < low:
        return "🔵 Low", "#3498db"
    elif value > high:
        return "🔴 High", "#e74c3c"
    else:
        return "🟢 Normal", "#2ecc71"

# ---------- PREMIUM CSS ----------
st.markdown("""
<style>

/* Background */
.main {
    background: linear-gradient(135deg, #141e30, #243b55);
}

/* Header */
.header {
    background: linear-gradient(90deg,#ff416c,#ff4b2b);
    padding: 30px;
    border-radius: 20px;
    text-align: center;
    color: white;
    box-shadow: 0px 10px 30px rgba(0,0,0,0.2);
    margin-bottom: 25px;
}

/* Cards */
.card {
    background: rgba(255,255,255,0.08);
    backdrop-filter: blur(12px);
    padding: 20px;
    border-radius: 18px;
    box-shadow: 0px 8px 25px rgba(0,0,0,0.15);
    margin-bottom: 20px;
    color: white;
}

/* Section title */
.section-title {
    font-size: 22px;
    font-weight: 600;
    margin-bottom: 10px;
}

/* Badge */
.badge {
    padding: 4px 10px;
    border-radius: 10px;
    font-size: 13px;
    font-weight: bold;
    color: white;
}

/* Button */
div.stButton > button {
    background: linear-gradient(90deg,#00c6ff,#0072ff);
    color: white;
    border-radius: 12px;
    height: 50px;
    font-size: 18px;
    border: none;
}

div.stButton > button:hover {
    transform: scale(1.05);
}

</style>
""", unsafe_allow_html=True)

# ---------- HEADER ----------
st.markdown("""
<div class="header">
    <h1>🩺 Diabetes Health Dashboard</h1>
    <p>AI-Powered Smart Health Risk Analysis</p>
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

    glucose = st.number_input("🍬 Glucose", 0, 200, 100)
    status, color = check_range(glucose,70,140)
    st.markdown(f"<span class='badge' style='background:{color}'>{status}</span>", unsafe_allow_html=True)

with col2:
    bp = st.number_input("💓 Blood Pressure", 0, 150, 80)
    status, color = check_range(bp,80,120)
    st.markdown(f"<span class='badge' style='background:{color}'>{status}</span>", unsafe_allow_html=True)

    bmi = st.number_input("⚖️ BMI", 10.0, 50.0, 22.0)
    status, color = check_range(bmi,18.5,24.9)
    st.markdown(f"<span class='badge' style='background:{color}'>{status}</span>", unsafe_allow_html=True)

with col3:
    insulin = st.number_input("💉 Insulin", 0, 300, 80)
    status, color = check_range(insulin,16,166)
    st.markdown(f"<span class='badge' style='background:{color}'>{status}</span>", unsafe_allow_html=True)

    age = st.number_input("🎂 Age", 1, 100, 30)

skin = st.slider("🧬 Skin Thickness", 0, 100, 20)
dpf = st.slider("🧬 Diabetes Pedigree Function", 0.0, 2.5, 0.5)

st.markdown('</div>', unsafe_allow_html=True)

# ---------- BUTTON ----------
if st.button("🚀 Analyze Health", use_container_width=True):

    input_data = np.array([[preg, glucose, bp, skin, insulin, bmi, dpf, age]])
    input_data = scaler.transform(input_data)

    prediction = model.predict(input_data)

    # ---------- RESULT ----------
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🧾 Health Report</div>', unsafe_allow_html=True)

    if prediction[0] == 1:
        st.error("⚠️ High Risk of Diabetes Detected")

        colA, colB = st.columns(2)

        with colA:
            st.markdown("### 🧠 Types of Diabetes")
            st.write("""
            🔹 Type 1 – No insulin production  
            🔹 Type 2 – Insulin resistance  
            🔹 Gestational – During pregnancy  
            """)

        with colB:
            st.markdown("### 🩺 Recommendations")
            st.write("""
            ✔ Consult doctor  
            ✔ Blood test  
            ✔ Healthy diet  
            ✔ Exercise daily  
            """)

    else:
        st.success("✅ You are Healthy")

        st.markdown("### 💡 Maintain Your Health")
        st.write("""
        🥗 Balanced diet  
        🏃 Exercise regularly  
        ⚖ Maintain weight  
        🩺 Regular checkups  
        """)

    st.markdown('</div>', unsafe_allow_html=True)

# ---------- FOOTER ----------
st.markdown("---")
st.caption("🌟 Stay Healthy | Built with ❤️")

# ---------- DISCLAIMER ----------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.warning("""
⚠️ This is not medical advice. Always consult a doctor.
""")
st.markdown('</div>', unsafe_allow_html=True)
