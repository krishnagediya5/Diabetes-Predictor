import streamlit as st
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# ---------- PAGE ----------
st.set_page_config(page_title="Diabetes Dashboard", page_icon="🩺", layout="wide")

# ---------- RANGE FUNCTION ----------
def check_range(value, low, high):
    if value < low:
        return "🔵 Low"
    elif value > high:
        return "🔴 High"
    else:
        return "🟢 Normal"

# ---------- MODERN CSS ----------
st.markdown("""
<style>

/* Background */
.main {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    color: white;
}

/* Header */
.header {
    background: rgba(255,255,255,0.08);
    backdrop-filter: blur(15px);
    padding: 25px;
    border-radius: 20px;
    text-align: center;
    margin-bottom: 20px;
    border: 1px solid rgba(255,255,255,0.2);
}

/* Cards */
.card {
    background: rgba(255,255,255,0.08);
    backdrop-filter: blur(12px);
    padding: 20px;
    border-radius: 18px;
    border: 1px solid rgba(255,255,255,0.15);
    margin-bottom: 20px;
}

/* Titles */
.section-title {
    font-size: 22px;
    font-weight: 600;
    margin-bottom: 10px;
}

/* Button */
div.stButton > button {
    background: linear-gradient(90deg, #00c6ff, #0072ff);
    color: white;
    font-size: 18px;
    border-radius: 12px;
    padding: 12px;
    border: none;
    transition: 0.3s;
}

div.stButton > button:hover {
    transform: scale(1.05);
    background: linear-gradient(90deg, #0072ff, #00c6ff);
}

</style>
""", unsafe_allow_html=True)

# ---------- HEADER ----------
st.markdown("""
<div class="header">
    <h1>🩺 Diabetes Health Dashboard</h1>
    <p>AI-Based Smart Risk Detection System</p>
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

# ---------- INPUT SECTION ----------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">📋 Patient Details</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    preg = st.number_input("🤰 Pregnancies", 0, 20, 1)
    glucose = st.number_input("🍬 Glucose", 0, 200, 100)
    st.caption(f"{check_range(glucose,70,140)}")

with col2:
    bp = st.number_input("💓 Blood Pressure", 0, 150, 80)
    bmi = st.number_input("⚖️ BMI", 10.0, 50.0, 22.0)
    st.caption(f"{check_range(bmi,18.5,24.9)}")

with col3:
    insulin = st.number_input("💉 Insulin", 0, 300, 80)
    age = st.number_input("🎂 Age", 1, 100, 30)

skin = st.slider("🧬 Skin Thickness", 0, 100, 20)
dpf = st.slider("🧬 Diabetes Pedigree Function", 0.0, 2.5, 0.5)

st.markdown('</div>', unsafe_allow_html=True)

# ---------- BUTTON ----------
if st.button("🔍 Analyze Health", use_container_width=True):

    input_data = np.array([[preg, glucose, bp, skin, insulin, bmi, dpf, age]])
    input_data = scaler.transform(input_data)

    prediction = model.predict(input_data)

    # ---------- RESULT ----------
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🧾 Health Report</div>', unsafe_allow_html=True)

    if prediction[0] == 1:
        st.error("⚠️ High Risk of Diabetes")

        colA, colB = st.columns(2)

        with colA:
            st.markdown("### 🧠 Types")
            st.write("""
            🔹 Type 1  
            🔹 Type 2  
            🔹 Gestational  
            """)

        with colB:
            st.markdown("### 🩺 Advice")
            st.write("""
            ✔ Consult doctor  
            ✔ Blood test  
            ✔ Exercise  
            ✔ Healthy diet  
            """)

    else:
        st.success("✅ You are Healthy")

        st.markdown("### 💡 Tips")
        st.write("""
        🥗 Balanced diet  
        🏃 Exercise  
        ⚖ Maintain weight  
        🩺 Checkups  
        """)

    st.markdown('</div>', unsafe_allow_html=True)

# ---------- FOOTER ----------
st.markdown("---")
st.caption("🌟 AI Health Assistant")

# ---------- DISCLAIMER ----------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.warning("""
⚠️ This is not medical advice.  
Consult a doctor for proper diagnosis.
""")
st.markdown('</div>', unsafe_allow_html=True)
