import streamlit as st
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# ---------- PAGE ----------
st.set_page_config(page_title="AI Diabetes Dashboard", layout="wide")

# ---------- SIDEBAR ----------
st.sidebar.title("⚙️ Settings")
theme = st.sidebar.toggle("🌙 Dark Mode", value=True)

st.sidebar.markdown("---")
st.sidebar.info("🧠 AI Health Predictor\n\nBuilt using Machine Learning")

# ---------- THEME ----------
if theme:
    bg = "linear-gradient(135deg,#0f2027,#203a43,#2c5364)"
    text = "white"
else:
    bg = "#f5f7fa"
    text = "black"

# ---------- CSS ----------
st.markdown(f"""
<style>

.main {{
    background: {bg};
    color: {text};
}}

/* HEADER */
.header {{
    background: linear-gradient(90deg,#ff416c,#ff4b2b);
    padding: 25px;
    border-radius: 20px;
    text-align: center;
    color: white;
    box-shadow: 0px 10px 30px rgba(0,0,0,0.2);
}}

/* CARD */
.card {{
    background: rgba(255,255,255,0.08);
    backdrop-filter: blur(12px);
    padding: 20px;
    border-radius: 18px;
    margin-bottom: 20px;
}}

/* BADGE */
.badge {{
    padding: 4px 10px;
    border-radius: 10px;
    color: white;
    font-size: 13px;
}}

</style>
""", unsafe_allow_html=True)

# ---------- RANGE ----------
def check_range(value, low, high):
    if value < low:
        return "🔵 Low", "#3498db"
    elif value > high:
        return "🔴 High", "#e74c3c"
    else:
        return "🟢 Normal", "#2ecc71"

# ---------- HEADER ----------
st.markdown("""
<div class="header">
    <h1>🩺 AI Diabetes Dashboard</h1>
    <p>Smart Health Risk Detection System</p>
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

# ---------- INPUT ----------
st.markdown("### 📋 Patient Details")

col1, col2, col3 = st.columns(3)

with col1:
    preg = st.number_input("🤰 Pregnancies", 0, 20, 1)

    glucose = st.number_input("🍬 Glucose", 0, 200, 100)
    s,c = check_range(glucose,70,140)
    st.markdown(f"<span class='badge' style='background:{c}'>{s}</span>", unsafe_allow_html=True)

with col2:
    bp = st.number_input("💓 Blood Pressure", 0, 150, 80)
    s,c = check_range(bp,80,120)
    st.markdown(f"<span class='badge' style='background:{c}'>{s}</span>", unsafe_allow_html=True)

    bmi = st.number_input("⚖️ BMI", 10.0, 50.0, 22.0)
    s,c = check_range(bmi,18.5,24.9)
    st.markdown(f"<span class='badge' style='background:{c}'>{s}</span>", unsafe_allow_html=True)

with col3:
    insulin = st.number_input("💉 Insulin", 0, 300, 80)
    s,c = check_range(insulin,16,166)
    st.markdown(f"<span class='badge' style='background:{c}'>{s}</span>", unsafe_allow_html=True)

    age = st.number_input("🎂 Age", 1, 100, 30)

skin = st.slider("🧬 Skin Thickness", 0, 100, 20)
dpf = st.slider("🧬 Diabetes Pedigree Function", 0.0, 2.5, 0.5)

# ---------- BUTTON ----------
if st.button("🚀 Analyze Health", use_container_width=True):

    input_data = np.array([[preg, glucose, bp, skin, insulin, bmi, dpf, age]])
    input_data = scaler.transform(input_data)

    prediction = model.predict(input_data)

    # ---------- RESULT ----------
    st.markdown("## 🧾 Health Report")

    # ---------- VISUAL RISK ----------
    if prediction[0] == 1:
        st.error("⚠️ High Risk")
        st.progress(85)
    else:
        st.success("✅ Low Risk")
        st.progress(25)

    # ---------- REPORT LAYOUT ----------
    colA, colB = st.columns(2)

    with colA:
        st.markdown("### 🧠 Diabetes Info")
        st.write("""
        🔹 Type 1 – Insulin not produced  
        🔹 Type 2 – Insulin resistance  
        🔹 Gestational – During pregnancy  
        """)

    with colB:
        st.markdown("### 🩺 Recommendations")
        st.write("""
        ✔ Doctor consultation  
        ✔ Blood sugar test  
        ✔ Healthy diet  
        ✔ Daily exercise  
        """)

# ---------- FOOTER ----------
st.markdown("---")
st.caption("🌟 AI Health Assistant | Not Medical Advice")
