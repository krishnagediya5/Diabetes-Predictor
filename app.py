import streamlit as st
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import plotly.graph_objects as go

# ---------- PAGE ----------
st.set_page_config(page_title="Diabetes Dashboard", page_icon="🩺", layout="wide")

# ---------- THEME TOGGLE ----------
theme = st.toggle("🌙 Dark Mode", value=True)

if theme:
    bg = "#0f2027"
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

.card {{
    padding:20px;
    border-radius:15px;
    background: rgba(255,255,255,0.08);
    margin-bottom:20px;
}}

</style>
""", unsafe_allow_html=True)

# ---------- RANGE ----------
def check_range(value, low, high):
    if value < low:
        return "🔵 Low"
    elif value > high:
        return "🔴 High"
    else:
        return "🟢 Normal"

# ---------- HEADER ----------
st.title("🩺 Diabetes Health Dashboard")
st.caption("AI Smart Risk Analysis")

# ---------- LOAD DATA ----------
df = pd.read_csv("diabetes.csv")

X = df.drop(columns='Outcome', axis=1)
Y = df['Outcome']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

model = SVC(kernel='linear', probability=True)  # only change for %
model.fit(X_train, Y_train)

# ---------- INPUT ----------
st.markdown("### 📋 Patient Details")

col1, col2, col3 = st.columns(3)

with col1:
    preg = st.number_input("Pregnancies", 0, 20, 1)
    glucose = st.number_input("Glucose", 0, 200, 100)

with col2:
    bp = st.number_input("Blood Pressure", 0, 150, 80)
    bmi = st.number_input("BMI", 10.0, 50.0, 22.0)

with col3:
    insulin = st.number_input("Insulin", 0, 300, 80)
    age = st.number_input("Age", 1, 100, 30)

skin = st.slider("Skin Thickness", 0, 100, 20)
dpf = st.slider("DPF", 0.0, 2.5, 0.5)

# ---------- BUTTON ----------
if st.button("🔍 Analyze Health", use_container_width=True):

    input_data = np.array([[preg, glucose, bp, skin, insulin, bmi, dpf, age]])
    input_data = scaler.transform(input_data)

    prediction = model.predict(input_data)
    prob = model.predict_proba(input_data)[0][1] * 100  # risk %

    st.markdown("## 🧾 Report")

    # ---------- RISK ----------
    st.metric("🧠 Diabetes Risk", f"{prob:.2f}%")

    if prediction[0] == 1:
        st.error("⚠️ High Risk")
    else:
        st.success("✅ Low Risk")

    # ---------- BMI GAUGE ----------
    st.subheader("⚖️ BMI Gauge")

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=bmi,
        title={'text': "BMI"},
        gauge={
            'axis': {'range': [10, 50]},
            'steps': [
                {'range': [10, 18.5], 'color': "lightblue"},
                {'range': [18.5, 25], 'color': "green"},
                {'range': [25, 50], 'color': "red"}
            ],
        }
    ))
    st.plotly_chart(fig, use_container_width=True)

    # ---------- GLUCOSE CHART ----------
    st.subheader("📊 Glucose Comparison")

    fig2 = go.Figure()
    fig2.add_trace(go.Bar(x=["Your Value", "Normal Avg"], y=[glucose, 110]))
    st.plotly_chart(fig2, use_container_width=True)

# ---------- CHATBOT ----------
st.markdown("## 🤖 Health Assistant")

if "chat" not in st.session_state:
    st.session_state.chat = []

user_msg = st.text_input("Ask something...")

if user_msg:
    st.session_state.chat.append(("You", user_msg))

    # simple logic chatbot
    if "diabetes" in user_msg.lower():
        reply = "Diabetes is a condition where blood sugar is high."
    elif "bmi" in user_msg.lower():
        reply = "BMI between 18.5–24.9 is considered healthy."
    else:
        reply = "Try asking about diabetes, BMI, or health tips."

    st.session_state.chat.append(("Bot", reply))

for sender, msg in st.session_state.chat:
    st.write(f"**{sender}:** {msg}")

# ---------- MOBILE FIX ----------
st.markdown("""
<style>
@media (max-width: 768px) {
    .main {
        padding:10px;
    }
}
</style>
""", unsafe_allow_html=True)

# ---------- FOOTER ----------
st.markdown("---")
st.caption("⚠️ Not medical advice")
