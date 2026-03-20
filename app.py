import streamlit as st
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import plotly.graph_objects as go

# ---------- PAGE ----------
st.set_page_config(page_title="AI Health Dashboard", layout="wide")

# ---------- CUSTOM CSS ----------
st.markdown("""
<style>

/* Background */
.main {
    background: linear-gradient(135deg, #1f4037, #99f2c8);
}

/* Card */
.card {
    background: white;
    padding: 20px;
    border-radius: 20px;
    box-shadow: 0px 10px 25px rgba(0,0,0,0.15);
    margin-bottom: 20px;
}

/* Title */
.title {
    font-size: 32px;
    font-weight: bold;
    text-align: center;
    color: #2c3e50;
}

/* Badge */
.badge {
    padding: 5px 10px;
    border-radius: 10px;
    font-size: 14px;
    font-weight: bold;
}

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
st.markdown('<div class="title">🩺 AI Diabetes Health Dashboard</div>', unsafe_allow_html=True)
st.caption("Smart Risk Detection System")

# ---------- LOAD DATA ----------
df = pd.read_csv("diabetes.csv")

X = df.drop(columns='Outcome', axis=1)
Y = df['Outcome']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

model = SVC(kernel='linear', probability=True)
model.fit(X_train, Y_train)

# ---------- INPUT ----------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("📋 Patient Details")

col1, col2, col3 = st.columns(3)

with col1:
    preg = st.number_input("🤰 Pregnancies", 0, 20, 1)

    glucose = st.number_input("🍬 Glucose", 0, 200, 100)
    status, color = check_range(glucose, 70, 140)
    st.markdown(f"<span class='badge' style='background:{color};color:white'>{status}</span>", unsafe_allow_html=True)

with col2:
    bp = st.number_input("💓 Blood Pressure", 0, 150, 80)
    status, color = check_range(bp, 80, 120)
    st.markdown(f"<span class='badge' style='background:{color};color:white'>{status}</span>", unsafe_allow_html=True)

    bmi = st.number_input("⚖️ BMI", 10.0, 50.0, 22.0)
    status, color = check_range(bmi, 18.5, 24.9)
    st.markdown(f"<span class='badge' style='background:{color};color:white'>{status}</span>", unsafe_allow_html=True)

with col3:
    insulin = st.number_input("💉 Insulin", 0, 300, 80)
    status, color = check_range(insulin, 16, 166)
    st.markdown(f"<span class='badge' style='background:{color};color:white'>{status}</span>", unsafe_allow_html=True)

    age = st.number_input("🎂 Age", 1, 100, 30)

skin = st.slider("🧬 Skin Thickness", 0, 100, 20)
dpf = st.slider("🧬 Diabetes Pedigree Function", 0.0, 2.5, 0.5)

st.markdown('</div>', unsafe_allow_html=True)

# ---------- BUTTON ----------
if st.button("🚀 Analyze Health", use_container_width=True):

    input_data = np.array([[preg, glucose, bp, skin, insulin, bmi, dpf, age]])
    input_data = scaler.transform(input_data)

    prediction = model.predict(input_data)
    prob = model.predict_proba(input_data)[0][1] * 100

    # ---------- RESULT ----------
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("🧾 Health Report")

    # ---------- RISK BAR ----------
    st.progress(int(prob))
    st.metric("🧠 Risk Score", f"{prob:.2f}%")

    if prediction[0] == 1:
        st.error("⚠️ High Diabetes Risk Detected")
    else:
        st.success("✅ You are Healthy")

    # ---------- BMI GAUGE ----------
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=bmi,
        title={'text': "BMI Level"},
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
    fig2 = go.Figure()
    fig2.add_trace(go.Bar(
        x=["Your Glucose", "Normal Avg"],
        y=[glucose, 110],
        text=[glucose, 110],
        textposition='auto'
    ))
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)

# ---------- CHATBOT ----------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("🤖 Health Assistant")

if "chat" not in st.session_state:
    st.session_state.chat = []

user_msg = st.text_input("💬 Ask anything about health...")

if user_msg:
    st.session_state.chat.append(("You", user_msg))

    if "diabetes" in user_msg.lower():
        reply = "🧠 Diabetes = High blood sugar problem."
    elif "bmi" in user_msg.lower():
        reply = "⚖️ Healthy BMI: 18.5–24.9"
    elif "exercise" in user_msg.lower():
        reply = "🏃 Daily 30 min walk recommended!"
    else:
        reply = "🤖 Try asking about diabetes, BMI, diet, exercise."

    st.session_state.chat.append(("Bot", reply))

for sender, msg in st.session_state.chat:
    st.write(f"**{sender}:** {msg}")

st.markdown('</div>', unsafe_allow_html=True)

# ---------- FOOTER ----------
st.markdown("---")
st.caption("⚠️ Not medical advice | Built with ❤️")
