import streamlit as st
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import plotly.graph_objects as go
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet

# ---------- PAGE ----------
st.set_page_config(page_title="Ultimate AI Health Dashboard", layout="wide")

# ---------- SIDEBAR ----------
st.sidebar.title("⚙️ Settings")
theme = st.sidebar.toggle("🌙 Dark Mode", True)

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

.card {{
    background: rgba(255,255,255,0.08);
    padding:20px;
    border-radius:20px;
    margin-bottom:20px;
}}

.badge {{
    padding:4px 10px;
    border-radius:10px;
    color:white;
    font-size:13px;
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
st.title("🩺 Ultimate AI Diabetes Dashboard")

# ---------- DATA ----------
df = pd.read_csv("diabetes.csv")
X = df.drop(columns='Outcome', axis=1)
Y = df['Outcome']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

model = SVC(kernel='linear', probability=True)
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
    bp = st.number_input("💓 BP", 0, 150, 80)
    bmi = st.number_input("⚖️ BMI", 10.0, 50.0, 22.0)

with col3:
    insulin = st.number_input("💉 Insulin", 0, 300, 80)
    age = st.number_input("🎂 Age", 1, 100, 30)

skin = st.slider("Skin", 0, 100, 20)
dpf = st.slider("DPF", 0.0, 2.5, 0.5)

# ---------- ANALYZE ----------
if st.button("🚀 Analyze"):

    input_data = np.array([[preg, glucose, bp, skin, insulin, bmi, dpf, age]])
    input_data = scaler.transform(input_data)

    pred = model.predict(input_data)
    prob = model.predict_proba(input_data)[0][1] * 100

    st.markdown("## 🧾 Report")

    st.metric("🧠 Risk", f"{prob:.2f}%")
    st.progress(int(prob))

    if pred[0] == 1:
        st.error("High Risk")
    else:
        st.success("Low Risk")

    # ---------- BMI GAUGE ----------
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=bmi,
        title={'text': "BMI"},
        gauge={'axis': {'range': [10, 50]}}
    ))
    st.plotly_chart(fig, use_container_width=True)

    # ---------- GLUCOSE CHART ----------
    fig2 = go.Figure()
    fig2.add_trace(go.Bar(x=["You","Normal"], y=[glucose,110]))
    st.plotly_chart(fig2, use_container_width=True)

    # ---------- PDF ----------
    def create_pdf():
        doc = SimpleDocTemplate("report.pdf")
        styles = getSampleStyleSheet()
        content = []
        content.append(Paragraph(f"Risk: {prob:.2f}%", styles["Title"]))
        doc.build(content)

    create_pdf()

    with open("report.pdf", "rb") as f:
        st.download_button("📄 Download Report", f, file_name="report.pdf")

# ---------- CHATBOT ----------
st.markdown("## 🤖 AI Assistant")

if "chat" not in st.session_state:
    st.session_state.chat = []

msg = st.text_input("Ask...")

if msg:
    st.session_state.chat.append(("You", msg))

    if "diabetes" in msg.lower():
        reply = "Diabetes = High blood sugar condition"
    else:
        reply = "Ask about health, BMI, diabetes"

    st.session_state.chat.append(("Bot", reply))

for s,m in st.session_state.chat:
    st.write(f"**{s}:** {m}")

# ---------- FOOTER ----------
st.markdown("---")
st.caption("⚠️ Not medical advice")
