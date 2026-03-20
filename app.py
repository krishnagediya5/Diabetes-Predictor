import streamlit as st
import sqlite3
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import plotly.graph_objects as go
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
import time

# ---------- DATABASE ----------
conn = sqlite3.connect("users.db", check_same_thread=False)
c = conn.cursor()

c.execute("CREATE TABLE IF NOT EXISTS users(username TEXT,password TEXT)")
c.execute("CREATE TABLE IF NOT EXISTS history(username TEXT,result TEXT,risk REAL)")

# ---------- FUNCTIONS ----------
def add_user(u,p):
    c.execute("INSERT INTO users VALUES (?,?)",(u,p))
    conn.commit()

def login_user(u,p):
    c.execute("SELECT * FROM users WHERE username=? AND password=?",(u,p))
    return c.fetchone()

def save_result(u,r,prob):
    c.execute("INSERT INTO history VALUES (?,?,?)",(u,r,prob))
    conn.commit()

# ---------- PAGE ----------
st.set_page_config(page_title="Diabetes AI Dashboard", layout="wide")

# ---------- SESSION ----------
if "login" not in st.session_state:
    st.session_state.login=False

# ---------- LOGIN ----------
if not st.session_state.login:

    st.title("🍬 Diabetes AI System")

    tab1,tab2 = st.tabs(["Login","Signup"])

    with tab1:
        u = st.text_input("Username")
        p = st.text_input("Password", type="password")

        if st.button("Login"):
            if login_user(u,p):
                st.session_state.login=True
                st.session_state.user=u
                st.rerun()
            else:
                st.error("Invalid")

    with tab2:
        u = st.text_input("New Username")
        p = st.text_input("New Password", type="password")

        if st.button("Create"):
            add_user(u,p)
            st.success("Account created")

# ---------- MAIN ----------
else:

    st.sidebar.success(f"👋 {st.session_state.user}")
    if st.sidebar.button("Logout"):
        st.session_state.login=False
        st.rerun()

    st.title("🩺 Diabetes Health Dashboard")

    # ---------- DATA ----------
    df = pd.read_csv("diabetes.csv")
    X = df.drop(columns='Outcome', axis=1)
    Y = df['Outcome']

    X_train,_,Y_train,_ = train_test_split(X,Y,test_size=0.2)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)

    model = SVC(kernel='linear', probability=True)
    model.fit(X_train,Y_train)

    # ---------- INPUT ----------
    col1,col2,col3 = st.columns(3)

    with col1:
        preg = st.number_input("Pregnancies",0,20,1)
        glucose = st.number_input("Glucose",0,200,100)

    with col2:
        bp = st.number_input("BP",0,150,80)
        bmi = st.number_input("BMI",10.0,50.0,22.0)

    with col3:
        insulin = st.number_input("Insulin",0,300,80)
        age = st.number_input("Age",1,100,30)

    skin = st.slider("Skin",0,100,20)
    dpf = st.slider("DPF",0.0,2.5,0.5)

    # ---------- ANALYZE ----------
    if st.button("🚀 Analyze Diabetes Risk"):

        with st.spinner("Analyzing..."):
            time.sleep(1)

        input_data = np.array([[preg,glucose,bp,skin,insulin,bmi,dpf,age]])
        input_data = scaler.transform(input_data)

        pred = model.predict(input_data)
        prob = model.predict_proba(input_data)[0][1]*100

        result = "High Risk" if pred[0]==1 else "Low Risk"
        save_result(st.session_state.user,result,prob)

        # ---------- GAUGE ----------
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob,
            title={'text': "Diabetes Risk %"},
            gauge={'axis': {'range': [0,100]}}
        ))
        st.plotly_chart(fig, use_container_width=True)

        # ---------- RESULT ----------
        st.metric("Risk %", f"{prob:.2f}")
        st.progress(int(prob))

        if pred[0]==1:
            st.error("⚠️ High Risk")
        else:
            st.success("✅ Healthy")

        # ---------- LIVE ANALYTICS ----------
        st.subheader("📊 Live Health Analytics")

        colA,colB = st.columns(2)

        with colA:
            fig2 = go.Figure(go.Bar(x=["Glucose","BP","BMI"], y=[glucose,bp,bmi]))
            st.plotly_chart(fig2, use_container_width=True)

        with colB:
            fig3 = go.Figure(go.Pie(labels=["Risk","Safe"], values=[prob,100-prob]))
            st.plotly_chart(fig3, use_container_width=True)

        # ---------- PDF ----------
        def create_pdf():
            doc = SimpleDocTemplate("report.pdf")
            styles = getSampleStyleSheet()
            content = []

            content.append(Paragraph("Diabetes Health Report", styles["Title"]))
            content.append(Spacer(1,10))
            content.append(Paragraph(f"Patient: {st.session_state.user}", styles["Normal"]))
            content.append(Paragraph(f"Result: {result}", styles["Normal"]))
            content.append(Paragraph(f"Risk: {prob:.2f}%", styles["Normal"]))
            content.append(Spacer(1,10))

            content.append(Paragraph("Recommendations:", styles["Heading2"]))
            content.append(Paragraph("• Healthy diet", styles["Normal"]))
            content.append(Paragraph("• Regular exercise", styles["Normal"]))
            content.append(Paragraph("• Doctor consultation", styles["Normal"]))

            doc.build(content)

        create_pdf()

        with open("report.pdf","rb") as f:
            st.download_button("📄 Download Medical Report", f, file_name="Diabetes_Report.pdf")

    # ---------- HISTORY ANALYTICS ----------
    st.subheader("📈 Your Health Trends")

    c.execute("SELECT risk FROM history WHERE username=?", (st.session_state.user,))
    data = c.fetchall()

    if data:
        risks = [i[0] for i in data]

        fig4 = go.Figure(go.Scatter(y=risks, mode='lines+markers'))
        st.plotly_chart(fig4, use_container_width=True)

        st.write("📊 Your risk trend over time")
