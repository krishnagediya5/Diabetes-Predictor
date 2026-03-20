import streamlit as st
import sqlite3
import numpy as np
import pandas as pd
import datetime
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import plotly.graph_objects as go
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet
import matplotlib.pyplot as plt

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
st.set_page_config(page_title="Diabetes Dashboard", layout="wide")

# ---------- UI ----------
st.markdown("""
<style>
.main {
    background: linear-gradient(135deg,#0f2027,#203a43,#2c5364);
    color:white;
}
.card {
    background: rgba(255,255,255,0.08);
    backdrop-filter: blur(15px);
    padding:20px;
    border-radius:20px;
    margin-bottom:20px;
}
.title {
    font-size:40px;
    text-align:center;
    font-weight:bold;
    background: linear-gradient(90deg,#00f2fe,#4facfe);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
</style>
""", unsafe_allow_html=True)

# ---------- SESSION ----------
if "login" not in st.session_state:
    st.session_state.login=False

# ---------- LOGIN ----------
if not st.session_state.login:

    st.markdown('<div class="title">🍬 Diabetes AI System</div>', unsafe_allow_html=True)

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
                st.error("Invalid credentials")

    with tab2:
        u = st.text_input("New Username")
        p = st.text_input("New Password", type="password")

        if st.button("Create"):
            add_user(u,p)
            st.success("Account created")

# ---------- MAIN ----------
else:

    st.sidebar.success(f"👋 {st.session_state.user}")
    admin_mode = st.sidebar.toggle("🧑‍⚕️ Admin Dashboard")

    if st.sidebar.button("Logout"):
        st.session_state.login=False
        st.rerun()

    st.markdown('<div class="title">🩺 Diabetes Dashboard</div>', unsafe_allow_html=True)

    # ---------- ADMIN ----------
    if admin_mode:
        st.subheader("📊 Admin Panel")

        c.execute("SELECT COUNT(*) FROM users")
        users = c.fetchone()[0]

        c.execute("SELECT AVG(risk) FROM history")
        avg = c.fetchone()[0]

        st.metric("Users", users)
        st.metric("Avg Risk", f"{avg:.2f}" if avg else "0")

        c.execute("SELECT * FROM history")
        data = c.fetchall()

        if data:
            df_admin = pd.DataFrame(data, columns=["User","Result","Risk"])
            st.dataframe(df_admin)

            risks = [i[2] for i in data]
            fig = go.Figure(go.Scatter(y=risks, mode='lines+markers'))
            st.plotly_chart(fig)

        st.stop()

    # ---------- MODEL ----------
    df = pd.read_csv("diabetes.csv")
    X = df.drop(columns='Outcome', axis=1)
    Y = df['Outcome']

    X_train,_,Y_train,_ = train_test_split(X,Y,test_size=0.2)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)

    model = SVC(kernel='linear', probability=True)
    model.fit(X_train,Y_train)

    # ---------- INPUT ----------
    st.markdown('<div class="card">', unsafe_allow_html=True)

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

    st.markdown('</div>', unsafe_allow_html=True)

    # ---------- ANALYZE ----------
    if st.button("🚀 Analyze"):

        input_data = np.array([[preg,glucose,bp,skin,insulin,bmi,dpf,age]])
        input_data = scaler.transform(input_data)

        pred = model.predict(input_data)
        prob = model.predict_proba(input_data)[0][1]*100

        result = "High Risk" if pred[0]==1 else "Low Risk"
        save_result(st.session_state.user,result,prob)

        st.metric("Risk %", f"{prob:.2f}")
        st.progress(int(prob))

        if pred[0]==1:
            st.error("⚠️ High Risk")
        else:
            st.success("✅ Healthy")

        # ---------- GRAPHS ----------
        fig1 = go.Figure(go.Bar(x=["You","Normal"], y=[glucose,110]))
        st.plotly_chart(fig1)

        fig2 = go.Figure(go.Pie(labels=["Risk","Safe"], values=[prob,100-prob]))
        st.plotly_chart(fig2)

        # ---------- PDF GRAPH ----------
        plt.figure()
        plt.bar(["You","Normal"], [glucose,110])
        plt.savefig("graph.png")
        plt.close()

        # ---------- PDF ----------
        def create_pdf():
            doc = SimpleDocTemplate("report.pdf")
            styles = getSampleStyleSheet()
            content = []

            content.append(Paragraph("DIABETES MEDICAL REPORT", styles["Title"]))
            content.append(Spacer(1,10))

            content.append(Paragraph(f"Patient: {st.session_state.user}", styles["Normal"]))
            content.append(Paragraph(f"Date: {datetime.date.today()}", styles["Normal"]))

            content.append(Spacer(1,10))
            content.append(Paragraph("Test Results:", styles["Heading2"]))
            content.append(Paragraph(f"Glucose: {glucose}", styles["Normal"]))
            content.append(Paragraph(f"BP: {bp}", styles["Normal"]))
            content.append(Paragraph(f"BMI: {bmi}", styles["Normal"]))

            content.append(Spacer(1,10))
            content.append(Paragraph(f"Diagnosis: {result}", styles["Heading2"]))
            content.append(Paragraph(f"Risk: {prob:.2f}%", styles["Normal"]))

            content.append(Spacer(1,10))
            content.append(Image("graph.png", width=400, height=200))

            content.append(Spacer(1,10))
            content.append(Paragraph("Recommendations:", styles["Heading2"]))
            content.append(Paragraph("• Healthy diet", styles["Normal"]))
            content.append(Paragraph("• Exercise daily", styles["Normal"]))

            doc.build(content)

        create_pdf()

        with open("report.pdf","rb") as f:
            st.download_button("📄 Download Full Medical Report", f, file_name="Diabetes_Report.pdf")

    # ---------- HISTORY ----------
    st.subheader("📈 Your Trend")

    c.execute("SELECT risk FROM history WHERE username=?", (st.session_state.user,))
    data = c.fetchall()

    if data:
        risks = [i[0] for i in data]
        fig3 = go.Figure(go.Scatter(y=risks, mode='lines+markers'))
        st.plotly_chart(fig3)

    # ---------- TABLE ----------
    st.subheader("📊 Your History")
    c.execute("SELECT * FROM history WHERE username=?", (st.session_state.user,))
    data = c.fetchall()

    if data:
        st.dataframe(pd.DataFrame(data, columns=["User","Result","Risk"]))

    # ---------- DISCLAIMER ----------
    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.subheader("⚠️ Medical Disclaimer")

    st.write("""
    This Diabetes Prediction System is developed for educational and informational purposes only. 
    The predictions generated by this application are based on machine learning models and should not be considered as medical advice, diagnosis, or treatment.

    This tool is designed to help users understand potential diabetes risk factors, but it does not replace professional healthcare consultation.

    Always consult a qualified doctor or healthcare provider for accurate diagnosis and medical guidance.
    """)

    st.markdown('</div>', unsafe_allow_html=True)
