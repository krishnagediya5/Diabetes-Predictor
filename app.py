import streamlit as st
import sqlite3
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import plotly.graph_objects as go
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib import colors
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
st.set_page_config(page_title="Diabetes AI Platform", layout="wide")

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

    user = st.session_state.user

    st.sidebar.success(f"👋 {user}")

    if st.sidebar.button("Logout"):
        st.session_state.login=False
        st.rerun()

    # ---------- ADMIN CHECK ----------
    if user == "admin":

        st.title("🧑‍💼 Admin Dashboard")

        # USERS
        c.execute("SELECT COUNT(*) FROM users")
        total_users = c.fetchone()[0]

        c.execute("SELECT AVG(risk) FROM history")
        avg_risk = c.fetchone()[0] or 0

        col1,col2 = st.columns(2)

        col1.metric("👥 Total Users", total_users)
        col2.metric("🧠 Avg Risk", f"{avg_risk:.2f}")

        # ALL DATA
        st.subheader("📊 All User Data")
        c.execute("SELECT * FROM history")
        data = c.fetchall()

        if data:
            df_all = pd.DataFrame(data, columns=["User","Result","Risk"])
            st.dataframe(df_all)

            # GRAPH
            fig = go.Figure(go.Bar(x=df_all["User"], y=df_all["Risk"]))
            st.plotly_chart(fig, use_container_width=True)

    else:

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
            save_result(user,result,prob)

            st.metric("Risk %", f"{prob:.2f}")
            st.progress(int(prob))

            if pred[0]==1:
                st.error("⚠️ High Risk")
            else:
                st.success("✅ Healthy")

            # ---------- PDF REPORT ----------
            def create_pdf():
                doc = SimpleDocTemplate("report.pdf")
                styles = getSampleStyleSheet()

                data = [
                    ["Parameter","Value"],
                    ["Pregnancies",preg],
                    ["Glucose",glucose],
                    ["BP",bp],
                    ["BMI",bmi],
                    ["Insulin",insulin],
                    ["Age",age],
                    ["Risk %",f"{prob:.2f}"],
                    ["Result",result]
                ]

                table = Table(data)
                table.setStyle(TableStyle([
                    ('BACKGROUND',(0,0),(-1,0),colors.grey),
                    ('TEXTCOLOR',(0,0),(-1,0),colors.white),
                    ('GRID',(0,0),(-1,-1),1,colors.black)
                ]))

                content = []
                content.append(Paragraph("Diabetes Medical Report", styles["Title"]))
                content.append(Spacer(1,10))
                content.append(table)

                doc.build(content)

            create_pdf()

            with open("report.pdf","rb") as f:
                st.download_button("📄 Download Report", f, file_name="Medical_Report.pdf")

        # ---------- HISTORY ----------
        st.subheader("📊 Your History")

        c.execute("SELECT * FROM history WHERE username=?", (user,))
        data = c.fetchall()

        if data:
            st.dataframe(pd.DataFrame(data, columns=["User","Result","Risk"]))
