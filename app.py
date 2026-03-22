import streamlit as st
import sqlite3
import numpy as np
import pandas as pd
import datetime

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet

# ---------- DATABASE ----------
conn = sqlite3.connect("users.db", check_same_thread=False)
c = conn.cursor()

c.execute("CREATE TABLE IF NOT EXISTS users(username TEXT,password TEXT)")
c.execute("CREATE TABLE IF NOT EXISTS history(username TEXT,result TEXT,risk REAL)")

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
st.set_page_config(page_title="Diabetes AI System", layout="wide")

st.title("🩺 Diabetes Prediction AI System")

# ---------- SESSION ----------
if "login" not in st.session_state:
    st.session_state.login=False

# ---------- LOGIN ----------
if not st.session_state.login:

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

    st.sidebar.markdown("### 🔗 Project Links")
    st.sidebar.link_button(
        "View GitHub Repository",
        "https://github.com/YOUR_USERNAME/YOUR_REPO"
    )

    if st.sidebar.button("Logout"):
        st.session_state.login=False
        st.rerun()

    # ---------- LOAD DATA ----------
    df = pd.read_csv("diabetes.csv")

    X = df.drop(columns='Outcome')
    Y = df['Outcome']

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = SVC(kernel='linear', probability=True)
    model.fit(X_train, Y_train)

    # ---------- METRICS ----------
    Y_pred = model.predict(X_test)

    accuracy = accuracy_score(Y_test, Y_pred)
    precision = precision_score(Y_test, Y_pred)
    recall = recall_score(Y_test, Y_pred)
    f1 = f1_score(Y_test, Y_pred)

    # ---------- ABOUT MODEL ----------
    st.header("📘 About the Model")

    st.write("""
    Model: Support Vector Machine (SVM)  
    Dataset: PIMA Indians Diabetes Dataset  
    Features: 8 medical indicators  
    Goal: Predict diabetes risk  
    """)

    # ---------- METRICS DISPLAY ----------
    st.subheader("📊 Model Performance")

    col1,col2,col3,col4 = st.columns(4)

    col1.metric("Accuracy", f"{accuracy:.2f}")
    col2.metric("Precision", f"{precision:.2f}")
    col3.metric("Recall", f"{recall:.2f}")
    col4.metric("F1 Score", f"{f1:.2f}")

    # ---------- HEATMAP ----------
    st.subheader("📊 Feature Correlation")

    fig, ax = plt.subplots()

    sns.heatmap(
        df.corr(),
        annot=True,
        cmap="coolwarm",
        ax=ax
    )

    st.pyplot(fig)

    # ---------- INPUT ----------
    st.header("🧪 Enter Patient Details")

    col1,col2,col3 = st.columns(3)

    with col1:
        preg = st.number_input(
            "Pregnancies",
            0,20,1,
            help="Number of pregnancies"
        )

        glucose = st.number_input(
            "Glucose",
            0,200,100,
            help="Blood glucose level"
        )

    with col2:
        bp = st.number_input(
            "Blood Pressure",
            0,150,80
        )

        bmi = st.number_input(
            "BMI",
            10.0,50.0,22.0
        )

    with col3:
        insulin = st.number_input(
            "Insulin",
            0,300,80
        )

        age = st.number_input(
            "Age",
            1,100,30
        )

    skin = st.slider("Skin Thickness",0,100,20)

    dpf = st.slider(
        "Diabetes Pedigree Function",
        0.0,2.5,0.5
    )

    # ---------- VALIDATION ----------
    if glucose <= 0:
        st.warning("Glucose should be greater than 0")

    # ---------- PREDICTION ----------
    if st.button("🚀 Analyze"):

        try:

            with st.spinner("Analyzing patient data..."):

                input_data = np.array([[
                    preg,
                    glucose,
                    bp,
                    skin,
                    insulin,
                    bmi,
                    dpf,
                    age
                ]])

                input_data = scaler.transform(input_data)

                pred = model.predict(input_data)

                prob = model.predict_proba(input_data)[0][1] * 100

                result = (
                    "High Risk"
                    if pred[0]==1
                    else "Low Risk"
                )

                save_result(
                    st.session_state.user,
                    result,
                    prob
                )

            st.success("Prediction Complete")

            st.metric(
                "Diabetes Risk %",
                f"{prob:.2f}%"
            )

            st.progress(int(prob))

            if pred[0]==1:

                st.error("⚠️ High Risk")

            else:

                st.success("✅ Low Risk")

            # ---------- BAR ----------
            fig1 = go.Figure(
                go.Bar(
                    x=["You","Normal"],
                    y=[glucose,110]
                )
            )

            st.plotly_chart(fig1)

            # ---------- PIE ----------
            fig2 = go.Figure(
                go.Pie(
                    labels=["Risk","Safe"],
                    values=[prob,100-prob]
                )
            )

            st.plotly_chart(fig2)

            # ---------- PDF ----------
            plt.figure()

            plt.bar(
                ["You","Normal"],
                [glucose,110]
            )

            plt.savefig("graph.png")

            plt.close()

            def create_pdf():

                doc = SimpleDocTemplate(
                    "report.pdf"
                )

                styles = getSampleStyleSheet()

                content = []

                content.append(
                    Paragraph(
                        "DIABETES MEDICAL REPORT",
                        styles["Title"]
                    )
                )

                content.append(
                    Spacer(1,10)
                )

                content.append(
                    Paragraph(
                        f"Patient: {st.session_state.user}",
                        styles["Normal"]
                    )
                )

                content.append(
                    Paragraph(
                        f"Date: {datetime.date.today()}",
                        styles["Normal"]
                    )
                )

                content.append(
                    Spacer(1,10)
                )

                content.append(
                    Paragraph(
                        f"Diagnosis: {result}",
                        styles["Heading2"]
                    )
                )

                content.append(
                    Paragraph(
                        f"Risk: {prob:.2f}%",
                        styles["Normal"]
                    )
                )

                content.append(
                    Spacer(1,10)
                )

                content.append(
                    Image(
                        "graph.png",
                        width=400,
                        height=200
                    )
                )

                doc.build(content)

            create_pdf()

            with open(
                "report.pdf",
                "rb"
            ) as f:

                st.download_button(
                    "📄 Download Report",
                    f,
                    file_name="Diabetes_Report.pdf"
                )

        except Exception as e:

            st.error(
                "Something went wrong"
            )

    # ---------- HISTORY ----------
    st.subheader("📈 Your Risk Trend")

    c.execute(
        "SELECT risk FROM history WHERE username=?",
        (st.session_state.user,)
    )

    data = c.fetchall()

    if data:

        risks = [i[0] for i in data]

        fig3 = go.Figure(
            go.Scatter(
                y=risks,
                mode='lines+markers'
            )
        )

        st.plotly_chart(fig3)

    # ---------- FOOTER ----------
    st.markdown("---")

    st.markdown(
        "Developed by **Your Name** | Machine Learning Project"
    )
