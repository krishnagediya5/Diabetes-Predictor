import streamlit as st
import sqlite3
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import plotly.graph_objects as go
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet

# ---------- DATABASE ----------
conn = sqlite3.connect("users.db", check_same_thread=False)
c = conn.cursor()

c.execute("""CREATE TABLE IF NOT EXISTS users(
            username TEXT,
            password TEXT)""")

c.execute("""CREATE TABLE IF NOT EXISTS history(
            username TEXT,
            result TEXT,
            risk REAL)""")

# ---------- FUNCTIONS ----------
def add_user(username, password):
    c.execute("INSERT INTO users VALUES (?,?)", (username, password))
    conn.commit()

def login_user(username, password):
    c.execute("SELECT * FROM users WHERE username=? AND password=?", (username,password))
    return c.fetchone()

def save_result(username, result, risk):
    c.execute("INSERT INTO history VALUES (?,?,?)", (username,result,risk))
    conn.commit()

# ---------- PAGE ----------
st.set_page_config(page_title="AI Diabetes Dashboard", layout="wide")

# ---------- SESSION ----------
if "login" not in st.session_state:
    st.session_state.login = False

# ---------- LOGIN UI ----------
if not st.session_state.login:

    st.title("🔐 Login / Signup")

    choice = st.radio("Select", ["Login","Signup"])

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if choice == "Signup":
        if st.button("Create Account"):
            add_user(username, password)
            st.success("Account created")

    if choice == "Login":
        if st.button("Login"):
            user = login_user(username, password)
            if user:
                st.session_state.login = True
                st.session_state.user = username
                st.success("Logged in")
                st.rerun()
            else:
                st.error("Invalid credentials")

# ---------- MAIN APP ----------
else:

    st.sidebar.success(f"👋 Welcome {st.session_state.user}")
    if st.sidebar.button("Logout"):
        st.session_state.login = False
        st.rerun()

    # ---------- THEME ----------
    theme = st.sidebar.toggle("🌙 Dark Mode", True)

    if theme:
        bg = "linear-gradient(135deg,#0f2027,#203a43,#2c5364)"
        text = "white"
    else:
        bg = "#f5f7fa"
        text = "black"

    st.markdown(f"""
    <style>
    .main {{background:{bg}; color:{text};}}
    .card {{
        background: rgba(255,255,255,0.08);
        padding:20px;
        border-radius:20px;
        margin-bottom:20px;
    }}
    </style>
    """, unsafe_allow_html=True)

    st.title("🩺 AI Diabetes Dashboard")

    # ---------- LOAD DATA ----------
    df = pd.read_csv("diabetes.csv")

    X = df.drop(columns='Outcome', axis=1)
    Y = df['Outcome']

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)

    model = SVC(kernel='linear', probability=True)
    model.fit(X_train, Y_train)

    # ---------- INPUT ----------
    col1, col2, col3 = st.columns(3)

    with col1:
        preg = st.number_input("Pregnancies", 0, 20, 1)
        glucose = st.number_input("Glucose", 0, 200, 100)

    with col2:
        bp = st.number_input("BP", 0, 150, 80)
        bmi = st.number_input("BMI", 10.0, 50.0, 22.0)

    with col3:
        insulin = st.number_input("Insulin", 0, 300, 80)
        age = st.number_input("Age", 1, 100, 30)

    skin = st.slider("Skin", 0, 100, 20)
    dpf = st.slider("DPF", 0.0, 2.5, 0.5)

    # ---------- ANALYZE ----------
    if st.button("🚀 Analyze"):

        input_data = np.array([[preg, glucose, bp, skin, insulin, bmi, dpf, age]])
        input_data = scaler.transform(input_data)

        pred = model.predict(input_data)
        prob = model.predict_proba(input_data)[0][1] * 100

        result = "High Risk" if pred[0]==1 else "Low Risk"

        save_result(st.session_state.user, result, prob)

        st.metric("Risk %", f"{prob:.2f}")
        st.progress(int(prob))

        if pred[0]==1:
            st.error("High Risk")
        else:
            st.success("Low Risk")

        # ---------- CHART ----------
        fig = go.Figure(go.Bar(x=["You","Normal"], y=[glucose,110]))
        st.plotly_chart(fig)

        # ---------- PDF ----------
        def create_pdf():
            doc = SimpleDocTemplate("report.pdf")
            styles = getSampleStyleSheet()
            content = []
            content.append(Paragraph(f"Result: {result}", styles["Title"]))
            content.append(Paragraph(f"Risk: {prob:.2f}%", styles["Normal"]))
            doc.build(content)

        create_pdf()

        with open("report.pdf","rb") as f:
            st.download_button("Download Report", f)

    # ---------- HISTORY ----------
    st.subheader("📊 Your History")

    c.execute("SELECT * FROM history WHERE username=?", (st.session_state.user,))
    data = c.fetchall()

    if data:
        df_hist = pd.DataFrame(data, columns=["User","Result","Risk"])
        st.dataframe(df_hist)
