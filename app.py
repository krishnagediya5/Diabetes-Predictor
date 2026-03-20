import streamlit as st
import sqlite3
import bcrypt
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import plotly.graph_objects as go
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet

# ---------- DATABASE ----------
conn = sqlite3.connect("secure_users.db", check_same_thread=False)
c = conn.cursor()

c.execute("""CREATE TABLE IF NOT EXISTS users(
            username TEXT PRIMARY KEY,
            password BLOB)""")

c.execute("""CREATE TABLE IF NOT EXISTS history(
            username TEXT,
            result TEXT,
            risk REAL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)""")

# ---------- SECURITY FUNCTIONS ----------
def hash_password(password):
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt())

def verify_password(password, hashed):
    return bcrypt.checkpw(password.encode(), hashed)

def add_user(username, password):
    hashed = hash_password(password)
    try:
        c.execute("INSERT INTO users VALUES (?,?)", (username, hashed))
        conn.commit()
        return True
    except:
        return False

def login_user(username, password):
    c.execute("SELECT password FROM users WHERE username=?", (username,))
    data = c.fetchone()
    if data and verify_password(password, data[0]):
        return True
    return False

def save_result(username, result, risk):
    c.execute("INSERT INTO history(username,result,risk) VALUES (?,?,?)",
              (username, result, risk))
    conn.commit()

# ---------- PAGE ----------
st.set_page_config(page_title="Secure AI Dashboard", layout="wide")

# ---------- SESSION ----------
if "login" not in st.session_state:
    st.session_state.login = False

# ---------- LOGIN ----------
if not st.session_state.login:

    st.title("🔐 Secure Login System")

    choice = st.radio("Select", ["Login", "Signup"])

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    # ---------- VALIDATION ----------
    if len(username) > 20:
        st.warning("Username too long")

    if choice == "Signup":
        if st.button("Create Account"):
            if username and password:
                if add_user(username, password):
                    st.success("Account created securely ✅")
                else:
                    st.error("User already exists")
            else:
                st.warning("Fill all fields")

    if choice == "Login":
        if st.button("Login"):
            if login_user(username, password):
                st.session_state.login = True
                st.session_state.user = username
                st.success("Login successful 🔓")
                st.rerun()
            else:
                st.error("Invalid credentials")

# ---------- MAIN ----------
else:

    st.sidebar.success(f"👋 {st.session_state.user}")
    if st.sidebar.button("Logout"):
        st.session_state.login = False
        st.rerun()

    st.title("🩺 Secure AI Diabetes Dashboard")

    # ---------- LOAD DATA ----------
    df = pd.read_csv("diabetes.csv")
    X = df.drop(columns='Outcome', axis=1)
    Y = df['Outcome']

    X_train, _, Y_train, _ = train_test_split(X, Y, test_size=0.2)

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
    if st.button("🚀 Analyze Securely"):

        input_data = np.array([[preg, glucose, bp, skin, insulin, bmi, dpf, age]])
        input_data = scaler.transform(input_data)

        pred = model.predict(input_data)
        prob = model.predict_proba(input_data)[0][1] * 100

        result = "High Risk" if pred[0] == 1 else "Low Risk"

        save_result(st.session_state.user, result, prob)

        st.metric("Risk %", f"{prob:.2f}")
        st.progress(int(prob))

        if pred[0] == 1:
            st.error("⚠️ High Risk")
        else:
            st.success("✅ Low Risk")

        # ---------- CHART ----------
        fig = go.Figure(go.Bar(x=["You", "Normal"], y=[glucose, 110]))
        st.plotly_chart(fig)

        # ---------- PDF ----------
        def create_pdf():
            doc = SimpleDocTemplate("secure_report.pdf")
            styles = getSampleStyleSheet()
            content = []
            content.append(Paragraph(f"User: {st.session_state.user}", styles["Title"]))
            content.append(Paragraph(f"Result: {result}", styles["Normal"]))
            content.append(Paragraph(f"Risk: {prob:.2f}%", styles["Normal"]))
            doc.build(content)

        create_pdf()

        with open("secure_report.pdf", "rb") as f:
            st.download_button("📄 Download Secure Report", f)

    # ---------- HISTORY ----------
    st.subheader("📊 Secure History")

    c.execute("SELECT result, risk, timestamp FROM history WHERE username=?",
              (st.session_state.user,))
    data = c.fetchall()

    if data:
        df_hist = pd.DataFrame(data, columns=["Result", "Risk", "Time"])
        st.dataframe(df_hist)
