import streamlit as st
import sqlite3
import bcrypt
import numpy as np
import pandas as pd
import random, smtplib, time
from email.mime.text import MIMEText
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import plotly.graph_objects as go
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet

# ---------- EMAIL CONFIG ----------
EMAIL = "your_email@gmail.com"
APP_PASSWORD = "your_app_password"

def send_otp(email, otp):
    msg = MIMEText(f"Your OTP is: {otp}")
    msg['Subject'] = "Login OTP"
    msg['From'] = EMAIL
    msg['To'] = email

    server = smtplib.SMTP_SSL('smtp.gmail.com', 465)
    server.login(EMAIL, APP_PASSWORD)
    server.send_message(msg)
    server.quit()

# ---------- DATABASE ----------
conn = sqlite3.connect("secure_users.db", check_same_thread=False)
c = conn.cursor()

c.execute("""CREATE TABLE IF NOT EXISTS users(
    username TEXT PRIMARY KEY,
    password BLOB
)""")

c.execute("""CREATE TABLE IF NOT EXISTS history(
    username TEXT,
    result TEXT,
    risk REAL,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
)""")

# ---------- SECURITY ----------
def hash_password(p):
    return bcrypt.hashpw(p.encode(), bcrypt.gensalt())

def verify_password(p, h):
    return bcrypt.checkpw(p.encode(), h)

def add_user(u, p):
    try:
        c.execute("INSERT INTO users VALUES (?,?)", (u, hash_password(p)))
        conn.commit()
        return True
    except:
        return False

def login_user(u, p):
    c.execute("SELECT password FROM users WHERE username=?", (u,))
    data = c.fetchone()
    return data and verify_password(p, data[0])

def save_result(u, r, prob):
    c.execute("INSERT INTO history(username,result,risk) VALUES (?,?,?)",(u,r,prob))
    conn.commit()

# ---------- SESSION ----------
if "login" not in st.session_state:
    st.session_state.login = False
if "otp" not in st.session_state:
    st.session_state.otp = None
if "otp_time" not in st.session_state:
    st.session_state.otp_time = 0

# ---------- PAGE ----------
st.set_page_config(page_title="Ultimate Secure AI Dashboard", layout="wide")

# ---------- LOGIN PAGE ----------
if not st.session_state.login:

    st.title("🔐 Secure Login System")

    tab1, tab2, tab3 = st.tabs(["🔑 Login", "🆕 Signup", "📧 OTP Login"])

    # LOGIN
    with tab1:
        u = st.text_input("Username")
        p = st.text_input("Password", type="password")

        if st.button("Login"):
            if login_user(u, p):
                st.session_state.login = True
                st.session_state.user = u
                st.success("Login successful")
                st.rerun()
            else:
                st.error("Invalid credentials")

    # SIGNUP
    with tab2:
        u = st.text_input("New Username")
        p = st.text_input("New Password", type="password")

        if st.button("Create Account"):
            if add_user(u, p):
                st.success("Account created")
            else:
                st.error("User already exists")

    # OTP LOGIN
    with tab3:
        email = st.text_input("Enter Email")

        if st.button("Send OTP"):
            otp = str(random.randint(100000,999999))
            st.session_state.otp = otp
            st.session_state.otp_time = time.time()

            send_otp(email, otp)
            st.success("OTP sent 📩")

        otp_input = st.text_input("Enter OTP")

        if st.button("Verify OTP"):
            if time.time() - st.session_state.otp_time > 300:
                st.error("OTP expired")
            elif otp_input == st.session_state.otp:
                st.session_state.login = True
                st.session_state.user = email
                st.success("OTP verified")
                st.rerun()
            else:
                st.error("Wrong OTP")

# ---------- MAIN APP ----------
else:

    st.sidebar.success(f"👋 {st.session_state.user}")
    if st.sidebar.button("Logout"):
        st.session_state.login = False
        st.rerun()

    st.title("🩺 AI Diabetes Dashboard")

    # ---------- DATA ----------
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
            st.error("⚠️ High Risk")
        else:
            st.success("✅ Low Risk")

        # CHART
        fig = go.Figure(go.Bar(x=["You","Normal"], y=[glucose,110]))
        st.plotly_chart(fig)

        # PDF
        def create_pdf():
            doc = SimpleDocTemplate("report.pdf")
            styles = getSampleStyleSheet()
            content = []
            content.append(Paragraph(f"User: {st.session_state.user}", styles["Title"]))
            content.append(Paragraph(f"Result: {result}", styles["Normal"]))
            content.append(Paragraph(f"Risk: {prob:.2f}%", styles["Normal"]))
            doc.build(content)

        create_pdf()

        with open("report.pdf","rb") as f:
            st.download_button("📄 Download Report", f)

    # ---------- HISTORY ----------
    st.subheader("📊 Your History")

    c.execute("SELECT result, risk, timestamp FROM history WHERE username=?", 
              (st.session_state.user,))
    data = c.fetchall()

    if data:
        st.dataframe(pd.DataFrame(data, columns=["Result","Risk","Time"]))
