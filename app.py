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

# ---------- PREMIUM CSS ----------
st.markdown("""
<style>

/* Background */
.main {
    background: linear-gradient(135deg,#141e30,#243b55);
    color: white;
}

/* Header */
.header {
    background: linear-gradient(90deg,#ff416c,#ff4b2b);
    padding: 25px;
    border-radius: 18px;
    text-align: center;
    margin-bottom: 20px;
}

/* Cards */
.card {
    background: rgba(255,255,255,0.08);
    backdrop-filter: blur(10px);
    padding: 20px;
    border-radius: 15px;
    margin-bottom: 20px;
}

/* Button */
div.stButton > button {
    background: linear-gradient(90deg,#00c6ff,#0072ff);
    color: white;
    border-radius: 10px;
    height: 45px;
    font-size: 16px;
    border: none;
}

div.stButton > button:hover {
    transform: scale(1.05);
}

</style>
""", unsafe_allow_html=True)

# ---------- SESSION ----------
if "login" not in st.session_state:
    st.session_state.login = False

# ---------- LOGIN UI ----------
if not st.session_state.login:

    st.markdown("""
    <div class="header">
        <h1>🔐 Secure Login System</h1>
        <p>Access your AI Health Dashboard</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)

    choice = st.radio("Select Option", ["Login","Signup"])

    username = st.text_input("👤 Username")
    password = st.text_input("🔑 Password", type="password")

    if choice == "Signup":
        if st.button("🚀 Create Account"):
            add_user(username, password)
            st.success("Account created successfully")

    if choice == "Login":
        if st.button("🔓 Login"):
            user = login_user(username, password)
            if user:
                st.session_state.login = True
                st.session_state.user = username
                st.success("Login successful")
                st.rerun()
            else:
                st.error("Invalid credentials")

    st.markdown('</div>', unsafe_allow_html=True)

# ---------- MAIN APP ----------
else:

    st.sidebar.success(f"👋 {st.session_state.user}")
    if st.sidebar.button("Logout"):
        st.session_state.login = False
        st.rerun()

    st.markdown("""
    <div class="header">
        <h1>🩺 AI Diabetes Dashboard</h1>
        <p>Smart Health Risk Detection</p>
    </div>
    """, unsafe_allow_html=True)

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
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📋 Patient Details")

    col1, col2, col3 = st.columns(3)

    with col1:
        preg = st.number_input("🤰 Pregnancies", 0, 20, 1)
        glucose = st.number_input("🍬 Glucose", 0, 200, 100)

    with col2:
        bp = st.number_input("💓 Blood Pressure", 0, 150, 80)
        bmi = st.number_input("⚖️ BMI", 10.0, 50.0, 22.0)

    with col3:
        insulin = st.number_input("💉 Insulin", 0, 300, 80)
        age = st.number_input("🎂 Age", 1, 100, 30)

    skin = st.slider("🧬 Skin Thickness", 0, 100, 20)
    dpf = st.slider("🧬 Diabetes Pedigree Function", 0.0, 2.5, 0.5)

    st.markdown('</div>', unsafe_allow_html=True)

    # ---------- ANALYZE ----------
    if st.button("🚀 Analyze Health", use_container_width=True):

        input_data = np.array([[preg, glucose, bp, skin, insulin, bmi, dpf, age]])
        input_data = scaler.transform(input_data)

        pred = model.predict(input_data)
        prob = model.predict_proba(input_data)[0][1] * 100

        result = "High Risk" if pred[0]==1 else "Low Risk"

        save_result(st.session_state.user, result, prob)

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("🧾 Health Report")

        st.metric("🧠 Risk Score", f"{prob:.2f}%")
        st.progress(int(prob))

        if pred[0]==1:
            st.error("⚠️ High Risk Detected")
        else:
            st.success("✅ Low Risk")

        # CHART
        fig = go.Figure(go.Bar(x=["You","Normal"], y=[glucose,110]))
        st.plotly_chart(fig, use_container_width=True)

        # PDF
        def create_pdf():
            doc = SimpleDocTemplate("report.pdf")
            styles = getSampleStyleSheet()
            content = []
            content.append(Paragraph(f"Result: {result}", styles["Title"]))
            content.append(Paragraph(f"Risk: {prob:.2f}%", styles["Normal"]))
            doc.build(content)

        create_pdf()

        with open("report.pdf","rb") as f:
            st.download_button("📄 Download Report", f)

        st.markdown('</div>', unsafe_allow_html=True)

    # ---------- HISTORY ----------
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📊 Your History")

    c.execute("SELECT * FROM history WHERE username=?", (st.session_state.user,))
    data = c.fetchall()

    if data:
        df_hist = pd.DataFrame(data, columns=["User","Result","Risk"])
        st.dataframe(df_hist)

    st.markdown('</div>', unsafe_allow_html=True)
