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

c.execute("""CREATE TABLE IF NOT EXISTS users(username TEXT,password TEXT)""")
c.execute("""CREATE TABLE IF NOT EXISTS history(username TEXT,result TEXT,risk REAL)""")

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
st.set_page_config(page_title="AI Dashboard", layout="wide")

# ---------- SUPER UI CSS ----------
st.markdown("""
<style>

/* Background */
.main {
    background: linear-gradient(135deg,#0f0c29,#302b63,#24243e);
    color:white;
}

/* Title Glow */
.title {
    font-size:40px;
    font-weight:700;
    text-align:center;
    background: linear-gradient(90deg,#00f2fe,#4facfe);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

/* Card */
.card {
    background: rgba(255,255,255,0.08);
    backdrop-filter: blur(20px);
    border-radius: 20px;
    padding:20px;
    margin-bottom:20px;
    box-shadow:0 10px 30px rgba(0,0,0,0.3);
}

/* Metric Card */
.metric {
    background: linear-gradient(135deg,#667eea,#764ba2);
    padding:20px;
    border-radius:15px;
    text-align:center;
    color:white;
}

/* Button */
div.stButton > button {
    background: linear-gradient(90deg,#ff00cc,#3333ff);
    color:white;
    border-radius:12px;
    height:50px;
    font-size:18px;
    border:none;
}

div.stButton > button:hover {
    transform:scale(1.05);
}

</style>
""", unsafe_allow_html=True)

# ---------- SESSION ----------
if "login" not in st.session_state:
    st.session_state.login=False

# ---------- LOGIN ----------
if not st.session_state.login:

    st.markdown('<div class="title">🔐 Welcome to AI Health System</div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["Login","Signup"])

    with tab1:
        u = st.text_input("👤 Username")
        p = st.text_input("🔑 Password", type="password")

        if st.button("Login 🚀"):
            if login_user(u,p):
                st.session_state.login=True
                st.session_state.user=u
                st.rerun()
            else:
                st.error("Invalid credentials")

    with tab2:
        u = st.text_input("New Username")
        p = st.text_input("New Password", type="password")

        if st.button("Create Account"):
            add_user(u,p)
            st.success("Account created")

    st.markdown('</div>', unsafe_allow_html=True)

# ---------- MAIN ----------
else:

    st.sidebar.success(f"👋 {st.session_state.user}")
    if st.sidebar.button("Logout"):
        st.session_state.login=False
        st.rerun()

    st.markdown('<div class="title">🩺 AI Diabetes Dashboard</div>', unsafe_allow_html=True)

    # ---------- LOAD DATA ----------
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
    st.subheader("📋 Patient Inputs")

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
    if st.button("🚀 Analyze Health", use_container_width=True):

        input_data = np.array([[preg,glucose,bp,skin,insulin,bmi,dpf,age]])
        input_data = scaler.transform(input_data)

        pred = model.predict(input_data)
        prob = model.predict_proba(input_data)[0][1]*100

        result = "High Risk" if pred[0]==1 else "Low Risk"
        save_result(st.session_state.user,result,prob)

        # ---------- TOP METRICS ----------
        colA,colB,colC = st.columns(3)

        with colA:
            st.markdown(f'<div class="metric">🧠 Risk<br><h2>{prob:.1f}%</h2></div>', unsafe_allow_html=True)

        with colB:
            st.markdown(f'<div class="metric">🍬 Glucose<br><h2>{glucose}</h2></div>', unsafe_allow_html=True)

        with colC:
            st.markdown(f'<div class="metric">⚖️ BMI<br><h2>{bmi}</h2></div>', unsafe_allow_html=True)

        # ---------- RESULT ----------
        st.markdown('<div class="card">', unsafe_allow_html=True)

        st.progress(int(prob))

        if pred[0]==1:
            st.error("⚠️ High Risk Detected")
        else:
            st.success("✅ Healthy")

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
    st.subheader("📊 History")

    c.execute("SELECT * FROM history WHERE username=?", (st.session_state.user,))
    data = c.fetchall()

    if data:
        st.dataframe(pd.DataFrame(data, columns=["User","Result","Risk"]))

    st.markdown('</div>', unsafe_allow_html=True)
