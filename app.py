import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
import base64
from io import BytesIO
from fpdf import FPDF
from datetime import datetime

# Load model and label encoder
model = joblib.load("ppd_model_pipeline.pkl")
le = joblib.load("label_encoder.pkl")

# Set page config
st.set_page_config(page_title="PPD Risk Predictor", page_icon="🧠", layout="wide")

# Set background image via CSS
def add_animated_bg():
    st.markdown("""
<style>
@keyframes gradient {
    0% {background-color: #000000;}
    25% {background-color: #111111;}
    50% {background-color: #222222;}
    75% {background-color: #111111;}
    100% {background-color: #000000;}
}
.stApp {
    animation: gradient 12s ease-in-out infinite;
    background-size: cover;
    background-repeat: no-repeat;
}
</style>
""", unsafe_allow_html=True)

# Sidebar navigation using session_state with key
if "page" not in st.session_state:
    st.session_state.page = "🏠 Home"

st.session_state.page = st.sidebar.radio(
    "Navigate",
    ["🏠 Home", "📝 Take Test", "📊 Result Explanation", "📬 Feedback", "🧰 Resources"],
    index=["🏠 Home", "📝 Take Test", "📊 Result Explanation", "📬 Feedback", "🧰 Resources"].index(st.session_state.page),
    key="menu"
)

menu = st.session_state.page

# Apply section-specific background images
section_backgrounds = {
    "🏠 Home": "background.jpg",
    "📝 Take Test": "test_bg.jpg",
    "📊 Result Explanation": "result_bg.jpg",
    "📬 Feedback": "feedback_bg.jpg",
    "🧰 Resources": "resources_bg.jpg"
}

add_animated_bg()

# === HEADER BAR ===
st.markdown("""
<style>
.header {
    background-color: #0f0f0f;
    padding: 15px 30px;
    color: white;
    font-size: 24px;
    font-weight: bold;
    text-align: center;
    position: sticky;
    top: 0;
    z-index: 9999;
    border-bottom: 2px solid #444;
}
</style>
<div class="header">
    🧠 POSTPARTUM DEPRESSION RISK PREDICTOR
</div>
""", unsafe_allow_html=True)

if menu == "🏠 Home":
    st.markdown("""
    <style>
.nav-btn {
    position: fixed;
    bottom: 30px;
    right: 30px;
    z-index: 9999;
}
.nav-btn button {
    padding: 12px 25px;
    font-size: 1rem;
    background-color: black;
    color: white;
    border: none;
    border-radius: 8px;
    box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.2);
    transition: 0.3s;
}
.nav-btn button:hover {
    background-color: #333;
    transform: scale(1.05);
}
</style>
<div class="nav-btn">
    <form>
        <button type="submit" onClick="window.location.reload();">📝 Take Test</button>
    </form>
</div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="text-align: center; padding: 40px 20px;">
        <h1 style="font-size: 3.5em; color: black; font-weight: 900; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; text-transform: uppercase; letter-spacing: 2px;">
            POSTPARTUM DEPRESSION RISK PREDICTOR
        </h1>
        <h3 style="font-size: 1.6em; color: black; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;">
            Empowering maternal health through smart technology
        </h3>
        <p style="font-size: 1.2em; color: #222222; max-width: 750px; margin: 20px auto; line-height: 1.6em;">
            This AI-powered application helps identify potential risk levels of postpartum depression
            based on user inputs through a guided questionnaire. Designed for awareness, not diagnosis.
        </p>
    </div>
""", unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align: center; padding: 40px 20px;">
        <h1 style="font-size: 3.5em; color: black; font-weight: 900; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; text-transform: uppercase; letter-spacing: 2px;">
            POSTPARTUM DEPRESSION RISK PREDICTOR
        </h1>
        <h3 style="font-size: 1.6em; color: black; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;">
            Empowering maternal health through smart technology
        </h3>
        <p style="font-size: 1.2em; color: #222222; max-width: 750px; margin: 20px auto; line-height: 1.6em;">
            This AI-powered application helps identify potential risk levels of postpartum depression
            based on user inputs through a guided questionnaire. Designed for awareness, not diagnosis.
        </p>
    </div>
""", unsafe_allow_html=True)

elif menu == "📝 Take Test":
    st.header("📝 Depression Questionnaire")

    if 'question_index' not in st.session_state:
        st.session_state.question_index = 0
        st.session_state.responses = []
        st.session_state.age = 25
        st.session_state.support = "Medium"

    if st.session_state.question_index == 0:
        st.session_state.age = st.slider("Age", 18, 45, st.session_state.age)

    q_responses = [
        ("I have been able to laugh and see the funny side of things.",
         {"As much as I always could": 0, "Not quite so much now": 1, "Definitely not so much now": 2, "Not at all": 3}),

        ("I have looked forward with enjoyment to things",
         {"As much as I ever did": 0, "Rather less than I used to": 1, "Definitely less than I used to": 2, "Hardly at all": 3}),

        ("I have blamed myself unnecessarily when things went wrong",
         {"No, never": 0, "Not very often": 1, "Yes, some of the time": 2, "Yes, most of the time": 3}),

        ("I have been anxious or worried for no good reason",
         {"No, not at all": 0, "Hardly ever": 1, "Yes, sometimes": 2, "Yes, very often": 3}),

        ("I have felt scared or panicky for no very good reason",
         {"No, not at all": 0, "No, not much": 1, "Yes, sometimes": 2, "Yes, quite a lot": 3}),

        ("Things have been getting on top of me",
         {"No, I have been coping as well as ever": 0, "No, most of the time I have coped quite well": 1,
          "Yes, sometimes I haven't been coping as well as usual": 2,
          "Yes, most of the time I haven't been able to cope at all": 3}),

        ("I have been so unhappy that I have had difficulty sleeping",
         {"No, not at all": 0, "Not very often": 1, "Yes, sometimes": 2, "Yes, most of the time": 3}),

        ("I have felt sad or miserable",
         {"No, not at all": 0, "Not very often": 1, "Yes, quite often": 2, "Yes, most of the time": 3}),

        ("I have been so unhappy that I have been crying",
         {"No, never": 0, "Only occasionally": 1, "Yes, quite often": 2, "Yes, most of the time": 3}),

        ("The thought of harming myself has occurred to me",
         {"Never": 0, "Hardly ever": 1, "Sometimes": 2, "Yes, quite often": 3})
    ]

    idx = st.session_state.question_index

    if 1 <= idx <= 10:
        q_text, options = q_responses[idx - 1]
        choice = st.radio(f"{idx}. {q_text}", list(options.keys()), key=f"q{idx}")
        if st.button("Next"):
            st.session_state.responses.append(options[choice])
            st.session_state.question_index += 1

    elif idx == 0:
        if st.button("Start Questionnaire"):
            st.session_state.question_index += 1

    elif idx == 11:
        age = st.session_state.age
        support = "Medium"  # Default or placeholder value, not asked in UI
        q_values = st.session_state.responses
        score = sum(q_values)

        input_df = pd.DataFrame([{
            "Age": age,
            "FamilySupport": support,
            **{f"Q{i+1}": val for i, val in enumerate(q_values)},
            "EPDS_Score": score
        }])

        pred_encoded = model.predict(input_df)[0]
        pred_label = le.inverse_transform([pred_encoded])[0]

        st.success(f"Your Predicted Risk: **{pred_label}**")
        st.markdown("""
        <div style='background-color: white; padding: 20px; border-radius: 10px; margin-top: 20px;'>
            <h4 style='color: black;'>📄 Download Result</h4>
        </div>
        """, unsafe_allow_html=True)

        def create_pdf():
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=12)
            pdf.set_text_color(33, 33, 33)
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            pdf.cell(200, 10, txt="Postpartum Depression Risk Report", ln=True, align='C')
            pdf.ln(10)
            pdf.cell(200, 10, txt=f"Timestamp: {now}", ln=True)
            pdf.cell(200, 10, txt=f"Predicted Risk: {pred_label}", ln=True)
            pdf.cell(200, 10, txt=f"EPDS Score: {score}", ln=True)
            pdf.ln(10)
            for i, val in enumerate(q_values):
                pdf.cell(200, 10, txt=f"Q{i+1}: {val}", ln=True)
            pdf.ln(10)
            pdf.cell(200, 10, txt="Thank you for using the predictor.", ln=True)
            return pdf.output(dest='S').encode('latin-1')

        pdf_bytes = create_pdf()
        b64_pdf = base64.b64encode(pdf_bytes).decode()
        href = f'<a href="data:application/octet-stream;base64,{b64_pdf}" download="PPD_Risk_Report.pdf">📥 Download PDF Report</a>'
        st.markdown(href, unsafe_allow_html=True)

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=pred_encoded,
            gauge={"axis": {"range": [0, 3]},
                   "bar": {"color": "crimson"},
                   "steps": [
                       {"range": [0, 1], "color": "lightgreen"},
                       {"range": [1, 2], "color": "gold"},
                       {"range": [2, 3], "color": "orange"}
                   ]},
            title={"text": "Risk Level"}
        ))
        st.plotly_chart(fig, use_container_width=True)

        if st.button("Reset"):
            st.session_state.question_index = 0
            st.session_state.responses = []

elif menu == "📊 Result Explanation":
    if st.button("🏠 Go to Home"):
        st.session_state.page = "🏠 Home"
        st.rerun()
    st.header("Understanding Risk Levels")
    st.markdown("""
    | Risk Level | Meaning |
    |------------|---------|
    | Mild (0)   | Normal ups and downs |
    | Moderate (1) | Requires monitoring |
    | Severe (2) | Suggests possible clinical depression |
    | Profound (3) | Needs professional help urgently |
    """)

elif menu == "📬 Feedback":
    if st.button("📊 View Risk Explanation"):
        st.session_state.page = "📊 Result Explanation"
        st.rerun()
    st.header("📬 Share Feedback")
    name = st.text_input("Your Name")
    message = st.text_area("Your Feedback")
    if st.button("Submit"):
        st.success("Thank you for your feedback!")

elif menu == "🧰 Resources":
    if st.button("🏠 Back to Home"):
        st.session_state.page = "🏠 Home"
        st.rerun()
    st.header("Helpful Links and Support")
    st.markdown("""
    - [📞 National Mental Health Helpline - 1800-599-0019](https://www.mohfw.gov.in)
    - [🌐 WHO Maternal Mental Health](https://www.who.int/news-room/fact-sheets/detail/mental-health-of-women-during-pregnancy-and-after-childbirth)
    - [📝 Postpartum Support International](https://www.postpartum.net/)
    """)


