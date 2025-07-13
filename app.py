import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
import shap
import matplotlib.pyplot as plt
from fpdf import FPDF
import base64

# Load model and encoder
model = joblib.load("ppd_model_pipeline.pkl")
le = joblib.load("label_encoder.pkl")

# Set Streamlit page config
st.set_page_config(page_title="PPD Risk Predictor", page_icon="🧠", layout="wide")

# Background animation
def add_bg_animation():
    st.markdown("""
        <style>
        .stApp {
            animation: bgColorChange 15s infinite alternate;
            background-color: #001f3f;
        }
        @keyframes bgColorChange {
            0%   {background-color: #001f3f;}
            50%  {background-color: #003366;}
            100% {background-color: #001f3f;}
        }
        </style>
    """, unsafe_allow_html=True)

add_bg_animation()

# Sidebar navigation
if "page" not in st.session_state:
    st.session_state.page = "🏠 Home"

st.session_state.page = st.sidebar.radio(
    "Navigate",
    ["🏠 Home", "📝 Take Test", "📊 Result Explanation", "📬 Feedback", "🧰 Resources"],
    index=["🏠 Home", "📝 Take Test", "📊 Result Explanation", "📬 Feedback", "🧰 Resources"].index(st.session_state.page),
    key="menu"
)

menu = st.session_state.page

# HOME PAGE
if menu == "🏠 Home":
    st.markdown("""
    <div style="text-align:center; padding: 50px;">
        <h1 style="color:white;">Postpartum Depression Risk Predictor</h1>
        <h3 style="color:white;">Based on the Edinburgh Postnatal Depression Scale (EPDS)</h3>
        <p style="color:white; max-width:700px; margin:auto;">
        This AI-powered tool helps assess postpartum depression risk. 
        It's built on the trusted EPDS screening scale and enhanced with explainable AI.
        </p>
    </div>
    """, unsafe_allow_html=True)

    if st.button("📝 Start Test"):
        st.session_state.page = "📝 Take Test"
        st.experimental_rerun()

# TAKE TEST
elif menu == "📝 Take Test":
    st.header("📝 Depression Questionnaire")

    if "question_index" not in st.session_state:
        st.session_state.question_index = 0
        st.session_state.responses = []
        st.session_state.name = ""
        st.session_state.age = 25
        st.session_state.place = ""
        st.session_state.support = "Medium"

    if st.session_state.question_index == 0:
        st.session_state.name = st.text_input("Your Name")
        st.session_state.age = st.slider("Your Age", 18, 45, st.session_state.age)
        st.session_state.place = st.text_input("Your Place")
        st.session_state.support = st.selectbox("Family Support Level", ["High", "Medium", "Low"], index=1)

        if st.button("Start Questionnaire"):
            if st.session_state.name.strip() != "":
                st.session_state.question_index += 1
                st.experimental_rerun()
            else:
                st.warning("Please enter your name before starting.")

    q_responses = [
        ("I have been able to laugh and see the funny side of things.", {
            "As much as I always could": 0, "Not quite so much now": 1, "Definitely not so much now": 2, "Not at all": 3}),
        ("I have looked forward with enjoyment to things", {
            "As much as I ever did": 0, "Rather less than I used to": 1, "Definitely less than I used to": 2, "Hardly at all": 3}),
        ("I have blamed myself unnecessarily when things went wrong", {
            "No, never": 0, "Not very often": 1, "Yes, some of the time": 2, "Yes, most of the time": 3}),
        ("I have been anxious or worried for no good reason", {
            "No, not at all": 0, "Hardly ever": 1, "Yes, sometimes": 2, "Yes, very often": 3}),
        ("I have felt scared or panicky for no very good reason", {
            "No, not at all": 0, "No, not much": 1, "Yes, sometimes": 2, "Yes, quite a lot": 3}),
        ("Things have been getting on top of me", {
            "No, I have been coping as well as ever": 0, "No, most of the time I have coped quite well": 1,
            "Yes, sometimes I haven't been coping as well as usual": 2, "Yes, most of the time I haven't been able to cope at all": 3}),
        ("I have been so unhappy that I have had difficulty sleeping", {
            "No, not at all": 0, "Not very often": 1, "Yes, sometimes": 2, "Yes, most of the time": 3}),
        ("I have felt sad or miserable", {
            "No, not at all": 0, "Not very often": 1, "Yes, quite often": 2, "Yes, most of the time": 3}),
        ("I have been so unhappy that I have been crying", {
            "No, never": 0, "Only occasionally": 1, "Yes, quite often": 2, "Yes, most of the time": 3}),
        ("The thought of harming myself has occurred to me", {
            "Never": 0, "Hardly ever": 1, "Sometimes": 2, "Yes, quite often": 3})
    ]

    idx = st.session_state.question_index
    if 1 <= idx <= 10:
        q_text, options = q_responses[idx - 1]
        choice = st.radio(f"{idx}. {q_text}", list(options.keys()), key=f"q{idx}")
        col1, col2 = st.columns(2)
        if col1.button("⬅️ Back") and idx > 1:
            st.session_state.question_index -= 1
            st.session_state.responses.pop()
            st.experimental_rerun()
        if col2.button("Next ➡️"):
            st.session_state.responses.append(options[choice])
            st.session_state.question_index += 1
            st.experimental_rerun()

    elif idx == 11:
        name = st.session_state.name
        age = st.session_state.age
        place = st.session_state.place
        support = st.session_state.support
        q_values = st.session_state.responses
        score = sum(q_values)

        input_df = pd.DataFrame([{
            "Name": name,
            "Age": age,
            "Place": place,
            "FamilySupport": support,
            **{f"Q{i+1}": val for i, val in enumerate(q_values)},
            "EPDS_Score": score
        }])

        pred_encoded = model.predict(input_df.drop(columns=["Name", "Place"]))[0]
        pred_label = le.inverse_transform([pred_encoded])[0]

        st.success(f"{name}, your predicted PPD Risk is: **{pred_label}**")

        # Gauge chart
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=pred_encoded,
            number={"suffix": " / 3"},
            gauge={
                "axis": {"range": [0, 3]},
                "bar": {"color": "deeppink"},
                "steps": [
                    {"range": [0, 1], "color": "lightgreen"},
                    {"range": [1, 2], "color": "gold"},
                    {"range": [2, 3], "color": "red"}
                ]
            },
            title={"text": "Risk Level"}
        ))
        st.plotly_chart(fig, use_container_width=True)

        # SHAP Explainability
        try:
            explainer = shap.Explainer(model.named_steps["classifier"])
            shap_values = explainer(pd.DataFrame(input_df.drop(columns=["Name", "Place"])))
            st.subheader("📊 Explanation of Model Decision")
            st.set_option('deprecation.showPyplotGlobalUse', False)
            shap.plots.bar(shap_values[0], show=False)
            st.pyplot(bbox_inches="tight")
        except Exception as e:
            st.warning(f"Explainability failed: {e}")

        # Suggestions based on risk
        st.info("📌 **Tips Based on Risk Level**")
        if pred_encoded == 0:
            st.markdown("✅ You're doing well! Keep practicing self-care and stay connected.")
        elif pred_encoded == 1:
            st.markdown("⚠️ You might be experiencing mild symptoms. Talk to someone you trust or a counselor.")
        elif pred_encoded == 2:
            st.markdown("⚠️ Moderate risk. We recommend speaking with a mental health professional.")
        else:
            st.markdown("🚨 High risk. Please seek professional support immediately.")

        # PDF result
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt="PPD Risk Prediction Report", ln=True, align='C')
        pdf.cell(200, 10, txt=f"Name: {name}", ln=True)
        pdf.cell(200, 10, txt=f"Age: {age}", ln=True)
        pdf.cell(200, 10, txt=f"Place: {place}", ln=True)
        pdf.cell(200, 10, txt=f"Family Support: {support}", ln=True)
        pdf.cell(200, 10, txt=f"EPDS Score: {score}", ln=True)
        pdf.cell(200, 10, txt=f"Predicted Risk Level: {pred_label}", ln=True)

        file_path = f"{name.replace(' ', '_')}_PPD_Report.pdf"
        pdf.output(file_path)

        with open(file_path, "rb") as file:
            b64 = base64.b64encode(file.read()).decode()
            st.markdown(f'<a href="data:application/pdf;base64,{b64}" download="{file_path}">📄 Download Your Report</a>', unsafe_allow_html=True)

        if st.button("🔁 Restart"):
            for key in ["question_index", "responses", "name", "age", "place", "support"]:
                st.session_state.pop(key, None)
            st.experimental_rerun()

# RESULT EXPLANATION
elif menu == "📊 Result Explanation":
    st.header("📊 EPDS Risk Levels Explained")
    st.markdown("""
    | Score | Risk Level | Explanation |
    |-------|------------|-------------|
    | 0     | Mild       | Normal emotional changes |
    | 1     | Moderate   | Possible mood swings, needs monitoring |
    | 2     | Severe     | Likely clinical depression symptoms |
    | 3     | Profound   | Immediate clinical help recommended |
    """)

# FEEDBACK
elif menu == "📬 Feedback":
    st.header("📬 Share Feedback")
    user = st.text_input("Your Name")
    msg = st.text_area("Your Feedback")
    if st.button("Submit"):
        st.success("Thank you! 💬")

# RESOURCES
elif menu == "🧰 Resources":
    st.header("🧰 Mental Health Resources")
    st.markdown("""
    - [📞 National Helpline - 1800-599-0019](https://www.mohfw.gov.in)
    - [🌐 WHO: Maternal Mental Health](https://www.who.int)
    - [📝 Postpartum Support Intl](https://www.postpartum.net/)
    """)

