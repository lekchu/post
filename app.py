import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from fpdf import FPDF
import base64

# Load model and label encoder
model = joblib.load("ppd_model_pipeline.pkl")
le = joblib.load("label_encoder.pkl")

# Set page config
st.set_page_config(page_title="PPD Risk Predictor", page_icon="🧠", layout="wide")

# Animated dark blue background
def add_page_animation():
    st.markdown("""
    <style>
    .stApp {
        animation: fadeBg 10s ease-in-out infinite;
        background-color: #001f3f;
    }
    @keyframes fadeBg {
        0%   { background-color: #001f3f; }
        50%  { background-color: #003366; }
        100% { background-color: #001f3f; }
    }
    </style>
    """, unsafe_allow_html=True)

add_page_animation()

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

# HOME
if menu == "🏠 Home":
    st.markdown("""
    <div style="text-align: center; padding: 40px;">
        <h1 style="font-size: 3em; color: white;">Postpartum Depression Risk Predictor</h1>
        <h3 style="color: white;">Based on the Edinburgh Postnatal Depression Scale (EPDS)</h3>
        <p style="color: #ccc; font-size: 1.1em;">
            This AI-powered tool uses machine learning to predict the risk level of postpartum depression. 
            Please note, this tool is for awareness purposes only—not a medical diagnosis.
        </p>
    </div>
    """, unsafe_allow_html=True)

    if st.button("📝 Start Test"):
        st.session_state.page = "📝 Take Test"
        st.rerun()

# TAKE TEST
elif menu == "📝 Take Test":
    st.header("📝 EPDS Depression Questionnaire")

    if 'question_index' not in st.session_state:
        st.session_state.question_index = 0
        st.session_state.responses = []
        st.session_state.name = ""
        st.session_state.age = 25
        st.session_state.place = ""
        st.session_state.support = "Medium"

    if st.session_state.question_index == 0:
        st.session_state.name = st.text_input("Your First Name")
        st.session_state.age = st.slider("Your Age", 18, 45, st.session_state.age)
        st.session_state.place = st.text_input("Your Place")
        st.session_state.support = st.selectbox("Family Support Level", ["High", "Medium", "Low"], index=1)
        if st.button("Start Questionnaire"):
            if st.session_state.name.strip():
                st.session_state.question_index = 1
                st.rerun()
            else:
                st.warning("Please enter your name before continuing.")

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
          "Yes, sometimes I haven't been coping as well as usual": 2, "Yes, most of the time I haven't been able to cope at all": 3}),
        ("I have been so unhappy that I have had difficulty sleeping",
         {"No, not at all": 0, "Not very often": 1, "Yes, sometimes": 2, "Yes, most of the time": 3}),
        ("I have felt sad or miserable",
         {"No, not at all": 0, "Not very often": 1, "Yes, quite often": 2, "Yes, most of the time": 3}),
        ("I have been so unhappy that I have been crying",
         {"No, never": 0, "Only occasionally": 1, "Yes, quite often": 2, "Yes, most of the time": 3}),
        ("The thought of harming myself has occurred to me",
         {"Never": 0, "Hardly ever": 1, "Sometimes": 2, "Yes, quite often": 3}),
    ]

    idx = st.session_state.question_index
    if 1 <= idx <= 10:
        q_text, options = q_responses[idx - 1]
        choice = st.radio(f"{idx}. {q_text}", list(options.keys()), key=f"q{idx}")
        col1, col2 = st.columns(2)
        if col1.button("⬅️ Back") and idx > 1:
            st.session_state.question_index -= 1
            st.session_state.responses.pop()
            st.rerun()
        if col2.button("Next ➡️"):
            st.session_state.responses.append(options[choice])
            st.session_state.question_index += 1
            st.rerun()

    elif idx == 11:
        name = st.session_state.name
        age = st.session_state.age
        place = st.session_state.place
        support = st.session_state.support
        answers = st.session_state.responses
        score = sum(answers)

        input_df = pd.DataFrame([{
            "Name": name,
            "Age": age,
            "FamilySupport": support,
            **{f"Q{i+1}": val for i, val in enumerate(answers)},
            "EPDS_Score": score
        }])

        # Prediction
        pred_encoded = model.predict(input_df.drop(columns=["Name"]))[0]
        pred_label = le.inverse_transform([pred_encoded])[0]

        st.success(f"🧠 {name}, your predicted PPD Risk is: **{pred_label}**")

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

        # SHAP Explanation
        st.subheader("📊 Explanation of Model Decision")
        classifier = model.named_steps['classifier']
        preprocessor = model.named_steps['preprocess']
        X_transformed = preprocessor.transform(input_df.drop(columns=["Name"]))
        explainer = shap.Explainer(classifier.predict, X_transformed)
        shap_values = explainer(X_transformed)
        fig, ax = plt.subplots()
        shap.plots.bar(shap_values[0], show=False)
        st.pyplot(fig)

        # PDF Report
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt="Postpartum Depression Risk Report", ln=True, align='C')
        pdf.cell(200, 10, txt=f"Name: {name}", ln=True)
        pdf.cell(200, 10, txt=f"Age: {age}", ln=True)
        pdf.cell(200, 10, txt=f"Place: {place}", ln=True)
        pdf.cell(200, 10, txt=f"Family Support: {support}", ln=True)
        pdf.cell(200, 10, txt=f"EPDS Total Score: {score}", ln=True)
        pdf.cell(200, 10, txt=f"Predicted Risk: {pred_label}", ln=True)
        output_file = f"{name.replace(' ', '_')}_PPD_Result.pdf"
        pdf.output(output_file)
        with open(output_file, "rb") as file:
            b64 = base64.b64encode(file.read()).decode()
            st.markdown(f'<a href="data:application/octet-stream;base64,{b64}" download="{output_file}">📥 Download PDF Report</a>', unsafe_allow_html=True)

        if st.button("🔁 Retake Test"):
            for key in ["question_index", "responses", "name", "age", "place", "support"]:
                st.session_state.pop(key, None)
            st.rerun()

# EXPLANATION PAGE
elif menu == "📊 Result Explanation":
    st.header("📊 Understanding Risk Levels")
    st.markdown("""
    | Risk Level | Meaning |
    |------------|---------|
    | **Mild (0)**     | Normal emotional variation |
    | **Moderate (1)** | Needs observation, possible support |
    | **Severe (2)**   | Suggests clinical symptoms |
    | **Profound (3)** | Immediate professional help advised |
    """, unsafe_allow_html=True)

# FEEDBACK
elif menu == "📬 Feedback":
    st.header("📬 Share Feedback")
    name = st.text_input("Your Name")
    message = st.text_area("Your Feedback")
    if st.button("Submit"):
        st.success("Thank you for your valuable feedback! 💌")

# RESOURCES
elif menu == "🧰 Resources":
    st.header("🔗 Resources & Support")
    st.markdown("""
    - [🧠 WHO Maternal Mental Health](https://www.who.int/news-room/fact-sheets/detail/mental-health-of-women-during-pregnancy-and-after-childbirth)
    - [📞 National Mental Health Helpline (India): 1800-599-0019](https://www.mohfw.gov.in)
    - [🌐 Postpartum Support International](https://www.postpartum.net/)
    """)
