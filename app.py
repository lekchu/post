
# Now prepare the final app.py including all requested features
import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
from fpdf import FPDF
import base64

# Load model and label encoder
model = joblib.load("ppd_model_pipeline.pkl")
le = joblib.load("label_encoder.pkl")

# Set page config
st.set_page_config(page_title="PPD Risk Predictor", page_icon="🧠", layout="wide")

# Background animations for each section
def add_page_animation(page):
    styles = {
        "🏠 Home": "#ffe6f0",
        "📝 Take Test": "#e6f7ff",
        "📊 Result Explanation": "#e6ffe6",
        "📬 Feedback": "#fff5e6",
        "🧰 Resources": "#f2e6ff"
    }
    color = styles.get(page, "#ffffff")
    st.markdown(f'''
    <style>
    .stApp {{
        animation: fadeBg 10s ease-in-out infinite;
        background-color: {color};
    }}
    @keyframes fadeBg {{
        0% {{ background-color: white; }}
        50% {{ background-color: {color}; }}
        100% {{ background-color: white; }}
    }}
    </style>
    ''', unsafe_allow_html=True)

# Sidebar page state
if "page" not in st.session_state:
    st.session_state.page = "🏠 Home"

# Sidebar navigation
st.session_state.page = st.sidebar.radio(
    "Navigate",
    ["🏠 Home", "📝 Take Test", "📊 Result Explanation", "📬 Feedback", "🧰 Resources"],
    index=["🏠 Home", "📝 Take Test", "📊 Result Explanation", "📬 Feedback", "🧰 Resources"].index(st.session_state.page),
    key="menu"
)

menu = st.session_state.page
add_page_animation(menu)

# HOME
if menu == "🏠 Home":
    st.markdown(\"""
    <div style="text-align: center; padding: 40px 20px;">
        <h1 style="font-size: 3.5em; color: black;">POSTPARTUM DEPRESSION RISK PREDICTOR</h1>
        <h3 style="font-size: 1.6em; color: black;">Empowering maternal health through smart technology</h3>
        <p style="font-size: 1.2em; color: #222; max-width: 750px; margin: 20px auto;">
            This AI-powered app helps identify potential risk levels of postpartum depression
            based on user inputs. For awareness, not diagnosis.
        </p>
    </div>
    \""", unsafe_allow_html=True)

    if st.button("📝 Start Test"):
        st.session_state.page = "📝 Take Test"
        st.rerun()

# TEST
elif menu == "📝 Take Test":
    st.header("📝 Depression Questionnaire")

    if 'question_index' not in st.session_state:
        st.session_state.question_index = 0
        st.session_state.responses = []
        st.session_state.age = 25
        st.session_state.support = "Medium"
        st.session_state.name = ""

    if st.session_state.question_index == 0:
        st.session_state.name = st.text_input("Your Name")
        st.session_state.age = st.slider("Your Age", 18, 45, st.session_state.age)
        st.session_state.support = st.selectbox("Level of Family Support", ["High", "Medium", "Low"], index=1)
        if st.button("Start Questionnaire"):
            if st.session_state.name.strip() != "":
                st.session_state.question_index += 1
            else:
                st.warning("Please enter your name before starting.")

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
         {"Never": 0, "Hardly ever": 1, "Sometimes": 2, "Yes, quite often": 3})
    ]

    idx = st.session_state.question_index
    if 1 <= idx <= 10:
        q_text, options = q_responses[idx - 1]
        choice = st.radio(f"{idx}. {q_text}", list(options.keys()), key=f"q{idx}")
        col1, col2 = st.columns(2)
        if col1.button("⬅️ Back") and idx > 1:
            st.session_state.question_index -= 1
            st.session_state.responses.pop()
        if col2.button("Next ➡️"):
            st.session_state.responses.append(options[choice])
            st.session_state.question_index += 1

    elif idx == 11:
        name = st.session_state.name
        age = st.session_state.age
        support = st.session_state.support
        q_values = st.session_state.responses
        score = sum(q_values)

        input_df = pd.DataFrame([{
            "Name": name,
            "Age": age,
            "FamilySupport": support,
            **{f"Q{i+1}": val for i, val in enumerate(q_values)},
            "EPDS_Score": score
        }])

        pred_encoded = model.predict(input_df.drop(columns=["Name"]))[0]
        pred_label = le.inverse_transform([pred_encoded])[0]

        st.success(f"{name}, your predicted PPD Risk is: **{pred_label}**")

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

        # PDF Generation
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt="Postpartum Depression Risk Prediction", ln=True, align='C')
        pdf.cell(200, 10, txt=f"Name: {name}", ln=True)
        pdf.cell(200, 10, txt=f"Age: {age}", ln=True)
        pdf.cell(200, 10, txt=f"Support Level: {support}", ln=True)
        pdf.cell(200, 10, txt=f"Total Score: {score}", ln=True)
        pdf.cell(200, 10, txt=f"Predicted Risk Level: {pred_label}", ln=True)

        pdf_output = f"{name.replace(' ', '_')}_PPD_Result.pdf"
        pdf.output(pdf_output)
        with open(pdf_output, "rb") as file:
            b64_pdf = base64.b64encode(file.read()).decode('utf-8')
            href = f'<a href="data:application/pdf;base64,{b64_pdf}" download="{pdf_output}">📥 Download Your Result (PDF)</a>'
            st.markdown(href, unsafe_allow_html=True)

        if st.button("🔄 Restart"):
            for key in ["question_index", "responses", "age", "support", "name"]:
                st.session_state.pop(key, None)
            st.experimental_rerun()
"""

# Save final app.py
with open("/mnt/data/app.py", "w") as f:
    f.write(app_code.strip())

"/mnt/data/app.py"




