# Enhanced Streamlit App for Postpartum Depression Risk Prediction
# Developed for MSc Computer Science Project

import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
import base64

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
        0% {background-position: 0% 50%;}
        50% {background-position: 100% 50%;}
        100% {background-position: 0% 50%;}
    }
    .stApp {
        background: linear-gradient(-45deg, #ff9a9e, #fad0c4, #fbc2eb, #a18cd1);
        background-size: 400% 400%;
        animation: gradient 15s ease infinite;
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
        background-color: #ff4b4b;
        color: white;
        border: none;
        border-radius: 8px;
        box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.2);
        transition: 0.3s;
    }
    .nav-btn button:hover {
        background-color: #e84343;
        transform: scale(1.05);
    }
    </style>
    <div class="nav-btn">
        <form action="#">
            <button onclick="window.location.href='#'" type="button" onClick="streamlitSend({type: 'setPage', page: '📝 Take Test'})">📝 Go to Questionnaire</button>
        </form>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="text-align: center; padding: 40px 20px;">
        <h1 style="font-size: 3em; color: #ffffff; font-weight: bold;">Postpartum Depression Risk Predictor</h1>
        <h3 style="font-size: 1.5em; color: #f0f0f0;">Empowering maternal health through smart technology</h3>
        <p style="font-size: 1.1em; color: #dddddd; max-width: 700px; margin: 20px auto;">
            This AI-powered application helps identify potential risk levels of postpartum depression
            based on user inputs through a guided questionnaire. Designed for awareness, not diagnosis.
        </p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align: center; padding: 40px 20px;">
        <h1 style="font-size: 3em; color: #ffffff; font-weight: bold;">Postpartum Depression Risk Predictor</h1>
        <h3 style="font-size: 1.5em; color: #f0f0f0;">Empowering maternal health through smart technology</h3>
        <p style="font-size: 1.1em; color: #dddddd; max-width: 700px; margin: 20px auto;">
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
