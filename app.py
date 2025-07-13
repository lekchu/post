import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go

# Load model and label encoder
model = joblib.load("ppd_model_pipeline.pkl")
le = joblib.load("label_encoder.pkl")

# Page config
st.set_page_config(page_title="PPD Risk Predictor", page_icon="🧠", layout="wide")

# Add background animations
def add_animated_background(page):
    if page == "🏠 Home":
        st.markdown("""
        <style>
        @keyframes pinkPulse {
            0% {background-color: #ffe6f0;}
            25% {background-color: #ffc2d1;}
            50% {background-color: #ff99bb;}
            75% {background-color: #ffc2d1;}
            100% {background-color: #ffe6f0;}
        }
        .stApp {
            animation: pinkPulse 10s ease-in-out infinite;
        }
        </style>
        """, unsafe_allow_html=True)
    else:
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
        }
        </style>
        """, unsafe_allow_html=True)

# Sidebar state
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
add_animated_background(menu)

# 🏠 HOME PAGE
if menu == "🏠 Home":
    st.markdown("""
    <div style="text-align: center; padding: 40px 20px;">
        <h1 style="font-size: 3.5em; color: black;">POSTPARTUM DEPRESSION RISK PREDICTOR</h1>
        <h3 style="font-size: 1.6em; color: black;">Empowering maternal health through smart technology</h3>
        <p style="font-size: 1.2em; color: #222; max-width: 750px; margin: 20px auto;">
            This AI-powered app helps identify potential risk levels of postpartum depression
            based on user inputs. For awareness, not diagnosis.
        </p>
        <form action="?menu=📝 Take Test">
            <button style="padding: 14px 30px; background-color: #ff66a3; color: white; font-size: 1rem;
                    border: none; border-radius: 8px; cursor: pointer; box-shadow: 0 4px 10px rgba(0,0,0,0.2);">
                📝 Take Test
            </button>
        </form>
    </div>
    """, unsafe_allow_html=True)

# 📝 TEST PAGE
elif menu == "📝 Take Test":
    st.header("📝 Depression Questionnaire")

    if 'question_index' not in st.session_state:
        st.session_state.question_index = 0
        st.session_state.responses = []
        st.session_state.age = 25
        st.session_state.support = "Medium"

    if st.session_state.question_index == 0:
        st.session_state.age = st.slider("Age", 18, 45, st.session_state.age)
        st.session_state.support = st.selectbox("Level of Family Support", ["High", "Medium", "Low"], index=1)

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
        col1, col2 = st.columns([1, 1])
        if col1.button("⬅️ Back", key=f"back_{idx}") and idx > 1:
            st.session_state.question_index -= 1
            st.session_state.responses.pop()
        if col2.button("Next ➡️", key=f"next_{idx}"):
            st.session_state.responses.append(options[choice])
            st.session_state.question_index += 1

    elif idx == 0:
        if st.button("Start Questionnaire"):
            st.session_state.question_index += 1

    elif idx == 11:
        age = st.session_state.age
        support = st.session_state.support
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

        st.success(f"🎯 Your Predicted Risk: **{pred_label}**")

        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=pred_encoded,
            number={"suffix": " / 3"},
            gauge={
                "axis": {"range": [0, 3]},
                "bar": {"color": "deeppink"},
                "steps": [
                    {"range": [0, 1], "color": "lightgreen"},
                    {"range": [1, 2], "color": "gold"},
                    {"range": [2, 3], "color": "orangered"}
                ]
            },
            title={"text": "Risk Level Indicator"}
        ))
        st.plotly_chart(fig, use_container_width=True)

        if st.button("🔄 Restart"):
            st.session_state.question_index = 0
            st.session_state.responses = []

# 📊 RESULT EXPLANATION
elif menu == "📊 Result Explanation":
    if st.button("🏠 Go to Home"):
        st.session_state.page = "🏠 Home"
        st.rerun()
    st.header("Understanding Risk Levels")
    st.markdown("""
    | Risk Level | Description |
    |------------|-------------|
    | 0 - Mild   | Normal ups and downs |
    | 1 - Moderate | Requires monitoring |
    | 2 - Severe   | Signs of clinical depression |
    | 3 - Profound | Needs urgent professional support |
    """)

# 📬 FEEDBACK PAGE
elif menu == "📬 Feedback":
    if st.button("📊 View Result Explanation"):
        st.session_state.page = "📊 Result Explanation"
        st.rerun()
    st.header("📬 Share Feedback")
    name = st.text_input("Your Name")
    message = st.text_area("Your Feedback")
    if st.button("Submit"):
        st.success("Thank you for your valuable feedback! 💌")

# 🧰 RESOURCES PAGE
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





