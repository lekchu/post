if menu == "🏠 Home":
    st.markdown("""
    <style>
    .stApp {
        animation: fadeBg 10s ease-in-out infinite;
        background-color: #001f3f;
    }
    @keyframes fadeBg {
        0% { background-color: #001f3f; }
        50% { background-color: #002b5c; }
        100% { background-color: #001f3f; }
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="text-align: center; padding: 40px 20px;">
        <h1 style="font-size: 2.8em; color: white;">Postpartum Depression Risk Predictor</h1>
        <p style="font-size: 1.1em; color: #ccc;">
            Based on the validated Edinburgh Postnatal Depression Scale (EPDS)
        </p>
        <p style="font-size: 1em; color: #bbb; max-width: 700px; margin: 20px auto;">
            This AI-powered tool helps assess the risk of postpartum depression based on your responses. It is for awareness purposes only and not a substitute for professional diagnosis.
        </p>
    </div>
    """, unsafe_allow_html=True)

    if st.button("📝 Start Test"):
        st.session_state.page = "📝 Take Test"
        st.rerun()


