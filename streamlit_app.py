import streamlit as st
import requests

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Titanic Survival Predictor",
    page_icon="🚢",
    layout="centered"
)

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🚢 Titanic Survival Predictor")
st.markdown("**Would you have survived the Titanic?** Fill in the details below to find out.")
st.divider()

# ── FastAPI endpoint ──────────────────────────────────────────────────────────
API_URL = "http://localhost:8000/predict"  # change if deployed separately

# ── Input form ────────────────────────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    pclass = st.selectbox(
        "🎫 Passenger Class",
        options=[1, 2, 3],
        format_func=lambda x: f"Class {x} {'(First)' if x==1 else '(Second)' if x==2 else '(Third)'}",
        help="1st class = luxury; 3rd class = economy"
    )
    sex = st.radio("🧑 Gender", options=["male", "female"], horizontal=True)

with col2:
    age = st.slider("🎂 Age", min_value=1, max_value=80, value=28)
    fare = st.number_input("💷 Ticket Fare (£)", min_value=0.0, max_value=600.0, value=32.0, step=0.5)

st.divider()

# ── Info cards ────────────────────────────────────────────────────────────────
c1, c2, c3 = st.columns(3)
c1.metric("Class", f"{'1st' if pclass==1 else '2nd' if pclass==2 else '3rd'}")
c2.metric("Age", f"{age} yrs", delta="Child 👶" if age < 16 else None)
c3.metric("Fare Paid", f"£{fare:.2f}")

st.divider()

# ── Predict button ────────────────────────────────────────────────────────────
if st.button("🔮 Predict My Survival", use_container_width=True, type="primary"):
    payload = {"pclass": pclass, "sex": sex, "age": float(age), "fare": float(fare)}

    try:
        response = requests.post(API_URL, json=payload, timeout=5)
        response.raise_for_status()
        result = response.json()

        survived = result["survived"]
        prob = result["probability"] * 100

        if survived:
            st.success(f"## ✅ You would have SURVIVED!")
            st.balloons()
        else:
            st.error(f"## ❌ You would NOT have survived.")

        # Probability bar
        st.markdown(f"**Survival Probability: {prob:.1f}%**")
        st.progress(int(prob))

        # Context note
        st.info(
            "💡 **Key insight:** Women and children had priority on lifeboats. "
            "1st class passengers also had significantly higher survival rates."
        )

    except requests.exceptions.ConnectionError:
        st.error("⚠️ Cannot connect to the API. Make sure FastAPI is running on port 8000.")
    except Exception as e:
        st.error(f"Something went wrong: {e}")

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    "<center><small>Built by Abhishek · Logistic Regression · 77% Accuracy · Titanic Dataset (Kaggle)</small></center>",
    unsafe_allow_html=True
)
