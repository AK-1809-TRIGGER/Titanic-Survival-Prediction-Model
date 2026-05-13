import streamlit as st
import numpy as np
import joblib
import os

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Titanic Survival Predictor",
    page_icon="🚢",
    layout="centered"
)

# ── Load model & scaler ONCE ──────────────────────────────────────────────────
@st.cache_resource
def load_model():
    base = os.path.dirname(__file__)
    model_path  = os.path.join(base, "model.pkl")
    scaler_path = os.path.join(base, "scaler.pkl")
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        st.error("❌ model.pkl or scaler.pkl not found. Run `python titanic.py` first.")
        st.stop()
    return joblib.load(model_path), joblib.load(scaler_path)

model, scaler = load_model()

# ── Prediction (no API needed) ────────────────────────────────────────────────
def predict(pclass, sex, age, fare):
    sex_encoded = 1 if sex == "female" else 0
    is_child    = int(age < 16)
    priority    = int(sex_encoded == 1 or is_child)
    features    = np.array([[pclass, sex_encoded, age, fare, is_child, priority]])
    scaled      = scaler.transform(features)
    survived    = int(model.predict(scaled)[0])
    prob        = float(model.predict_proba(scaled)[0][1])
    return survived, prob

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🚢 Titanic Survival Predictor")
st.markdown("**Would you have survived the Titanic?** Fill in the details below to find out.")
st.divider()

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
    age  = st.slider("🎂 Age", min_value=1, max_value=80, value=28)
    fare = st.number_input("💷 Ticket Fare (£)", min_value=0.0, max_value=600.0,
                           value=32.0, step=0.5)

st.divider()

# ── Info cards ────────────────────────────────────────────────────────────────
c1, c2, c3 = st.columns(3)
c1.metric("Class", f"{'1st' if pclass==1 else '2nd' if pclass==2 else '3rd'}")
c2.metric("Age",   f"{age} yrs", delta="Child 👶" if age < 16 else None)
c3.metric("Fare Paid", f"£{fare:.2f}")

st.divider()

# ── Predict button ────────────────────────────────────────────────────────────
if st.button("🔮 Predict My Survival", use_container_width=True, type="primary"):
    survived, prob = predict(pclass, sex, age, fare)

    if survived:
        st.success("## ✅ You would have SURVIVED!")
        st.balloons()
    else:
        st.error("## ❌ You would NOT have survived.")

    st.markdown(f"**Survival Probability: {prob*100:.1f}%**")
    st.progress(int(prob * 100))

    st.info(
        "💡 **Key insight:** Women and children had priority on lifeboats. "
        "1st class passengers also had significantly higher survival rates."
    )

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    "<center><small>Built by Abhishek · Logistic Regression · 77% Accuracy · "
    "Titanic Dataset (Kaggle)</small></center>",
    unsafe_allow_html=True
)
