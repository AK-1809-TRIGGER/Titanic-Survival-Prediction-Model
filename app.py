from fastapi import FastAPI
from pydantic import BaseModel, Field
import numpy as np
import joblib

app = FastAPI(
    title="Titanic Survival Predictor API",
    description="Predict whether a Titanic passenger would survive 🚢",
    version="1.0.0"
)

# Load model and scaler at startup
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")


class PassengerInput(BaseModel):
    pclass: int = Field(..., ge=1, le=3, description="Passenger class (1, 2, or 3)")
    sex: str = Field(..., description="'male' or 'female'")
    age: float = Field(..., ge=0, le=100, description="Age in years")
    fare: float = Field(..., ge=0, description="Ticket fare")


class PredictionOutput(BaseModel):
    survived: bool
    probability: float
    message: str


@app.get("/")
def root():
    return {"message": "Titanic Survival Prediction API is running 🚢"}


@app.post("/predict", response_model=PredictionOutput)
def predict(passenger: PassengerInput):
    sex_encoded = 1 if passenger.sex.lower() == "female" else 0
    is_child = passenger.age < 16
    priority = sex_encoded == 1 or is_child  # female or child

    features = np.array([[
        passenger.pclass,
        sex_encoded,
        passenger.age,
        passenger.fare,
        int(is_child),
        int(priority)
    ]])

    features_scaled = scaler.transform(features)
    survived = bool(model.predict(features_scaled)[0])
    probability = float(model.predict_proba(features_scaled)[0][1])

    if survived:
        msg = f"✅ Likely survived with {probability:.1%} probability"
    else:
        msg = f"❌ Likely did NOT survive. Survival probability: {probability:.1%}"

    return PredictionOutput(
        survived=survived,
        probability=round(probability, 4),
        message=msg
    )


@app.get("/health")
def health():
    return {"status": "ok"}