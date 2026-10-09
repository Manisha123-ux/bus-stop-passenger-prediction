from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Load the trained MLP model
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "bus_passenger_mlp_model.joblib"

model = joblib.load(MODEL_PATH)

app = FastAPI(
    title="Bus Stop Passenger Prediction API",
    description="Predict passenger counts using an MLP Neural Network.",
    version="1.0.0",
)

# Allow the frontend to connect during development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PassengerInput(BaseModel):
    bus_stop: str
    hour: int = Field(ge=5, le=22)
    weather: str
    special_event: int = Field(ge=0, le=1)
    day_of_week: str
    is_weekend: int = Field(ge=0, le=1)
    previous_passenger_count: int = Field(ge=0, le=150)


@app.get("/")
def home():
    return {
        "message": "Bus Stop Passenger Prediction API is running!"
    }


@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": True}


@app.post("/predict")
def predict_passengers(data: PassengerInput):
    input_data = pd.DataFrame([data.model_dump()])

    prediction = float(model.predict(input_data)[0])
    prediction = max(0, round(prediction))

    if prediction < 30:
        demand_level = "Low"
    elif prediction < 70:
        demand_level = "Moderate"
    else:
        demand_level = "High"

    return {
        "predicted_passenger_count": prediction,
        "demand_level": demand_level,
        "model": "MLP Neural Network",
        "note": "Prediction based on a model trained with simulated educational data."
    }