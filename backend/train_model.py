import pandas as pd
import numpy as np
import joblib

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "bus_stop_passenger_prediction_dataset.csv"
MODEL_DIR = BASE_DIR / "backend" / "model"
MODEL_PATH = MODEL_DIR / "bus_passenger_mlp_model.joblib"

# Load dataset
df = pd.read_csv(DATA_PATH)
print("Dataset loaded:", df.shape)

# Features and target
features = [
    "bus_stop",
    "hour",
    "weather",
    "special_event",
    "day_of_week",
    "is_weekend",
    "previous_passenger_count",
]

X = df[features]
y = df["passenger_count"]

# Feature types
categorical_features = ["bus_stop", "weather", "day_of_week"]
numerical_features = [
    "hour",
    "special_event",
    "is_weekend",
    "previous_passenger_count",
]

# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features,
        ),
        ("numerical", StandardScaler(), numerical_features),
    ]
)

# MLP Neural Network
mlp = MLPRegressor(
    hidden_layer_sizes=(64, 32),
    activation="relu",
    solver="adam",
    max_iter=1000,
    random_state=42,
    early_stopping=True,
    validation_fraction=0.1,
)

# Full pipeline
model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("mlp", mlp),
    ]
)

# Split data into 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training the MLP model...")
model.fit(X_train, y_train)

# Evaluate using test data
predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
r2 = r2_score(y_test, predictions)

print("\nModel Evaluation Results")
print("------------------------")
print(f"MAE:  {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R2 Score: {r2:.4f}")

# Save the trained pipeline
MODEL_DIR.mkdir(parents=True, exist_ok=True)
joblib.dump(model, MODEL_PATH)

print("\nMLP model trained and saved successfully!")
print("Model location:", MODEL_PATH)