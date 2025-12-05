from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator
import joblib
import os
from typing import List

app = FastAPI()

# 1. Define the input data schema using Pydantic
# This ensures the API only accepts valid data types and structures.
class DigitInput(BaseModel):
    pixels: List[float]

    # Optional: Add validation to ensure exactly 64 pixels are passed
    @field_validator('pixels')
    def check_length(cls, v):
        if len(v) != 64:
            raise ValueError('Input list must contain exactly 64 pixel values (8x8 flattened).')
        return v

# 2. Load the model securely
# It is better to use a variable for the path or a relative path.
MODEL_PATH = "d:/Cybersecurity/101. Sunrise Institute/3. Artificial intelligence/project/digits_model.pkl"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at: {MODEL_PATH}")

model = joblib.load(MODEL_PATH)

@app.post("/predict_digit")
def predict_digit(data: DigitInput):
    try:
        # 3. Access data cleanly through the Pydantic model
        features = data.pixels
        
        # Predict expects a 2D array, so we wrap features in a list: [features]
        prediction = model.predict([features])
        
        # 4. Convert NumPy types to Python native types (int) for JSON serialization
        return {"prediction": int(prediction[0])}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))