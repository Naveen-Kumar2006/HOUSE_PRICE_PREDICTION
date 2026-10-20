from pathlib import Path

import joblib
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# Initialize FastAPI
app = FastAPI(title="House Price Prediction API")

# Project directory
BASE_DIR = Path(__file__).resolve().parent

# Load trained model
MODEL_PATH = BASE_DIR / "model_DecisionTreeRegressor_V1.joblib"

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

model = joblib.load(MODEL_PATH)

# Configure HTML templates
templates = Jinja2Templates(
    directory=str(BASE_DIR / "templates")
)

# Serve CSS and JavaScript files
STATIC_DIR = BASE_DIR / "static"

if STATIC_DIR.exists():
    app.mount(
        "/static",
        StaticFiles(directory=str(STATIC_DIR)),
        name="static"
    )


# Home page
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html"
    )


# House price prediction endpoint
@app.post("/predict")
async def predict(data: dict):
    try:
        # Map frontend fields to model feature names
        input_data = {
            "longitude": float(data.get("longitude", -122.23)),
            "latitude": float(data.get("latitude", 37.88)),
            "housing_median_age": float(data.get("age", 41)),
            "total_rooms": float(data.get("rooms", 880)),
            "total_bedrooms": float(data.get("bedrooms", 129)),
            "population": float(data.get("population", 322)),
            "households": float(data.get("households", 126)),
            "median_income": float(data.get("income", 8.3252)),
            "ocean_proximity": data.get("proximity", "NEAR BAY")
        }

        # Create DataFrame with the expected feature names
        input_df = pd.DataFrame([input_data])

        # Predict house price
        prediction = model.predict(input_df)[0]

        # Agent information
        agent_info = {
            "name": "Sarah Mitchell",
            "phone": "+1 (555) 012-3456",
            "email": "sarah.m@elegancerealty.com",
            "agency": "Elegance Luxury Realty"
        }

        return {
            "success": True,
            "prediction": round(float(prediction), 2),
            "agent": agent_info
        }

    except (TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid input data: {exc}"
        ) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {exc}"
        ) from exc


# Run application
if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="127.0.0.1",
        port=8000,
        reload=True
    )