import logging
from pathlib import Path

import joblib
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="House Price Prediction API")
BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "model_DecisionTreeRegressor_V1.joblib"
if not MODEL_PATH.is_file():
    raise FileNotFoundError(f"Trained model not found: {MODEL_PATH}")
model = joblib.load(MODEL_PATH)

TEMPLATE_DIR = BASE_DIR / "templates"
if not TEMPLATE_DIR.is_dir():
    raise FileNotFoundError(f"Templates directory not found: {TEMPLATE_DIR}")
templates = Jinja2Templates(directory=str(TEMPLATE_DIR))

STATIC_DIR = BASE_DIR / "static"
if not STATIC_DIR.is_dir():
    raise FileNotFoundError(f"Static directory not found: {STATIC_DIR}")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    try:
        return templates.TemplateResponse(
            request=request,
            name="index.html",
        )
    except Exception as exc:
        logger.exception("Home page rendering failed")
        raise HTTPException(
            status_code=500,
            detail="Unable to render the home page.",
        ) from exc


@app.post("/predict")
async def predict(data: dict):
    try:
        input_data = {
            "longitude": float(data.get("longitude", -122.23)),
            "latitude": float(data.get("latitude", 37.88)),
            "housing_median_age": float(data.get("age", 41)),
            "total_rooms": float(data.get("rooms", 880)),
            "total_bedrooms": float(data.get("bedrooms", 129)),
            "population": float(data.get("population", 322)),
            "households": float(data.get("households", 126)),
            "median_income": float(data.get("income", 8.3252)),
            "ocean_proximity": data.get("proximity", "NEAR BAY"),
        }
        input_df = pd.DataFrame([input_data])
        prediction = model.predict(input_df)[0]

        agent_info = {
            "name": "Sarah Mitchell",
            "phone": "+1 (555) 012-3456",
            "email": "sarah.m@elegancerealty.com",
            "agency": "Elegance Luxury Realty",
        }

        return {
            "success": True,
            "prediction": round(float(prediction), 2),
            "agent": agent_info,
        }

    except (TypeError, ValueError) as exc:
        logger.warning("Invalid prediction input: %s", exc)
        raise HTTPException(
            status_code=400,
            detail="Invalid input data.",
        ) from exc
    except Exception as exc:
        logger.exception("Prediction failed")
        raise HTTPException(
            status_code=500,
            detail="Prediction failed.",
        ) from exc


if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
