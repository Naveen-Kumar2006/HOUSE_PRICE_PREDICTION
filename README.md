# 🏠 House Price Prediction

## 📌 Overview
This project predicts house prices using a FastAPI application and a saved scikit-learn model.
It includes:
- A **training script** (`train_model.py`) to preprocess data and build the model.
- An **application script** (`app.py`) to serve the web interface and predictions.
- A dataset (`housing.csv`) used for training and testing.

---

## ⚙️ Features
- Data preprocessing and model training with **scikit-learn**.
- Interactive prediction app built with **FastAPI** and Jinja2.
- Simple workflow: train → save model → run app → get predictions.

## Run locally

Install the runtime dependencies and start the API from the project root:

```sh
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8000
```

The homepage is available at `http://localhost:8000/`, with API documentation at
`http://localhost:8000/docs`.

For Render, use this start command in the web service settings:

```sh
uvicorn app:app --host 0.0.0.0 --port $PORT
```

---
