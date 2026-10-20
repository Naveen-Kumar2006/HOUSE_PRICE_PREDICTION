# 🏠 House Price Prediction

A Machine Learning web application that predicts house prices using a trained **Decision Tree Regressor**. Built with Python, FastAPI, HTML, CSS, and JavaScript, and deployed on Render.

### 🔗 Project Links

* 🌐 **Live Demo:** https://house-price-prediction-api-x334.onrender.com/
* 📖 **API Documentation:** https://house-price-prediction-api-x334.onrender.com/docs
* 💻 **GitHub Repository:** https://github.com/Naveen-Kumar2006/HOUSE_PRICE_PREDICTION

---

## 📌 Overview

House Price Prediction is a Machine Learning project that estimates house prices based on user-provided property details.

The application integrates a trained regression model with a FastAPI backend and a web-based frontend, allowing users to interact with the model through a simple interface.

## ✨ Features

* 🏡 Predict house prices using Machine Learning.
* 🤖 Uses a Decision Tree Regressor.
* ⚡ FastAPI backend for prediction requests.
* 🎨 Interactive frontend built with HTML, CSS, and JavaScript.
* 🔗 REST API integration.
* 💾 Model serialization using Joblib.
* ☁️ Cloud deployment using Render.
* 📖 Interactive API documentation with Swagger UI.

## 🛠️ Tech Stack

| Technology   | Purpose                   |
| ------------ | ------------------------- |
| Python       | Core programming language |
| Pandas       | Data manipulation         |
| Scikit-learn | Machine Learning          |
| Joblib       | Model serialization       |
| FastAPI      | Backend API               |
| Uvicorn      | Application server        |
| HTML         | Webpage structure         |
| CSS          | Styling                   |
| JavaScript   | Frontend interactions     |
| Jinja2       | HTML template rendering   |
| Render       | Deployment                |
| Git & GitHub | Version control           |

## 🧠 Machine Learning Workflow

1. Load the housing dataset.
2. Prepare the features and target variable.
3. Train the Decision Tree Regressor.
4. Save the trained model using Joblib.
5. Load the model into the FastAPI application.
6. Accept property details from the user.
7. Generate and return the predicted house price.

**Trained model:** `model_DecisionTreeRegressor_V1.joblib`

## 📂 Project Structure

```text
HOUSE_PRICE_PREDICTION/
├── app.py
├── train_model.py
├── model_DecisionTreeRegressor_V1.joblib
├── housing.csv
├── requirements.txt
├── README.md
├── templates/
│   └── index.html
└── static/
    ├── css/
    │   └── style.css
    └── js/
        └── main.js
```

## ⚙️ Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/Naveen-Kumar2006/HOUSE_PRICE_PREDICTION.git
```

### 2. Navigate to the project directory

```bash
cd HOUSE_PRICE_PREDICTION
```

### 3. Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the application

```bash
uvicorn app:app --reload
```

### 6. Open the application

* **Website:** http://127.0.0.1:8000
* **API Documentation:** http://127.0.0.1:8000/docs

## 🔌 API Information

The application exposes a prediction endpoint:

* **Method:** `POST`
* **Endpoint:** `/predict`
* **Framework:** FastAPI
* **Request format:** JSON

Explore the actual request schema and test the endpoint here:

https://house-price-prediction-api-x334.onrender.com/docs

## ☁️ Deployment

The application is deployed on Render.

🌐 **Live Application:** https://house-price-prediction-api-x334.onrender.com/

The deployed application serves the frontend and processes prediction requests through the FastAPI backend.

*Note: On Render's free plan, the application may take a little time to respond after a period of inactivity.*

## 🔮 Future Improvements

* Evaluate model performance using MAE, RMSE, and R².
* Improve prediction accuracy through hyperparameter tuning.
* Compare multiple regression algorithms.
* Add automated API tests.
* Improve input validation and error handling.
* Enhance the user interface and visualizations.

## 🎯 Learning Outcomes

* Building a regression model using scikit-learn.
* Training and saving Machine Learning models.
* Developing REST APIs with FastAPI.
* Integrating frontend and backend components.
* Deploying a Machine Learning application to the cloud.
* Managing project dependencies and version control.

## 👨‍💻 Author

**Naveen Kumar**

* GitHub: https://github.com/Naveen-Kumar2006
* Project Repository: https://github.com/Naveen-Kumar2006/HOUSE_PRICE_PREDICTION

---

⭐ If you find this project useful, consider giving the repository a star!
