import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split , GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
import time
import logging
import mlflow
import mlflow.sklearn



logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


# Load the dataset
def train_and_save_model():
    
    try:
        logging.info("Loading data...")
        data = pd.read_csv('housing.csv')
    except Exception as e:
        logging.info(f"Error loading housing.csv: {e}")
        return

    # Define features and target
    X = data.drop('median_house_value', axis=1)
    y = data['median_house_value']

    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=3) 

    #Numerical_Columns
    num_cols=X_train.select_dtypes(include=["int64","float64"]).columns
    #Categorical_Columns
    cat_cols = X_train.select_dtypes(include=["object"]).columns

    #Numerical_Pipeline
    num_pipe=Pipeline([
        ("Imputer",SimpleImputer(strategy="mean")),
        ("Scaler",StandardScaler())
    ])
    #Categorical_Pipeline
    cat_pipe=Pipeline([
        ("Imputer",SimpleImputer(strategy="most_frequent")),
        ("Encoding",OneHotEncoder(handle_unknown="ignore"))
    ])

    #ColumnTransformer
    Preprocessor=ColumnTransformer([
        ("Numerics",num_pipe,num_cols),
        ("Category",cat_pipe,cat_cols)
    ])

    #Complete_Pipeline
    com_pipeline=Pipeline([
        ("Preprocessing",Preprocessor),
        ("model",DecisionTreeRegressor(random_state=3))
    ])

    #hyperparameter Grid
    params={
        "model__criterion":["squared_error","absolute_error","poisson"],
        "model__splitter":["best","random"],
        "model__max_depth":[2,5,7,10],
        "model__min_samples_split":[2,5,10],
        "model__min_samples_leaf":[1,2,4]
    }

    #GridSearchCV
    model_grid=GridSearchCV(
        estimator=com_pipeline,
        param_grid=params,
        cv=5,
        scoring="r2",
        n_jobs=-1,
        return_train_score=True
    )

    mlflow.set_experiment("DecisionTreeRegressor Experiment ")
    with mlflow.start_run(run_name="DecisionTreeRegressor"):
        start=time.time()
        #Model_train
        logging.info("Training the Model.....")
        model_grid.fit(X_train,y_train)
        end=time.time()

        #Prediction
        predictions=model_grid.predict(X_test)

        #Root_Mean_Squared_Error
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        test_r2 = r2_score(y_test, predictions)
        logging.info(f"Root_Mean_Squared_Error : {rmse:2f}")
        logging.info(f"Test R² Score : {test_r2:.4f}")

        logging.info(f"Best Parameters : {model_grid.best_params_}")
        logging.info(f"CV Score : {model_grid.best_score_:.4f}")
        logging.info(f"Best Estimator : {model_grid.best_estimator_}")
    

        #Time_Details
        logging.info(f"Training Time : {end-start:.2f} sec")

        #Model_Dumping
        joblib.dump(model_grid.best_estimator_,"model_DecisionTreeRegressor_V1.joblib")
        mlflow.log_artifact("model_DecisionTreeRegressor_V1.joblib")


        #Mlflow
        mlflow.log_params(model_grid.best_params_)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("test_r2", test_r2)
        mlflow.log_metric("cv_score", model_grid.best_score_)
        training_time = end - start
        mlflow.log_metric("training_time_sec", training_time)

        mlflow.sklearn.log_model(
                    sk_model=model_grid.best_estimator_,
                    artifact_path="DecisionTreeRegressor",
                    serialization_format="cloudpickle"
                )

        mlflow.set_tags({
                    "algorithm": "DecisionTreeRegressor",
                    "framework": "Scikit-Learn",
                    "dataset": "housing.csv",
                    "author": "NAVEEN KUMAR",
                    "stage": "development"
})

        #Log Dataset Information
        mlflow.log_param("rows", len(data))
        mlflow.log_param("columns", len(data.columns))
        mlflow.log_param("train_samples", len(X_train))
        mlflow.log_param("test_samples", len(X_test))

        #Log Feature Names
        mlflow.log_dict({"features": list(X.columns)},"features.json")

        #Log Predictions
        pred_df = pd.DataFrame({"Actual": y_test,"Predicted": predictions})
        pred_df.to_csv("predictions.csv", index=False)
        mlflow.log_artifact("predictions.csv")


if __name__ == "__main__":
    train_and_save_model()
