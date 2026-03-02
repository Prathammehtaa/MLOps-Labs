# Lab 3 - MLflow: Bank Customer Churn Prediction

## Overview
This lab demonstrates the full ML lifecycle using **MLflow** — experiment tracking, model comparison, model registry, and real-time serving using a Bank Customer Churn dataset.

## Setup

### Install Dependencies
```bash
pip install mlflow scikit-learn pandas seaborn matplotlib
```

### Dataset
The dataset `train.csv` is included in the `data/` folder. It is from the [Kaggle Bank Churn Competition](https://www.kaggle.com/competitions/playground-series-s4e1).

## Running the Lab

### Step 1: Run the Notebook
```bash
cd src/
jupyter notebook bank_churn_mlflow.ipynb
```
Run all cells **in order** from top to bottom. This will:
- Load and preprocess the dataset
- Perform EDA with visualizations
- Train 3 models (Logistic Regression, Random Forest, Gradient Boosting)
- Log all parameters, metrics, and artifacts to MLflow
- Compare models and pick the best one
- Register the best model and transition it to Production

### Step 2: View MLflow UI
Open a **new terminal**, navigate to `src/`, and run:
```bash
mlflow ui --backend-store-uri "sqlite:///mlflow.db" --port 5002
```
Then open [http://127.0.0.1:5002](http://127.0.0.1:5002) in your browser.

You should see the `bank-churn-prediction` experiment with 3 runs. Click into each run to view logged parameters, metrics, and confusion matrix artifacts.

### Step 3: Serve the Model as REST API
Open **another terminal**, navigate to `src/`, and run:
```bash
MLFLOW_TRACKING_URI="sqlite:///mlflow.db" mlflow models serve -m "runs:/<RUN_ID>/gradient_boosting" -h 0.0.0.0 -p 5001 --env-manager local
```

> **Note**: Replace `<RUN_ID>` with the actual run ID printed in the notebook output after training. You can also find it in the MLflow UI.

Wait until you see:
```
Uvicorn running on http://0.0.0.0:5001
```

### Step 4: Test Real-Time Inference
Go back to the notebook and run the last cell, or use this in a new Python script:
```python
import requests

url = 'http://localhost:5001/invocations'
# Use any sample data in the correct format
data = {"dataframe_split": {"columns": [...], "data": [...]}}
response = requests.post(url, json=data)
print(response.json())
```

The response will contain predictions like:
```json
{"predictions": [0, 1, 0, 0, 0]}
```
Where `0` = customer stays, `1` = customer churns.

## Important Notes
- All MLflow data is stored in `sqlite:///mlflow.db` inside the `src/` folder
- Always run terminal commands (`mlflow ui`, `mlflow models serve`) from the `src/` directory
- The notebook must be run completely before the MLflow UI or serve commands will work
- When serving the model, keep the terminal running while testing inference