from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime, timedelta
from src.ml_functions import *

default_args = {
    'owner': 'pratham',
    'start_date': datetime(2026, 2, 1),
    'retries': 1,
}

dag = DAG(
    'burnout_pipeline',
    default_args=default_args,
    schedule_interval=None,
    catchup=False,
)

# Task 1: Load data
load_data_task = PythonOperator(
    task_id='load_data',
    python_callable=load_data,
    dag=dag,
)

# Task 2: Preprocess data
preprocess_task = PythonOperator(
    task_id='preprocess_data',
    python_callable=preprocess_data,
    dag=dag,
)

# Task 3: Train Random Forest
train_rf_task = PythonOperator(
    task_id='train_random_forest',
    python_callable=train_random_forest,
    dag=dag,
)

# Task 4: Train XGBoost
train_xgb_task = PythonOperator(
    task_id='train_xgboost',
    python_callable=train_xgboost,
    dag=dag,
)

# Task 5: Train Logistic Regression
train_lr_task = PythonOperator(
    task_id='train_logistic_regression',
    python_callable=train_logistic_regression,
    dag=dag,
)

# Task 6: Compare models
compare_task = PythonOperator(
    task_id='compare_models',
    python_callable=compare_models,
    dag=dag,
)

# Task 7: Check accuracy threshold
check_threshold_task = PythonOperator(
    task_id='check_accuracy_threshold',
    python_callable=check_accuracy_threshold,
    dag=dag,
)

# Task 8: Save model
save_model_task = PythonOperator(
    task_id='save_model',
    python_callable=save_model,
    dag=dag,
)

# Define task dependencies
load_data_task >> preprocess_task >> [train_rf_task, train_xgb_task, train_lr_task] >> compare_task >> check_threshold_task >> save_model_task
