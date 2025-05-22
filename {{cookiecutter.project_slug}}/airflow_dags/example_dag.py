from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime

with DAG("llm_classification_pipeline", start_date=datetime(2024, 1, 1), schedule_interval="@daily", catchup=False) as dag:
    train = BashOperator(
        task_id="train_model",
        bash_command="python /app/src/train.py"
    )
