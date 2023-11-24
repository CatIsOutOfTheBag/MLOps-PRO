from datetime import datetime
from airflow.models import DAG, Variable
from airflow.operators.python_operator import PythonOperator
from bots.EmailHelper import send

default_args = {
    'owner': 'Olga',
    'start_date': datetime(2023, 9, 20),    
}

with DAG(
    dag_id='PythonEmailDag',
    default_args=default_args,
    schedule_interval=None) as dag:

    start_dag=PythonOperator(
        task_id="start_dag",
        python_callable=send
    )

    start_dag