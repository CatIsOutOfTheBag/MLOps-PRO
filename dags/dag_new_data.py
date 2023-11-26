from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta
from airflow.contrib.hooks.ssh_hook import SSHHook
from airflow.contrib.operators.sftp_operator import SFTPOperator
from airflow.operators.python_operator import PythonOperator
from airflow.contrib.operators.ssh_operator import SSHOperator
from bots.EmailHelper import send

default_args = {
    'owner': 'Olga',
    'start_date': datetime(2023, 9, 20),    
}

dag = DAG(
        'Olga_dag',
        default_args=default_args,       
        schedule_interval='@daily',
        )

# Создаем SSHHook для доступа к удаленному серверу
ssh_hook = SSHHook(ssh_conn_id='my_ssh_conn_id')


sftp_task1 = SFTPOperator(
    task_id='sftp_transfer_gen',
    ssh_hook=ssh_hook,
    local_filepath='/home/admin/airflow/gen.py',
    remote_filepath='/home/ubuntu/gen.py',
    operation='put',
    dag=dag,
)

exec_task1 = SSHOperator(
    task_id="run_gen",
    cmd_timeout=7200,	
    command='spark-submit gen.py',
    ssh_hook=ssh_hook,
    dag=dag)


send_email_task = PythonOperator(
    task_id="send_email",
    python_callable=send,
    dag=dag
    )

sftp_task1 >> exec_task1 >> send_email_task
