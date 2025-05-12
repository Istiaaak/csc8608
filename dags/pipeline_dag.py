import os, sys
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ingest import ingest
from src.train_mlflow import train_and_log

default_args = {
    "start_date": datetime(2025, 5, 8),
    "retries": 1
}

with DAG(
    dag_id="patchcore_full_pipeline",
    default_args=default_args,
    schedule_interval=None,
    catchup=False,
) as dag:

    # Ingestion / chargement des données
    t1 = PythonOperator(
        task_id="ingest",
        python_callable=ingest,
        op_kwargs={
            "cls": "bottle",
            "size": 224,
            "vanilla": True,
            "out_path": "./datasets"
        },
        do_xcom_push=True,  # pour retourner le chemin datasets/cls
    )

    # Entraînement + split train/test + évaluation + logging MLflow
    t2 = PythonOperator(
        task_id="train_and_log",
        python_callable=train_and_log,
        
        op_kwargs={
            "cls": "bottle",
            "backbone_key": "{{ dag_run.conf.get('backbone_key','WideResNet50') }}",
            "f_coreset": "{{ dag_run.conf.get('f_coreset',0.1) }}",
            "eps": "{{ dag_run.conf.get('eps',0.9) }}",
            "k_nn": "{{ dag_run.conf.get('k_nn',3) }}",
            "use_cache": True,
            "cache_root": "./patchcore_cache",
            "mlflow_experiment": "{{ dag_run.conf.get('mlflow_experiment','PatchCore_MVTec') }}"
        },
        do_xcom_push=True  # pour récupérer les metriques
    )

    t1 >> t2
