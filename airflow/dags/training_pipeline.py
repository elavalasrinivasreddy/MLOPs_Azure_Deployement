from airflow import DAG
from airflow.operators.python import PythonOperator
import pendulum
from textwrap import dedent
from __future__ import annotations
import numpy as np

# from src.logger.logger import logging
# from src.exceptions.exception import CustomException
# from src.pipeline.training_pipeline import ModelTrainer
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_training import ModelTrainer

training_pipeline = ModelTrainer()

with DAG(
    "gemstone_training_pipeline",
    default_args={"retries":2},
    description="Model Training Pipeline",
    schedule="@weekly",
    start_date=pendulum.datetime(2024,2,15,tz="UTC"),
    catchup=False,
    tags=["Machine_Learning","Gemstone","Prediction"],
) as dag:
    dag.doc_md = __doc__

    def data_ingestion(**kwargs):
        ti = kwargs["ti"]
        train_data_path, test_data_path = DataIngestion()
        ti.xcom_push("data_ingestion_artifact",
                     {"train_data_path":train_data_path,
                      "test_data_path":test_data_path})
    
    def data_transformation(**kwargs):
        ti = kwargs["ti"]
        data_ingestion_artifact = ti.xcom_pull(
            task_ids="data_ingestion",
            key = "data_ingestion_artifact"
        )
        train_arr, test_arr = DataTransformation(
            data_ingestion_artifact["train_data_path"],
            data_ingestion_artifact["test_data_path"]
        )
        train_arr = train_arr.tolist()
        test_arr = test_arr.tolist()
        ti.xcom_push("data_transformation_artifact",
                     {
                         "train_arr": train_arr,
                         "test_arr": test_arr
                     })
        
    def model_trainer(**kwargs):
        ti = kwargs["ti"]
        data_transformation_artifact = ti.xcom_pull(
            task_ids = "data_transformation",
            key = "data_transformation_artifact"
        )
        train_arr = np.array(data_transformation_artifact["train_arr"])
        test_arr = np.array(data_transformation_artifact["test_arr"])
        ModelTrainer(train_arr,test_arr)

    data_ingestion_task = PythonOperator(
        task_id = "data_ingestion",
        python_callable = data_ingestion,
    )
    data_ingestion_task.doc_md = dedent(
        """\
This tasks read the data from sources and it creates a Raw CSV file.
        """
    )

    data_transformation_task = PythonOperator(
        task_id = "data_transformation",
        python_callable = data_transformation,
    )
    data_transformation_task.doc_md = dedent(
        """\
This task preprocess the data and split the Raw data into train & test.
        """
    )

    model_trainer_task = PythonOperator(
        task_id = "model_trainer",
        python_callable = model_trainer,
    )
    model_trainer_task.doc_md = dedent(
        """\
This task train multiple models and save the best model.
        """
    )

data_ingestion_task >> data_transformation_task >> model_trainer_task
