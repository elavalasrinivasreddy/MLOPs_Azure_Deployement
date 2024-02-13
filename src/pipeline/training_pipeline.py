import os
import sys

from src.logger.logger import logging
from src.exceptions.exception import CustomException
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_training import ModelTrainer

ingestion_obj = DataIngestion()

train_path, test_path = ingestion_obj.init_data_ingestion()

data_preprocessing = DataTransformation()
train, test = data_preprocessing.init_data_transformation(train_path,test_path)

model_train_obj = ModelTrainer()
model_train_obj.init_model_training(train,test)

