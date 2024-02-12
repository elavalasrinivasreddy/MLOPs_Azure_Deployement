from dataclasses import dataclass
import mlflow
import sys
import mlflow.sklearn
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from src.logger.logger import logging
from src.exceptions.exception import CustomException

@dataclass
class ModelEvaluationConfig:
    pass

class init_model_evaluation:

    def __init__(self) -> None:
        pass

    def init_evaluate_model(self):
        try:
            pass
        except Exception as e:
            raise CustomException(e, sys)
