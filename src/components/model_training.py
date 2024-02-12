from src.logger.logger import logging
from src.exceptions.exception import CustomException
from src.utils.utils import save_object, load_object

from dataclasses import dataclass
from pathlib import Path
import sys
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

@dataclass
class ModelTrainerConfig:
    pass

class ModelTrainer:
    
    def __init__(self):
        pass

    def init_model_training():
        try:
            pass
        except Exception as e:
            raise CustomException(e,sys)
