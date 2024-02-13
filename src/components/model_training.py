from src.logger.logger import logging
from src.exceptions.exception import CustomException
from src.utils.utils import save_object, load_object, evaluate_model

from dataclasses import dataclass
from pathlib import Path
import sys
import os
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

@dataclass
class ModelTrainerConfig:
    trained_model_path = os.path.join("artifacts", "model.pkl")
    models = {
        "LinearRegression": LinearRegression(),
        "Lasso": Lasso(),
        "Ridge": Ridge(),
        "ElasticNet": ElasticNet(),
        "RandomForestRegressor": RandomForestRegressor(),
        "XGBRegressor": XGBRegressor()
    }

class ModelTrainer:
    
    def __init__(self):
        self.trainer_config = ModelTrainerConfig()

    def init_model_training(self, train_arr, test_arr):
        try:
            logging.info("Model Trainer is Started..!")

            # evaluate the model
            model_reports:dict = evaluate_model(
                X_train= train_arr[:,:-1],
                y_train= train_arr[:,-1],
                X_test= test_arr[:,:-1],
                y_test= test_arr[:,-1],
                models=self.trainer_config.models
            )
            logging.info(f"Model Report: \n{model_reports}")

            # Get the best model
            best_model = Keymax = max(model_reports, key= lambda x: model_reports[x][2])
            logging.info(f"Best Model: {best_model}")
            
        except Exception as e:
            raise CustomException(e,sys)
