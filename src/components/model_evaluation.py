from dataclasses import dataclass
import mlflow
import sys
import os
import mlflow.sklearn
from urllib.parse import urlparse
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from src.logger.logger import logging
from src.exceptions.exception import CustomException
from src.utils.utils import load_object

@dataclass
class ModelEvaluationConfig:
    pass

class ModelEvaluation:

    def __init__(self) -> None:
        logging.info('Model Evaluation started.')
        pass

    def evaluate_metrics(self,actual,pred):

        rmse = mean_squared_error(actual,pred)
        mae = mean_absolute_error(actual,pred)
        r2 = r2_score(actual,pred)
        # logging.info('Evaluation metrics are calculated.')
        return rmse, mae,r2

    def init_evaluate_model(self, test_arr):
        try:
            X_test, y_test = test_arr[:,:-1], test_arr[:,-1]

            model_path = os.path.join('artifacts','model.pkl')
            model = load_object(model_path)

            # mlflow.set_registry_uri("")
            tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
            print(tracking_url_type_store)

            # start mlflow server
            mlflow.start_run()
            prediction = model.predict(X_test)
            mse, mae, r2 = self.evaluate_metrics(y_test,prediction)
            mlflow.log_metric("rmse",mse)
            mlflow.log_metric("mae",mae)
            mlflow.log_metric("r2",r2)
            logging.info('Model evaluation metrics are captured.')

            # Model registry does not work with file store
            if tracking_url_type_store != "file":

                # Register the model
                # There are other ways to use the Model Registry, which depends on the use case,
                # please refer to the doc for more information:
                # https://mlflow.org/docs/latest/modelregistry.html#apiworkflow
                mlflow.sklearn.log_model(model, "model", registered_model_name="ml_model")
            else:
                mlflow.sklearn.log_model(model, "model")

            mlflow.end_run()

        except Exception as e:
            raise CustomException(e, sys)
