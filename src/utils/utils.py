import os
import sys
import pickle
from src.logger.logger import logging
from src.exceptions.exception import CustomException
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)

        with open(file_path,"wb") as f:
            pickle.dump(obj,f)
        logging.debug(f"Obj is saved {file_path}")

    except Exception as e:
        raise CustomException(e,sys)
    
def load_object(file_path):
    try:
        with open(file_path,'rb') as f:
            obj = pickle.load(f)

            logging.debug(f"Obj was loaded {file_path}")
            return obj
        
    except Exception as e:
        raise CustomException(e,sys)
    
def evaluate_model(X_train,y_train,X_test,y_test,models):
    try:
        report = {}
        for model_name, model_obj in models.items():
            # fitting on data
            model_obj.fit(X_train,y_train)

            # save the model into artifacts
            model_path = os.path.join("artifacts",f"{model_name}.pkl")
            save_object(file_path=model_path,
                        obj=model_obj)
            logging.info(f"{model_name} Model is saved into artifacts folder")

            # prediction
            y_pred = model_obj.predict(X_test)

            # Model evaluation
            R2 = r2_score(y_pred=y_pred,y_true=y_test)
            MSE = mean_squared_error(y_pred=y_pred,y_true=y_test)
            MAE = mean_absolute_error(y_pred=y_pred,y_true=y_test)

            print(f"Model Testing Performance {model_name}")
            print("MSE: ",MSE)
            print("MAE: ",MAE)
            print("R2-score: ",R2)
            print('\n','--'*20)

            logging.info(f"{model_name} Model Performance: {MSE, MAE, R2}")

            report[model_name] = [MSE, MAE, R2]

        return report
    
    except Exception as e:
        raise CustomException(e,sys)