import os
import sys
import pickle
from src.logger.logger import logging
from src.exceptions.exception import CustomException

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

            # prediction
            y_pred = model_obj.predict(X_test)

            # Model evaluation
            R2, MAE, MSE = evaluate_model(y_test,y_pred)

            # print(f"Model Testing Performance {k}")
            # print("MSE: ",MSE)
            # print("MAE: ",MAE)
            # print("R2-score: ",R2)
            # print('\n','--'*20)

            report[model_name] = [MSE, MAE, R2]

        return report
    
    except Exception as e:
        raise CustomException(e,sys)