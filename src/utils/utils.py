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