import os
import sys
import pandas as pd

from src.exceptions.exception import CustomException
from src.logger.logger import logging
from src.utils.utils import load_object

class PredictionPipeline:
    
    def __init__(self) -> None:
        pass

    def predict(self,features):
        try:
            preprocessor_obj_path = os.path.join("artifacts","preprocessor.pkl")
            model_path = os.path.join("artifacts","model.pkl")

            # load the obj's
            preprocessor_obj = load_object(preprocessor_obj_path)
            model = load_object(model_path)

            # transform the data
            preprocess_test = preprocessor_obj.transform(features)

            return model.predict(preprocess_test)

        except Exception as e:
            raise CustomException(e,sys)
        
class CustomData:
    def __init__(self,
                 carat:float,
                 depth:float,
                 table:float,
                 x:float,
                 y:float,
                 z:float,
                 cut:str,
                 color:str,
                 clarity:str):
        self.carat = carat
        self.depth = depth
        self.table = table
        self.x = x
        self.y = y
        self.z = z
        self.cut = cut
        self.color = color
        self.clarity = clarity

    def arr_to_datafarme(self):
        try:
            df = pd.DataFrame({
                "carat": [self.carat],
                "cut": [self.cut],
                "color": [self.color],
                "clarity": [self.clarity],
                "depth": [self.depth],
                "table": [self.table],
                "x": [self.x],
                "y": [self.y],
                "z": [self.z],
                }
            )
            logging.info('Prediction data is collected.')
            return df
        
        except Exception as e:
            raise CustomException(e,sys)