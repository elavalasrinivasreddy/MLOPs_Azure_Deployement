from src.logger.logger import logging
from src.exceptions.exception import CustomException
import pandas as pd
import glob

import os
import sys
from dataclasses import dataclass
import kaggle

from sklearn.model_selection import train_test_split

@dataclass
class DataIngestionConfig:
    data_from_github:bool = False
    raw_data_dir:str = os.path.join("artifacts", "data")
    train_data_path:str = os.path.join("artifacts", "data", "train.csv")
    test_data_path:str = os.path.join("artifacts", "data", "test.csv")

class DataIngestion:

    def __init__(self) -> None:
        self.ingestion_config = DataIngestionConfig()

    def init_data_ingestion(self):
        logging.info("Data Ingestion is started.")
        try:
            os.makedirs(self.ingestion_config.raw_data_dir, exist_ok=True)
            
            if self.ingestion_config.data_from_github:
                # download data from internet
                raw_data = pd.read_csv("https://raw.githubusercontent.com/nithin07/MLOPs/Gemstone_Price.csv")
                raw_data.to_csv(os.path.join(self.ingestion_config.raw_data_dir,'raw_data.csv'), index=False)
                logging.debug("Raw dataset is downloaded into artifact folder from Internet")
            else:
                # download data from the kaggle
                kaggle.api.authenticate()
                kaggle.api.dataset_download_files(dataset="sukeshbondada/pba-17-regression-gemstone-non-graded",
                                                  path=self.ingestion_config.raw_data_dir,
                                                  unzip=True)
                logging.debug("Raw dataset is downloaded into artifact folder from Kaggle")
                # list the files
                data_files = glob.glob(os.path.join(self.ingestion_config.raw_data_dir, "*.csv"))
                
                if len(data_files) ==1:
                    raw_data = pd.read_csv(data_files[0])
                    raw_data.to_csv(os.path.join(self.ingestion_config.raw_data_dir,'raw_data.csv'), index=False)

            # Split the dataset into train and test
            if not os.path.exists(self.ingestion_config.test_data_path):
                # split the dataset
                train_data, test_data = train_test_split(raw_data, test_size=0.25)
                logging.debug("Data was splitted into train & test")

                train_data.to_csv(self.ingestion_config.train_data_path, index=False)
                test_data.to_csv(self.ingestion_config.test_data_path, index=False)
            logging.info("Data ingestion is completed.!")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            raise CustomException(e,sys)
        
# if __name__=="__main__":
#     obj = DataIngestion()
#     obj.init_data_ingestion()