from src.logger.logger import logging
from src.exceptions.exception import CustomException
from src.utils.utils import save_object
import pandas as pd
import numpy as np

import os
import sys
from dataclasses import dataclass

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file = os.path.join('artifacts','preprocessor.pkl')
    target = "price"
    drop_cols = ["id"]
    # custom ranking for ordinal features
    ordinal_feature_map = {
        "cut_labels": ["Fair","Good","Very Good","Premium","Ideal"],
        "color_labels": ["D","E","F","G","H","I","J"],
        "clarity_labels": ["I1","SI2","SI1","VS2","VS1","VVS2","VVS1","IF"],
    }

class DataTransformation:

    def __init__(self) -> None:
        self.data_transformation_config = DataTransformationConfig()

    def data_preprocessing(self):
        try:
            logging.info('Data preprocessing pipeline started..!')

            # combining all the steps related to numerical features
            numeric_pipeline = Pipeline(
                steps= [
                    ("Imputer",SimpleImputer()),
                    ("Scaler",StandardScaler())
                ]
            )
            logging.debug("Numerical data preprocessing pipeline created.")

            object_pipeline = Pipeline(
                steps= [
                    ("Imputer",SimpleImputer(strategy="most_frequent")),
                    ("Ordinal_Encoder",OrdinalEncoder(categories=list(self.data_transformation_config.ordinal_feature_map.values())))
                ]
            )
            logging.debug("Categorical data preprocessing pipeline created.")

            # combining the different pipelines into single
            preprocessor = ColumnTransformer(
                            [
                                ("numeric_pipeline",numeric_pipeline,self.numeric_cols),
                                ("object_pipeline",object_pipeline,self.object_cols)
                            ]
                        )
            
            return preprocessor

        except Exception as e:
            raise CustomException(e,sys)

    def init_data_transformation(self, train_path:str, test_path:str):
        logging.info("Data Transformation is started.")
        try:
            train_data = pd.read_csv(train_path)
            test_data = pd.read_csv(test_path)
            logging.debug('Train & Test datasets are read')
            logging.debug(f"Train DataFrame: \n{train_data.head(2)}")
            logging.debug(f"Test DataFrame: \n{test_data.head(2)}")
            
            # drop the columns
            train_data.drop(columns=self.data_transformation_config.drop_cols,
                            axis=1, inplace=True)
            test_data.drop(columns=self.data_transformation_config.drop_cols,
                            axis=1, inplace=True)
            
            # separating dependent & independent features
            X_train = train_data.drop(self.data_transformation_config.target,axis=1)
            y_train = train_data[self.data_transformation_config.target]

            if self.data_transformation_config.target in test_data.columns:
                X_test = test_data.drop(self.data_transformation_config.target,axis=1)
                y_test = test_data[self.data_transformation_config.target]

            # separte the category columns
            self.object_cols = X_train.columns[X_train.dtypes=='object']
            # separate the numeric columns
            self.numeric_cols = X_train.columns[X_train.dtypes!='object']

            preprocessing_obj = self.data_preprocessing()
            # applying preprocessing steps 
            X_train_arr = preprocessing_obj.fit_transform(X_train)
            X_test_arr = preprocessing_obj.transform(X_test)
            logging.debug("Data preprocessing is applied on train & test dataset.")

            # vertical concatination [independent_features, target_feature]
            train_arr = np.c_[X_train_arr, np.array(y_train)]
            test_arr = np.c_[X_test_arr, np.array(y_test)]

            # save the preprocessor object
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file,
                obj=preprocessing_obj
            )
            logging.debug("Preprocessing Obj is saved into artifacts folder as a pickle file")

            logging.info("Data Transformation is completed.!")

            return (
                train_arr,
                test_arr
            )

        except Exception as e:
            raise CustomException(e,sys)