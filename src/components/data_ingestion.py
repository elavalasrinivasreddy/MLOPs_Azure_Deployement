from src.logger.logger import logging
from src.exceptions.exception import CustomException
import pandas as pd
import numpy as np

import os
import sys
from dataclasses import dataclass
from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

@dataclass
class DataTransformationConfig:
    pass

class DataTransformation:

    def __init__(self) -> None:
        pass

    def init_data_ingestion(self):
        try:
            pass
        except Exception as e:
            raise CustomException(e,sys)