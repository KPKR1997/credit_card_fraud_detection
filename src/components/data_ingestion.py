import os
import sys
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.exception import CustomException
from src.logger import logging
from configuration import config
from src.components.data_validation import DataValidation


@dataclass
class DataIngestionConfig:
    raw_data_path: str=os.path.join('data',config.CSV_NAME)
    train_data_path: str=os.path.join('artifacts',"train.csv")
    test_data_path: str=os.path.join('artifacts',"test.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info('Initiated CSV file loading')
        try:
            data = pd.read_csv(self.ingestion_config.raw_data_path)
            return data
        except Exception as e:
            raise CustomException(e,sys)


# Train test split and save in artifacts

    def split_train_test(self, data:pd.DataFrame):
        self.data = data
        self.ingestion_path = DataIngestionConfig()
        try:
            logging.info('Initiating train test data split')
            train_data, test_data = train_test_split(data, random_state=42, test_size=0.2, stratify=data['fraud'])
            
            os.makedirs(os.path.dirname(self.ingestion_path.train_data_path), exist_ok=True)
            train_data.to_csv(self.ingestion_path.train_data_path, index=False)
            test_data.to_csv(self.ingestion_path.test_data_path,index=False)


            logging.info('Completed data split by 0.2 ratio of test data size. Generated train and test csv files in artifacts')
        except Exception as e:
            raise CustomException(e,sys)