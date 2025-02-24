import os
import sys
import pandas as pd
import numpy as np

from src.exception import CustomException
from src.logger import logging


class DataValidation:
    def __init__(self, data:pd.DataFrame):
        self.data = data

    def default_data_validation(self):
        try:
            logging.info('Default data validation initiated')
            self.data = self.data.drop_duplicates().dropna()
        except Exception as e:
            raise CustomException(e,sys)
        
    def custom_data_validation(self):
        try:
            logging.info('custom data validation started')
            data = self.data.copy()
            data['Amount'] = data['Amount'].str.replace('£', '').astype(float)
            data = data.drop(columns=['Transaction ID', 'Date'])
            data.columns = ['day', 'time', 'card_type', 'entry_mode', 'amount',
                            'transaction_type', 'merchant', 'country_of_transaction',
                            'shipping_address', 'residence', 'gender', 'age', 'bank', 'fraud']
            logging.info('Validation and cleaning completed successfully')
            return data
        except Exception as e:
            raise CustomException(e,sys)