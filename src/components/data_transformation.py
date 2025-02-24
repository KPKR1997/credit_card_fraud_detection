
import os
import sys
import pandas as pd
from scipy.sparse import csr_matrix, issparse
import numpy as np

from dataclasses import dataclass

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

from src.utils import save_object, load_object
from src.exception import CustomException
from src.logger import logging



#take train, test data from artifacts
@dataclass
class DataTransformationConfig:
    preprocessor_file_path:str = os.path.join('artifacts', 'preprocessor.pkl')
    test_path = os.path.join('artifacts', 'test.csv')
    train_path = os.path.join('artifacts', 'train.csv')

class DataTransformation():
    def __init__(self):
        self.path = DataTransformationConfig() 

    def make_transformer(self):
        try:
            logging.info('Building data preprocessing transformer object')  
            numerical_columns = ['time', 'amount', 'age']
            categorical_columns = ['day', 'card_type', 'entry_mode',
                            'transaction_type', 'merchant', 'country_of_transaction',
                            'shipping_address', 'residence', 'gender', 'bank']

            pipeline_numerical = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy = "median")),
                    ('scaler', StandardScaler())
                ]
            )
            
            pipeline_categorical = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('encoder', OneHotEncoder()),
                    ('scaler', StandardScaler(with_mean=False))
                ]
            )

            preprocessor = ColumnTransformer(
                [
                    ('num_pipeline', pipeline_numerical, numerical_columns),
                    ('cat_pipline', pipeline_categorical, categorical_columns )
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self):
        try:
            self.data_path = DataTransformationConfig()
            test_data = pd.read_csv(self.data_path.test_path)
            train_data = pd.read_csv(self.data_path.train_path)

            logging.info('Imported train and test data, preparing for scaling and encoding')

            preprocessor = self.make_transformer()

            target_column_name = 'fraud'

            feature_train = train_data.drop(columns=[target_column_name], axis = 1)
            target_train = train_data[target_column_name]

            feature_test = test_data.drop(columns=[target_column_name], axis = 1)
            target_test = test_data[target_column_name]

            feature_train_arr = preprocessor.fit_transform(feature_train)
            feature_test_arr = preprocessor.transform(feature_test)

            if issparse(feature_test_arr):
                feature_test_arr = feature_test_arr.toarray()
                feature_train_arr = feature_train_arr.toarray()

            train_arr = np.c_[
                feature_train_arr, np.array(target_train)
            ]
            test_arr = np.c_[
                feature_test_arr, np.array(target_test)
            ]
            
            logging.info('Saving preprocessor file in artifacts')

            save_object(
                file_path = self.data_path.preprocessor_file_path,
                obj=preprocessor
            )
            logging.info('Successfully completed data transformation processing')
            return(
                train_arr,
                test_arr,
                self.data_path.preprocessor_file_path
            )
            
            
        except Exception as e:
            raise CustomException(e,sys)


       