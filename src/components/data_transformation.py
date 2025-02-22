
import os
import sys
import pandas as pd

from dataclasses import dataclass

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

from src.utils import save_object, load_object
from src.exception import CustomException
from src.logger import logging
from src.components.data_ingestion import DataIngestionConfig


#take train, test data from artifacts
@dataclass
class DataTransformationConfig:
    preprocessor_file_path:str = os.path.join('artifacts', 'preprocessor.pkl')
    test_path = os.path.join('artifacts', 'test.csv')
    train_path = os.path.join('artifacts', 'train.csv')

class DataTransformation():
    def __init__(self):
        self.path = DataTransformationConfig()

    logging.info('Building data preprocessing transformer object')   

    def make_transformer(self):
        try: 
            numerical_columns = ['time', 'amount', 'age']
            categorical_columns = ['day', 'card_type', 'entry_mode',
                            'transaction_type', 'merchant', 'country_of_transaction',
                            'shipping_addres', 'residence', 'gender', 'bank']

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
            preprocessor = self.make_transformer()

            target_column_name = 'fraud'

            feature_train = train_data.drop(columns=[target_column_name], axis = 1)
            target_train = train_data[target_column_name]

            feature_test = test_data.drop(columns=[target_column_name], axis = 1)
            target_test = test_data[target_column_name]

            feature_train_arr = preprocessor.fit_transform(feature_train)
            feature_test_arr = preprocessor.transform(feature_test)

            print(type(feature_test_arr))

            logging.info('Imported and preparing train and test data for scaling and encoding')
        except Exception as e:
            raise CustomException(e,sys)


       

    
#seperate it as features and target

#seperate categorical and numerical features in train and test data

#Pipeline test, train features via Onehot encoding and standardscalar transformations using preprocessor.pkl file

#Convert data to np_array

#concat feature array and target array for test and train data

#return train_arr and test_arr