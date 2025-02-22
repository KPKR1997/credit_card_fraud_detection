import sys


from src.exception import CustomException
from src.logger import logging
from configuration import config


from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation
from src.components.data_transformation import DataTransformation


def main():
        try:
            ingestion = DataIngestion()
            ingestion_data = ingestion.initiate_data_ingestion()
            validation_data = DataValidation(ingestion_data)
            validation_data.default_data_validation()
            custom_validated_data = validation_data.custom_data_validation()
            ingestion.split_train_test(custom_validated_data)

            transformation = DataTransformation()
            transformation.make_transformer()
            transformation.initiate_data_transformation()
        except Exception as e:
            raise CustomException(e,sys)



      
if __name__ == '__main__':
    main()