import sys


from src.exception import CustomException
from src.logger import logging



from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation
from src.components.data_transformation import DataTransformation
from src.components.model_training import ModelTraining


def pipeline():
        try:
            ingestion = DataIngestion()
            ingestion_data = ingestion.initiate_data_ingestion()
            validation_data = DataValidation(ingestion_data)
            validation_data.default_data_validation()
            custom_validated_data = validation_data.custom_data_validation()
            ingestion.split_train_test(custom_validated_data)

            transformation = DataTransformation()
            train, test, _ = transformation.initiate_data_transformation()
            training = ModelTraining()
            best_model, test_score = training.initiate_model_training(train, test)

            print(best_model, test_score)

        except Exception as e:
            raise CustomException(e,sys)



      
if __name__ == '__main__':
    pipeline()