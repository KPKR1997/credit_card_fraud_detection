import os
import sys

import numpy as np
import pandas as pd
import dill
from sklearn.metrics import r2_score
from sqlalchemy import create_engine

from src.exception import CustomException
import configuration.config as config



def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e, sys)
    

def load_object(file_path):
    try:
        with open(file_path, 'rb') as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise CustomException(e,sys)
    

def call_mysql(db, table, q):
    try:
        engine = create_engine(f"mysql+pymysql://{config.USER}:{config.PASSWORD}@{config.HOST}:{config.PORT}/{db}")
        if q == '*':
            query = f"SELECT * FROM {table};"
        else:
            query = q
        data_raw = pd.read_sql(query, engine)
        data = pd.DataFrame(data_raw)
        return data
    except Exception as e:
        print(f"Error collecting data: {e}")
