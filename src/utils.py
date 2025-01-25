import os
import sys

import numpy as np 
import pandas as pd
import pickle
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException
from src.logger import logging

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    
def load_object(file_path):
    try:
        with open(file_path,"rb") as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        raise CustomException(e,sys)


def evaluate_model(X_train,y_train,X_test,y_test,models):
    try:
        report = {}
        for model in models:
            model_name = models[model]['model_func']
            params = models[model]['params']
            gs=GridSearchCV(model_name,params,cv=3)
            gs.fit(X_train,y_train)
            
            model_name.set_params(**gs.best_params_)
            model_name.fit(X_train,y_train)

            y_train_pred = model_name.predict(X_train)
            y_test_pred = model_name.predict(X_test)

            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)

            report[model]=test_model_score
            logging.info("Model name {0} Model Params {1}".format(model_name,params))
        return report

    except Exception as e:
        raise CustomException(e,sys)
    
