import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
from src.utils import evaluate_model

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Split training and testing data")
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            # models = {
            #     "Random Forest":RandomForestRegressor(),
            #     "Decision Tree":DecisionTreeRegressor(),
            #     "Gradient Boosting":GradientBoostingRegressor(),
            #     "Linear Regression":LinearRegression(),
            #     "K-Neighbors Classifiers":KNeighborsRegressor(),
            #     "XGBClassifier":XGBRegressor(),
            #     "CatBoosting Classifier":CatBoostRegressor(verbose=False),
            #     "AdaBoost Classifier":AdaBoostRegressor(),
            # }
            models = {
                "Random Forest": {
                    "model_func":RandomForestRegressor(),
                    "params":{
                        # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                        # 'max_features':['sqrt','log2',None],
                        # 'n_estimators': [8,16,32,64,128,256]
                        'n_estimators': [8,16]
                    }
                },
                "Decision Tree": {
                    "model_func":DecisionTreeRegressor(),
                    "params":{
                        'criterion':['squared_error', 'friedman_mse']
                        #'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                        # 'splitter':['best','random'],
                        # 'max_features':['sqrt','log2'],
                    }
                },
                "Gradient Boosting": {
                    "model_func":GradientBoostingRegressor(),
                    "params":{
                        # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                        'learning_rate':[.1,.05],
                        'subsample':[0.6,0.9],
                        # 'criterion':['squared_error', 'friedman_mse'],
                        # 'max_features':['auto','sqrt','log2'],
                        'n_estimators': [8,32]
                        }
                },                
                "Linear Regression": {
                    "model_func":LinearRegression(),
                    "params":{}
                }
                # "K-Neighbors Classifiers":KNeighborsRegressor(),
                # "XGBClassifier":XGBRegressor(),
                # "CatBoosting Classifier":CatBoostRegressor(verbose=False),
                # "AdaBoost Classifier":AdaBoostRegressor(),
            }

            model_report:dict=evaluate_model(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,models=models)

            best_model_score=max(model_report.values())
            best_model_name=list(model_report.keys())[list(model_report.values()).index(best_model_score)]

            best_model=models[best_model_name]["model_func"]

            if best_model_score<0.6:
                raise CustomException("No best model found")
            logging.info("Best model {0} found on training and test datasets".format(best_model_name))

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted=best_model.predict(X_test)
            
            return best_model_score
        
        except Exception as e:
            logging.info(CustomException(e,sys))
            raise CustomException(e,sys)
