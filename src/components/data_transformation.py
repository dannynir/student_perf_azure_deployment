import sys
import os
import numpy as np
import pandas as pd
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

from dataclasses import dataclass



@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            numerical_columns = ["writing_score", "reading_score"]
            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
            ]

            num_pipeline = Pipeline(
                steps= [
                    ('Imputer',SimpleImputer(strategy='median')),
                    ('scaler',StandardScaler())
                ])
            
            cat_pipeline = Pipeline(
                steps=[
                    ('Imputer',SimpleImputer(strategy='most_frequent')),
                    ('one_hot_encoder',OneHotEncoder())
                ]
            )

            preprocessor = ColumnTransformer(
                [
                ('num_transform',num_pipeline,numerical_columns),
                ('cat_transform',cat_pipeline,categorical_columns)
                ]
            )
            logging.info("preprocessor done for numeric cols:{0} and cat cols:{1}".format(numerical_columns,categorical_columns))
            return preprocessor
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Train read with {0} rows from {1}".format(train_df.shape[0],train_path))
            logging.info("Test read with {0} rows from {1}".format(test_df.shape[0],test_path))

            numeric_columns = train_df.select_dtypes(exclude="object").columns
            cat_columns = train_df.select_dtypes(include="object").columns
            target_col = "math_score"

            input_train_feature_df = train_df.drop(columns=[target_col],axis = 1)
            target_train_feature_df = train_df[target_col].copy()

            input_test_feature_df = test_df.drop(columns=[target_col],axis = 1)
            target_test_feature_df = test_df[target_col].copy()

            preprocessing_obj = self.get_data_transformer_object()
            input_train_feature_array = preprocessing_obj.fit_transform(input_train_feature_df)
            input_test_feature_array = preprocessing_obj.fit_transform(input_test_feature_df)
            logging.info("Preprocessing done on train and test")

            train_arr = np.c_[input_train_feature_array,np.array(target_train_feature_df)]
            test_arr = np.c_[input_test_feature_array,np.array(target_test_feature_df)]

            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessing_obj
            )
            logging.info("Preprocessing object is saved to {}".format(self.data_transformation_config.preprocessor_obj_file_path))

            return (train_arr,test_arr,self.data_transformation_config.preprocessor_obj_file_path)

        except Exception as e:
            raise CustomException(e,sys)
        


        