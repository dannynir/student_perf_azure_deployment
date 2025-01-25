import sys
import pandas as pd
import os

from src.exception import CustomException
from src.utils import load_object
from src.logger import logging

class PredictPipeline:
    def __init__(self, base_dir):
        self.base_dir = base_dir 

    def predict(self,features):
        try:
            #model_path = 'artifacts\model.pkl'
            model_path = os.path.join(self.base_dir, 'artifacts', 'model.pkl')
            #preprocessor_path = 'artifacts\preprocessor.pkl'
            preprocessor_path = os.path.join(self.base_dir, 'artifacts', 'preprocessor.pkl')

            model = load_object(file_path=model_path)
            logging.info("Model read from :{}".format(model_path))
            preprocesor = load_object(file_path=preprocessor_path)
            logging.info("preprocesor read from :{}".format(preprocessor_path))

            data_scaled = preprocesor.transform(features)
            preds = model.predict(data_scaled)
            return preds
        except Exception as e:
            raise CustomException(e,sys)
        

class CustomData:
    def __init__(self,
                 gender:str,
                 race_ethnicity:str,
                 parental_level_of_education:str,
                 lunch:str,
                 test_preparation_course:str,
                 reading_score: int,
                 writing_score: int):
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education=parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score

    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                "gender":[self.gender],
                "race_ethnicity":[self.race_ethnicity],
                "parental_level_of_education":[self.parental_level_of_education],
                "lunch":[self.lunch],
                "test_preparation_course": [self.test_preparation_course],
                "reading_score":[self.reading_score],
                "writing_score":[self.writing_score] 
                }
            logging.info("Input from UI is:{}".format(custom_data_input_dict))
            return pd.DataFrame(custom_data_input_dict)
        except Exception as e:
            raise CustomException(e,sys)





        
         