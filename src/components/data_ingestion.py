import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

@dataclass
class DataIngestionConfig:
    train_data_path = os.path.join('artifacts','train.csv')
    test_data_path = os.path.join('artifacts','test.csv')
    raw_data_path = os.path.join('artifacts','data.csv')


class DataIngestion:
    def __init__(self):

        self.ingestion_config = DataIngestionConfig()


    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion methon or component")
        try:
            #df=pd.read_csv(r'C:\Users\danny\Learning\mlproject\src\notebook\data\stud.csv')#src\notebook\data\stud.csv
            current_dir = os.path.dirname(os.path.abspath(__file__))
            csv_path = os.path.join(current_dir, '..', 'notebook', 'data', 'stud.csv')
            #df=pd.read_csv('..\notebook\data\stud.csv')#src\notebook\data\stud.csv
            df=pd.read_csv(csv_path)
            #df=pd.read_csv(r'..\notebook\data\stud.csv')
            #logging.info('Read the dataset as dataframe from {}'.format(csv_path))
            logging.info('current_dir {}'.format(os.getcwd()))

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)

            train_set,test_set = train_test_split(df,test_size=0.2,random_state=42)
            
            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            logging.info('Ingestion is complete')

            return(self.ingestion_config.train_data_path,
                   self.ingestion_config.test_data_path)

        except Exception as e:
            raise CustomException(e,sys)

#if __name__ == '__main__':
#    obj = DataIngestion()
#    train_path,test_path = obj.initiate_data_ingestion()
#    DT = DataTransformation()
#    train_array, test_array, _ = DT.initiate_data_transformation(train_path,test_path)

#    MT=ModelTrainer()
#    print(MT.initiate_model_trainer(train_array, test_array))



