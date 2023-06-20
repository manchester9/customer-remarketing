import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.ingestion.ingestion import DataIngestionConfig
from src.ingestion.ingestion import DataIngestion

from src.transformation.transformation import DataTransformation
from src.transformation.transformation import DataTransformationConfig

from src.train.trainer import ModelTrainerConfig
from src.train.trainer import ModelTrainer

if __name__=="__main__":
    obj=DataIngestion()
    #obj.initiate_data_ingestion()
    train_data,test_data=obj.initiate_data_ingestion()

    data_transformation=DataTransformation()
    train_arr,test_arr,_=data_transformation.initiate_data_transformation(train_data,test_data)

    modeltrainer=ModelTrainer()
    print(modeltrainer.initiate_model_trainer(train_arr,test_arr))