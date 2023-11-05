import pandas as pd
import numpy as np
import os
from src.DimondPricePrediction.logger import logging
from src.DimondPricePrediction.exception import ExceptionHandler

from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from pathlib import Path

class DataIngestionConfig:
    raw_data_path: str=os.path.join("artifacts","raw.csv")
    train_data_path: str=os.path.join("artifacts","train.csv")
    test_data_path: str=os.path.join("artifacts","test.csv")

class DataIngestion:
    def __init__(self):
        self.config = DataIngestionConfig()
    
    def initiate_data_ingestion(self):
        logging.info("Initiating data ingestion")
        try:
            data = pd.read_csv(Path(os.path.join("notebooks/data","gemstone.csv")))
            logging.info("Data ingestion completed")

            os.makedirs(os.path.dirname(self.config.raw_data_path), exist_ok=True)
            data.to_csv(self.config.raw_data_path, index=False)
            logging.info("Raw data saved successfully")

            logging.info("perform train test split")
            train_data, test_data = train_test_split(data, test_size=0.25)
            logging.info("train test split completed")
            train_data.to_csv(self.config.train_data_path, index=False)
            test_data.to_csv(self.config.test_data_path, index=False)
            logging.info("Train and test data saved successfully")
            return (
                self.config.train_data_path,
                self.config.test_data_path
            )
        except ExceptionHandler as e:
            pass