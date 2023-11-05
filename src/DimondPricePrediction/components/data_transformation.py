import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path

from src.DimondPricePrediction.logger import logging
from src.DimondPricePrediction.exception import ExceptionHandler

from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from src.DimondPricePrediction.utils.utils import save_object

@dataclass
class DataTransformationConfig:
    # train_data_path: str=os.path.join("artifacts","train.csv")
    # test_data_path: str=os.path.join("artifacts","test.csv")
    preprocessor_path: str=os.path.join("artifacts","preprocessor.pkl")


class DataTransformation:
    def __init__(self):
        self.config = DataTransformationConfig()
    
    def get_data_transformation(self):
        try:
            logging.info("Initiating data transformation")
            # define categorical, numerical and target columns
            categorical_features = ['cut', 'color', 'clarity']
            numeric_features = ['carat', 'depth', 'table', 'x', 'y', 'z']

            # define the custome categorical
            cut_cats = ['Fair', 'Good', 'Very Good', 'Premium', 'Ideal']
            color_cats = ['D', 'E', 'F', 'G', 'H', 'I', 'J']
            clarity_cats = ['I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF']

            # define the pipeline for numerical features
            numeric_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())])
            
            # define the pipeline for categorical features
            categorical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('ordinal', OrdinalEncoder(categories=[cut_cats, color_cats, clarity_cats])),
                ('scaler', StandardScaler())])
            
            # define the preprocessor
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', numeric_transformer, numeric_features),
                    ('cat', categorical_transformer, categorical_features)])
            
            return preprocessor
            
        except Exception as e:
            logging.error("Error while performing data transformation")
            raise ExceptionHandler(e, sys)
    
    def initiate_data_transformation(self, train_data_path, test_data_path):
        try:
            train_data = pd.read_csv(train_data_path)
            test_data = pd.read_csv(test_data_path)
            logging.info("Data ingestion completed")
            logging.info(f'Train dataframe head: {train_data.head()}')
            logging.info(f'Test dataframe head: {test_data.head()}')

            preprocessing_obj = self.get_data_transformation()

            target_column_name = 'price'
            drop_cols = [target_column_name, 'id']

            input_feature_train_df = train_data.drop(drop_cols, axis=1)
            target_feature_train_df = train_data[target_column_name]

            input_feature_test_df = test_data.drop(drop_cols, axis=1)
            target_feature_test_df = test_data[target_column_name]

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]
            save_object(
                file_path=self.config.preprocessor_path,
                obj=preprocessing_obj
            )

            return (
                train_arr,
                test_arr
            )

        except Exception as e:
            logging.error("Error while performing data transformation")
            raise ExceptionHandler(e, sys)