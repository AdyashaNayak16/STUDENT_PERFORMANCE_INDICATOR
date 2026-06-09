import sys,os
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

class DatatransformationConfig:
    preprocessor_ob_file_path=os.path.join("artifacts","preprocess.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DatatransformationConfig()
    def get_data_tranformer_object(self):
        try:
            num_fea=[ 'reading_score', 'writing_score']
            cat_fea=['gender', 'race_ethnicity', 'parental_level_of_education', 'lunch', 'test_preparation_course']
            num_pipeline=Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="median")),
                    ("scalar",StandardScaler())
                ]
            )
            cat_pipeline=Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="most_frequent")),
                    ("encoding",OneHotEncoder()),
                    ("scalar",StandardScaler(with_mean=False))
                ]

            )
            logging.info("numerical columns standard scaling completed")
            logging.info("categorical columns encoding completed")
            preprocessor=ColumnTransformer(
                [
                    ("num_pipeline",num_pipeline,num_fea),
                    ("cat_pipeline",cat_pipeline,cat_fea)
                ]
            )
            return preprocessor
        except Exception as e:
            raise CustomException(e,sys)
    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)
            logging.info("read train and test data")
            logging.info("obtaining preprocessing object")
            preprocessing_object=self.get_data_tranformer_object()

            target_column='math_score'
            num_column=[ 'reading_score', 'writing_score']

            input_feature_train_df=train_df.drop(columns=[target_column])
            target_feature_train_df=train_df[target_column]


            input_feature_test_df=test_df.drop(columns=[target_column])
            target_feature_test_df=test_df[target_column]
            logging.info("applying the preprocessing on training and test data")

            input_feature_train_arr=preprocessing_object.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_object.transform(input_feature_test_df)


            train_arr=np.c_[input_feature_train_arr,np.array(target_feature_train_df)]
            test_arr=np.c_[input_feature_test_arr,np.array(target_feature_test_df)]

            logging.info("saved the object")
            save_object(
                file_path=self.data_transformation_config.preprocessor_ob_file_path,
                obj=preprocessing_object
            )
        
            return( train_arr,test_arr,self.data_transformation_config.preprocessor_ob_file_path)
        except Exception as e:
            raise CustomException(e,sys)
            