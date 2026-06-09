import os,sys
from dataclasses import dataclass
from catboost import CatBoostRegressor
from sklearn.ensemble import AdaBoostRegressor,GradientBoostingRegressor,RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object,eval_model


@dataclass 
class model_trainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")
class model_trainer:
    def __init__(self):
        self.model_trainer_config=model_trainerConfig()

    def initiate_model_trainer(self,train_arr,test_arr):
        try:
            logging.info("spliting training and test input data")
            x_train,y_train,x_test,y_test=(
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]
            )
            models={
                "RandomForestRegressor":RandomForestRegressor(),
                "DecisionTreeRegressor":DecisionTreeRegressor(),
                "GradientBoostingRegressor":GradientBoostingRegressor(),
                "LinearRegression":LinearRegression(),
                "KNeighborsRegressor":KNeighborsRegressor(),
                "XGBRegressor":XGBRegressor(),
                "CatBoostRegressor":CatBoostRegressor(),
                "AdaBoostRegressor":AdaBoostRegressor()
            }

            model_report:dict=eval_model(x_train=x_train,y_train=y_train,x_test=x_test,y_test=y_test,models=models)
            best_model_score=max(sorted(model_report.values()))
            best_model_name=list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            best_model=models[best_model_name]
            if best_model_score<0.6:
                raise CustomException("no best model")
            logging.info("best model found")
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            pred=best_model.predict(x_test)
            r2=r2_score(y_test,pred)
            return r2

        except Exception as e:
            raise CustomException(e,sys)