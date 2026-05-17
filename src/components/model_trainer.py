from src.logger import logger

from numpy.random import randint, uniform
import pandas as pd
import numpy as np
import os
import sys
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression,Ridge,Lasso,ElasticNet
from sklearn.metrics import r2_score,mean_squared_error
from xgboost import XGBRegressor
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object,evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
    
    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and testing input data")
            X_train, y_train, X_test, y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            models = {
                "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
                "Linear Regression": LinearRegression(),
                "Ridge Regression": Ridge(alpha=1.0),
                "Lasso Regression": Lasso(alpha=0.01, max_iter=10000),
                "Elastic Net": ElasticNet(alpha=0.01, max_iter=10000),
                "XGBRegressor": XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
            }
            
            params = {
                "XGBRegressor": {
                    "n_estimators": [100,200,300,400,500],
                    "learning_rate": [0.01, 0.05, 0.1, 0.2, 0.3],
                    "max_depth": [3, 5, 7, 10],
                    "subsample": [0.5, 0.6, 0.7, 0.8, 1.0],
                    "colsample_bytree": [0.5, 0.6, 0.7, 0.8,1.0]
            },
                "Random Forest": {
            "n_estimators": [100, 200, 300, 500],
            "max_depth": [None, 5, 10, 20],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ['sqrt', 'log2', None]
}
            }
            model_report: dict = evaluate_models(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models=models,params=params)

            sorted_models = sorted(model_report.items(), key=lambda x: x[1][0], reverse=True)
            best_model_name, (best_r2, best_rmse) = sorted_models[0]
            best_model = models[best_model_name]

            if best_r2 < 0.6:
              raise CustomException("No best model found")

            logging.info("Top 3 models:")
            for i, (model_name, (r2, rmse)) in enumerate(sorted_models[:3], 1):
                logging.info(f"#{i} {model_name}: R2 = {r2:.4f} | RMSE = {rmse:,.2f}")
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            # ✅ Clean — reuse what you already computed
            logging.info(f"Best Model: {best_model_name} | R2 = {best_r2:.4f} | RMSE = {best_rmse:,.2f}")
            return best_r2, best_rmse, sorted_models[:3]
        
        except Exception as e:
            raise CustomException(e, sys)