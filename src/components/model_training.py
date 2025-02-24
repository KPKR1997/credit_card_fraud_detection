import os
import sys
from dataclasses import dataclass
import seaborn as sns
import matplotlib.pyplot as plt


from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import GridSearchCV


@dataclass
class ModelTrainingConfig:
    model_file_path = os.path.join('artifacts', 'model.pkl')


class ModelTraining:
    def __init__(self):
        self.model_path = ModelTrainingConfig()

    def initiate_model_training(self, train_arr, test_arr):
        try:
            X_train = train_arr[:, :-1]
            y_train = train_arr[:, -1]
            X_test = test_arr[:, :-1]
            y_test = test_arr[:, -1]

            models = {
                'Random Forest': RandomForestClassifier(),
                'Gradient boost': GradientBoostingClassifier(),
                'Logistic Regression': LogisticRegression(),
                'Decision tree': DecisionTreeClassifier()
            }

            param_grids = {
                'Random Forest': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [None, 10, 20],
                    'min_samples_split': [2, 5, 10]
                },
                'Gradient boost': {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 5, 10]
                },
                'Logistic Regression': {
                    'C': [0.01, 0.1, 1, 10],
                    'solver': ['liblinear', 'lbfgs']
                },
                'Decision tree': {
                    'max_depth': [None, 10, 20],
                    'min_samples_split': [2, 5, 10]
                }
            }

            best_models = {}
            best_params = {}

            scoring = {
                'AUC-ROC': 'roc_auc',
                'Precision': make_scorer(precision_score),
                'Recall': make_scorer(recall_score),
                'F1 Score': make_scorer(f1_score),
                'accuracy': 'accuracy'
                }

            # Perform GridSearchCV for each model and store the best models
            for name, model in models.items():
                logging.info(f"Running GridSearchCV for {name}...")
                grid_search = GridSearchCV(
                    estimator=model,
                    param_grid=param_grids[name],
                    cv=5,
                    scoring=scoring,  
                    refit='AUC-ROC', 
                    n_jobs=-1,
                    verbose=2
                )

                grid_search.fit(X_train, y_train)

                best_models[name] = grid_search.best_estimator_
                best_params[name] = grid_search.best_params_

                logging.info(f"Best params for {name}: {grid_search.best_params_}")
                logging.info(f"Best AUC-ROC for {name}: {grid_search.best_score_}")

                logging.info(f"Best Accuracy for {name}: {grid_search.cv_results_['mean_test_accuracy'][grid_search.best_index_]}")
                logging.info(f"Best Precision for {name}: {grid_search.cv_results_['mean_test_Precision'][grid_search.best_index_]}")
                logging.info(f"Best Recall for {name}: {grid_search.cv_results_['mean_test_Recall'][grid_search.best_index_]}")
                logging.info(f"Best F1 Score for {name}: {grid_search.cv_results_['mean_test_F1 Score'][grid_search.best_index_]}")


            # Select the best model based on AUC-ROC score from grid search
            best_model_name = max(best_models, key=lambda model_name: best_models[model_name].score(X_test, y_test))
            best_model = best_models[best_model_name]
            
            logging.info(f"Best model selected: {best_model_name}")

            # Train the best model on the entire training data and evaluate on the test data
            best_model.fit(X_test, y_test)
            test_score = best_model.score(X_test, y_test)

            logging.info(f"Test score for {best_model_name}: {test_score}")

            # Save the best model

            save_object(
                file_path=self.model_path.model_file_path,
                obj = best_model
            )
            

            return best_model, test_score

        except Exception as e:
            raise CustomException(e, sys)
