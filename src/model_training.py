"""
Model Training Module
Handles model training, evaluation, hyperparameter tuning, and prediction
"""

import pandas as pd
import numpy as np
import joblib
import yaml
import logging
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelTrainer:
    def __init__(self, config_path="src/config.yaml"):
        """Initialize with configuration"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        self.models = {}
        self.results = {}
        self.best_model = None
        self.best_model_name = None
        
    def create_models(self):
        """Initialize models with configurations"""
        logger.info("Creating models...")
        
        # Linear Regression
        if self.config['models']['linear_regression']['enabled']:
            self.models['Linear Regression'] = LinearRegression()
            logger.info("Added Linear Regression")
        
        # Decision Tree
        if self.config['models']['decision_tree']['enabled']:
            dt_config = self.config['models']['decision_tree']
            self.models['Decision Tree'] = DecisionTreeRegressor(
                random_state=self.config['data']['random_state'],
                max_depth=dt_config.get('max_depth', 15)
            )
            logger.info("Added Decision Tree")
        
        # Random Forest
        if self.config['models']['random_forest']['enabled']:
            rf_config = self.config['models']['random_forest']
            self.models['Random Forest'] = RandomForestRegressor(
                random_state=self.config['data']['random_state'],
                n_estimators=rf_config.get('n_estimators', 100),
                max_depth=rf_config.get('max_depth', 15)
            )
            logger.info("Added Random Forest")
        
        return self.models
    
    def train_models(self, X_train, y_train, preprocessor):
        """Train all models"""
        logger.info("Training models...")
        
        for name, model in self.models.items():
            logger.info(f"Training {name}...")
            pipeline = Pipeline([
                ('preprocessor', preprocessor),
                ('regressor', model)
            ])
            pipeline.fit(X_train, y_train)
            self.models[name] = pipeline
            logger.info(f"Completed {name}")
        
        return self.models
    
    def evaluate_models(self, X_train, y_train, X_test, y_test):
        """Evaluate all models"""
        logger.info("Evaluating models...")
        
        for name, pipeline in self.models.items():
            # Predictions
            y_train_pred = pipeline.predict(X_train)
            y_test_pred = pipeline.predict(X_test)
            
            # Metrics
            self.results[name] = {
                'Train R²': r2_score(y_train, y_train_pred),
                'Test R²': r2_score(y_test, y_test_pred),
                'Train RMSE': np.sqrt(mean_squared_error(y_train, y_train_pred)),
                'Test RMSE': np.sqrt(mean_squared_error(y_test, y_test_pred)),
                'Train MAE': mean_absolute_error(y_train, y_train_pred),
                'Test MAE': mean_absolute_error(y_test, y_test_pred)
            }
            
            logger.info(f"{name}: Test R² = {self.results[name]['Test R²']:.4f}")
        
        return self.results
    
    def hyperparameter_tuning(self, X_train, y_train, preprocessor):
        """Perform hyperparameter tuning for Random Forest"""
        logger.info("Performing hyperparameter tuning...")
        
        if not self.config['grid_search']['enabled']:
            logger.info("Grid search disabled in config")
            return None
        
        param_grid = {}
        rf_config = self.config['grid_search']['param_grid']
        for key, value in rf_config.items():
            # Convert 'null' to None
            param_grid[f'regressor__{key}'] = [
                None if x == 'null' else x for x in value
            ]
        
        rf_pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', RandomForestRegressor(
                random_state=self.config['data']['random_state']
            ))
        ])
        
        grid_search = GridSearchCV(
            rf_pipeline,
            param_grid,
            cv=self.config['grid_search']['cv_folds'],
            scoring=self.config['grid_search']['scoring'],
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        logger.info(f"Best parameters: {grid_search.best_params_}")
        logger.info(f"Best cross-validation R²: {grid_search.best_score_:.4f}")
        
        self.best_model = grid_search.best_estimator_
        self.best_model_name = 'Tuned Random Forest'
        
        # Evaluate tuned model
        y_test_pred = self.best_model.predict(X_test)
        self.results[self.best_model_name] = {
            'Test R²': r2_score(y_test, y_test_pred),
            'Test RMSE': np.sqrt(mean_squared_error(y_test, y_test_pred)),
            'Test MAE': mean_absolute_error(y_test, y_test_pred)
        }
        
        return self.best_model
    
    def select_best_model(self):
        """Select best performing model"""
        if self.best_model is None:
            # Find best model based on test R²
            best_name = max(self.results.keys(), 
                          key=lambda x: self.results[x]['Test R²'])
            self.best_model_name = best_name
            self.best_model = self.models[best_name]
        
        logger.info(f"Best Model: {self.best_model_name}")
        logger.info(f"Test R²: {self.results[self.best_model_name]['Test R²']:.4f}")
        
        return self.best_model, self.best_model_name
    
    def identify_at_risk_students(self, X, y, threshold=None):
        """Identify students at risk of low performance"""
        if threshold is None:
            threshold = self.config['risk_threshold']
        
        logger.info(f"Identifying students with predicted score < {threshold}...")
        
        predictions = self.best_model.predict(X)
        
        student_data = pd.DataFrame({
            'predicted_score': predictions,
            'actual_score': y,
            'score_gap': y - predictions
        })
        
        at_risk = student_data[student_data['predicted_score'] < threshold]
        at_risk = at_risk.sort_values('predicted_score')
        
        logger.info(f"Found {len(at_risk)} at-risk students")
        
        return at_risk, student_data
    
    def save_models(self, model_path="best_student_performance_model.pkl", 
                   metadata_path="model_metadata.pkl"):
        """Save the best model and metadata"""
        logger.info("Saving models...")
        
        # Save model
        joblib.dump(self.best_model, model_path)
        logger.info(f"Model saved to {model_path}")
        
        # Save metadata
        metadata = {
            'model_name': self.best_model_name,
            'test_r2': self.results[self.best_model_name]['Test R²'],
            'test_rmse': self.results[self.best_model_name]['Test RMSE'],
            'features': self.config.get('features', []),
            'threshold': self.config['risk_threshold']
        }
        
        joblib.dump(metadata, metadata_path)
        logger.info(f"Metadata saved to {metadata_path}")
        
        return metadata
    
    def run_pipeline(self, X_train, y_train, X_test, y_test, preprocessor):
        """Execute complete model training pipeline"""
        self.create_models()
        self.train_models(X_train, y_train, preprocessor)
        self.evaluate_models(X_train, y_train, X_test, y_test)
        self.hyperparameter_tuning(X_train, y_train, preprocessor)
        self.select_best_model()
        return self.best_model, self.results