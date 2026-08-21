"""
Main Execution Script
Orchestrates the complete machine learning pipeline
"""

import logging
import pandas as pd
from src.data_preparation import DataPreparator
from src.model_training import ModelTrainer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Run the complete pipeline"""
    logger.info("="*60)
    logger.info("Student Performance Prediction Pipeline")
    logger.info("="*60)
    
    # Step 1: Data Preparation
    logger.info("\n" + "="*40)
    logger.info("STEP 1: Data Preparation")
    logger.info("="*40)
    
    data_preparator = DataPreparator()
    X_train, X_test, y_train, y_test = data_preparator.run_pipeline()
    
    logger.info(f"Training data shape: {X_train.shape}")
    logger.info(f"Test data shape: {X_test.shape}")
    
    # Step 2: Model Training
    logger.info("\n" + "="*40)
    logger.info("STEP 2: Model Training & Evaluation")
    logger.info("="*40)
    
    model_trainer = ModelTrainer()
    best_model, results = model_trainer.run_pipeline(
        X_train, y_train, X_test, y_test, data_preparator.preprocessor
    )
    
    # Step 3: At-Risk Student Identification
    logger.info("\n" + "="*40)
    logger.info("STEP 3: Identifying At-Risk Students")
    logger.info("="*40)
    
    # Combine X and y for prediction
    X_all = pd.concat([X_train, X_test])
    y_all = pd.concat([y_train, y_test])
    
    at_risk_students, student_predictions = model_trainer.identify_at_risk_students(
        X_all, y_all
    )
    
    logger.info(f"At-risk students identified: {len(at_risk_students)}")
    
    # Step 4: Save Results
    logger.info("\n" + "="*40)
    logger.info("STEP 4: Saving Results")
    logger.info("="*40)
    
    metadata = model_trainer.save_models()
    
    # Step 5: Summary
    logger.info("\n" + "="*60)
    logger.info("PIPELINE EXECUTION COMPLETE")
    logger.info("="*60)
    
    logger.info(f"Best Model: {metadata['model_name']}")
    logger.info(f"Test R²: {metadata['test_r2']:.4f}")
    logger.info(f"Test RMSE: {metadata['test_rmse']:.4f}")
    logger.info(f"At-risk students (score < {metadata['threshold']}): {len(at_risk_students)}")
    
    # Show top 5 at-risk students
    if len(at_risk_students) > 0:
        logger.info("\nTop 5 At-Risk Students:")
        print(at_risk_students.head(5))
    
    return best_model, results, at_risk_students

if __name__ == "__main__":
    main()