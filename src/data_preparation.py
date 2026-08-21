"""
Data Preparation Module
Handles loading, cleaning, preprocessing, and feature engineering
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import yaml
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataPreparator:
    def __init__(self, config_path="src/config.yaml"):
        """Initialize with configuration"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        self.label_encoders = {}
        self.preprocessor = None
        
    def load_data(self):
        """Load dataset from CSV"""
        logger.info(f"Loading data from {self.config['data']['file_path']}")
        self.df = pd.read_csv(self.config['data']['file_path'])
        logger.info(f"Dataset shape: {self.df.shape}")
        return self.df
    
    def clean_data(self):
        """Clean and standardize data"""
        logger.info("Cleaning data...")
        df = self.df.copy()
        
        # Standardize categorical values
        for col in ['tuition', 'gender', 'direct_admission', 'mode_of_transport']:
            if col in df.columns:
                df[col] = df[col].replace({'Y': 'Yes', 'N': 'No'})
        
        # Standardize CCA and learning_style
        for col in ['CCA', 'learning_style']:
            if col in df.columns:
                df[col] = df[col].str.upper()
        
        # Convert time columns to minutes
        def time_to_minutes(time_str):
            if pd.isna(time_str):
                return np.nan
            try:
                parts = str(time_str).split(':')
                if len(parts) == 2:
                    return int(parts[0]) * 60 + int(parts[1])
                return np.nan
            except:
                return np.nan
        
        # Calculate sleep duration
        if 'sleep_time' in df.columns and 'wake_time' in df.columns:
            df['sleep_time_minutes'] = df['sleep_time'].apply(time_to_minutes)
            df['wake_time_minutes'] = df['wake_time'].apply(time_to_minutes)
            df['sleep_duration'] = df['wake_time_minutes'] - df['sleep_time_minutes']
            df.loc[df['sleep_duration'] < 0, 'sleep_duration'] += 1440
            df = df.drop(['sleep_time', 'wake_time'], axis=1)
        
        # Handle negative age values
        if 'age' in df.columns:
            df.loc[df['age'] < 0, 'age'] = df['age'].abs()
        
        self.df = df
        return df
    
    def engineer_features(self):
        """Create new features"""
        logger.info("Engineering features...")
        df = self.df.copy()
        
        # Gender ratio
        df['gender_ratio'] = df['n_male'] / (df['n_female'] + 1)
        
        # Has siblings
        df['has_siblings'] = (df['number_of_siblings'] > 0).astype(int)
        
        # Study efficiency
        df['study_efficiency'] = df['hours_per_week'] / (df['age'] + 1)
        
        # Study effort
        df['study_effort'] = df['hours_per_week'] * (df['attendance_rate'] / 100)
        
        # Sleep quality (if available)
        if 'sleep_duration' in df.columns:
            df['sleep_quality'] = df['sleep_duration'].apply(
                lambda x: 1 if pd.notna(x) and 420 <= x <= 540 else 
                          0 if pd.isna(x) else -1
            )
        
        self.df = df
        return df
    
    def encode_categorical(self):
        """Encode categorical variables"""
        logger.info("Encoding categorical variables...")
        df = self.df.copy()
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        # Remove target and identifier columns
        target = self.config['data']['target_column']
        categorical_cols = [col for col in categorical_cols 
                          if col not in [target, 'student_id']]
        
        for col in categorical_cols:
            le = LabelEncoder()
            df[col] = df[col].fillna('Unknown')
            df[col] = le.fit_transform(df[col].astype(str))
            self.label_encoders[col] = le
            logger.info(f"Encoded {col} with {len(le.classes_)} classes")
        
        self.df = df
        return df
    
    def prepare_features(self):
        """Prepare features and target for modeling"""
        logger.info("Preparing features...")
        target = self.config['data']['target_column']
        
        # Drop unnecessary columns
        X = self.df.drop([target, 'index', 'student_id'], axis=1, errors='ignore')
        y = self.df[target]
        
        # Replace infinite values
        X = X.replace([np.inf, -np.inf], np.nan)
        
        # Remove rows with missing target
        valid_indices = y.notna()
        X = X[valid_indices]
        y = y[valid_indices]
        
        # Identify feature types
        self.numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_features = X.select_dtypes(include=['object']).columns.tolist()
        
        self.X = X
        self.y = y
        return X, y
    
    def create_preprocessor(self):
        """Create preprocessing pipeline"""
        logger.info("Creating preprocessor...")
        
        # Numeric pipeline
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(
                strategy=self.config['preprocessing']['numeric_imputation_strategy']
            )),
            ('scaler', StandardScaler())
        ])
        
        # Categorical pipeline
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(
                strategy=self.config['preprocessing']['categorical_imputation_strategy']
            )),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        
        # Combine pipelines
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, self.numeric_features),
                ('cat', categorical_transformer, self.categorical_features)
            ])
        
        return self.preprocessor
    
    def split_data(self):
        """Split data into train and test sets"""
        logger.info("Splitting data...")
        test_size = self.config['data']['test_size']
        random_state = self.config['data']['random_state']
        
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state
        )
        
        logger.info(f"Training set: {X_train.shape[0]} samples")
        logger.info(f"Test set: {X_test.shape[0]} samples")
        
        return X_train, X_test, y_train, y_test
    
    def run_pipeline(self):
        """Execute complete data preparation pipeline"""
        self.load_data()
        self.clean_data()
        self.engineer_features()
        self.encode_categorical()
        X, y = self.prepare_features()
        self.create_preprocessor()
        return self.split_data()