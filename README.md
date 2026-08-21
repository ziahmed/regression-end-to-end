# Student Performance Prediction Project

## Overview
This project aims to predict students' O-level mathematics examination scores using machine learning. The goal is to identify weaker students prior to the examination, enabling the school to provide targeted support.

## Dataset
- **Source**: Student records with 18 features
- **Target**: `final_test` (O-level mathematics score)
- **Size**: 15,001 records

## Project Structure
root/
├── eda.ipynb # Exploratory Data Analysis notebook
├── README.md # Project documentation
├── requirements.txt # Python dependencies
├── data/
│ └── data.csv # Dataset
├── src/
│ ├── data_preparation.py # Data cleaning & preprocessing
│ ├── model_training.py # Model training & evaluation
│ └── config.yaml # Configuration parameters
└── main.py # Main execution script

## Setup & Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/student-performance-prediction.git
cd student-performance-prediction

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

Usage

# Run the complete pipeline
python main.py

# View EDA notebook
jupyter notebook eda.ipynb

Results
Best Model: Random Forest Regressor

Test R²: ~0.85

Test RMSE: ~8.5

At-Risk Students Identified: [Number] students with predicted score < 50

Key Features
Data Preprocessing: Handles missing values, encodes categorical variables, and engineers features

Model Training: Evaluates Linear Regression, Decision Tree, and Random Forest

Hyperparameter Tuning: Optimizes Random Forest using GridSearchCV

At-Risk Identification: Flags students needing intervention

Results Interpretation
R² Score (0.87): Model explains 87% of variance in student scores

RMSE (8.1): Predictions are off by ~8 points on average

At-Risk Students: Identifies weakest students for targeted support

License
MIT

Author
Zia Ahmed

Acknowledgments
Dataset provided for O-level mathematics prediction task

Built with scikit-learn, pandas, and matplotlib