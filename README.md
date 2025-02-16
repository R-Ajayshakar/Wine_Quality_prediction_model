# Wine Quality Prediction Model

## Overview
This repository contains a machine learning model for predicting wine quality based on various physicochemical properties. The model is implemented using Python and utilizes multiple classification techniques.

## Dataset
The dataset used is `winequalityN.csv`, which contains features such as:
- Fixed Acidity
- Volatile Acidity
- Citric Acid
- Residual Sugar
- Chlorides
- Free Sulfur Dioxide
- Total Sulfur Dioxide
- Density
- pH
- Sulphates
- Alcohol
- Quality (Target Variable)

## Dependencies
Ensure you have the following dependencies installed:

```bash
pip install numpy pandas scikit-learn matplotlib seaborn xgboost
```

## Model Implementation
The notebook follows these steps:
1. **Data Loading & Exploration**
   - Reads the dataset (`winequalityN.csv`).
   - Performs exploratory data analysis (EDA) using Pandas and Seaborn.
2. **Data Preprocessing**
   - Handles missing values.
   - Applies normalization using `MinMaxScaler`.
   - Splits the dataset into training and testing sets.
3. **Model Selection & Training**
   - Logistic Regression
   - Support Vector Machine (SVM)
   - XGBoost Classifier
4. **Model Evaluation**
   - Accuracy Score
   - Confusion Matrix
   - Classification Report

## Usage
To train and evaluate the model, run the Jupyter Notebook:

```bash
jupyter notebook Wine_Quality_Prediction_Machine_Learning.ipynb
```

To use the trained model for predictions:

```python
from model import predict_wine_quality

sample_data = [[7.4, 0.7, 0.0, 1.9, 0.076, 11.0, 34.0, 0.9978, 3.51, 0.56, 9.4]]
prediction = predict_wine_quality(sample_data)
print("Predicted Quality:", prediction)
```

## Results
The model achieves an accuracy of approximately **XX%** (to be updated based on evaluation results). Feature importance analysis suggests that alcohol content and volatile acidity significantly impact wine quality.

## Future Improvements
- Hyperparameter tuning for better performance.
- Experimenting with deep learning approaches.
- Feature engineering to improve predictive power.

## Contributors
- [Ajay shankar R]

## License
This project is licensed under the MIT License.

