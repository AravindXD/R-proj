# Mental Health & Technology Usage Analysis 🧠💻

## Project Overview
This project analyzes the relationship between technology usage patterns and mental health/sleep quality using machine learning techniques. The analysis employs multiple classification and regression models to predict sleep quality and hours respectively based on various technology usage and mental health indicators.

### Key Features
- Multiple machine learning models implementation
- Ensemble learning approach
- Feature importance analysis
- Comprehensive model evaluation
- Data preprocessing and feature engineering

## Dataset

The project uses two main datasets:

- [mental_health_with_sleep_quality.csv](./mental_health_with_sleep_quality.csv)
  [(Feature Engg.)](https://www.kaggle.com/embed/aravindnagarajan/regression?cellIds=8&kernelSessionId=204436371)

- [mental_health_and_technology_usage_2024.csv](./mental_health_and_technology_usage_2024.csv)


## Requirements
### R (>= 4.0.0)
Required R packages:
- tidyverse
- caret
- randomForest
- xgboost
- nnet
- e1071
- ROSE
- rpart
- gbm
- MLmetrics
- kernlab


## Usage Guide
### For R Analysis
1. Install the required R packages:
   ```R
   install.packages(c("tidyverse", "caret", "randomForest", "xgboost", "nnet", "e1071", "ROSE", "rpart", "gbm", "MLmetrics", "kernlab"))
   ```
2. Run ✏️ [FinalClass.Rmd](FinalClass.Rmd) for Classification.
3. Run ✏️ [regression-analysis.ipynb](regression-analysis.ipynb) for regression.


## Model Performance Summary

### Classification Models
| Model                  | Accuracy |
|------------------------|----------|
| Multinomial Regression | 52.58%   |
| Decision Tree          | 52.55%   |
| GBM                    | 52.45%   |
| XGBoost                | 52.25%   |
| Random Forest          | 51.95%   |
| SVM                    | 49.25%   |

### Regression Models
| Model                  | RMSE     |
|------------------------|----------|
| Linear Regression      | 0.45     |
| Decision Tree          | 0.48     |
| Random Forest          | 0.42     |
| XGBoost                | 0.41     |
| GBM                    | 0.43     |
| SVM                    | 0.47     |
