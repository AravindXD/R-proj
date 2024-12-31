---
title: "Classification"
output:
  html_document: 
    keep_md: true
  word_document: default
  pdf_document: default
---

<style>
  /* General page styling */
  body {
    font-family: Arial, sans-serif;
    line-height: 1.6;
    color: #2C3E50;
    background-color: #F7F9FB;
  }

  h1, h2 {
    color: #2980B9;
    border-bottom: 2px solid #2980B9;
    padding-bottom: 5px;
    margin-top: 20px;
  }

  h3, h4 {
    color: #3498DB;
  }

  /* Code block styling */
  code {
    color: #D35400;
    background-color: #FBFCFC;
    padding: 2px 5px;
    border-radius: 4px;
    font-size: 0.9em;
  }

  pre {
    background-color: #FBFCFC;
    border-left: 4px solid #3498DB;
    padding: 10px;
    border-radius: 5px;
    font-size: 0.95em;
  }

  /* Table styling */
  .table {
    border-collapse: collapse;
    width: 100%;
    margin: 20px 0;
  }

  .table th, .table td {
    padding: 10px;
    border: 1px solid #ddd;
    text-align: left;
  }

  .table th {
    background-color: #2980B9;
    color: #FFFFFF;
    font-weight: bold;
  }

  .table tr:nth-child(even) {
    background-color: #F2F2F2;
  }

  /* Custom text color classes */
  .cat {
    color: #5D6D7E;
    font-style: italic;
  }

  /* Highlight important results */
  .highlight {
    background-color: #FCF3CF;
    padding: 5px;
    border-radius: 4px;
  }

  .section-title {
    font-size: 1.2em;
    font-weight: bold;
    color: #34495E;
    margin-top: 15px;
  }
</style>


# Load required libraries


``` r
library(tidyverse)
library(caret)
library(randomForest)
library(xgboost)
library(nnet)
library(e1071)
library(ROSE)
library(rpart)
library(gbm)
library(MLmetrics)
library(kernlab)
```

# Data Loading and Preprocessing


``` r
# Read data
data <- read.csv("mental_health_with_sleep_quality.csv")

# Remove unnecessary columns
data <- data %>% 
  select(-c(User_ID, Sleep_Hours))

# Convert categorical variables to factors
categorical_cols <- c("Gender", "Mental_Health_Status", "Work_Environment_Impact", 
                     "Support_Systems_Access", "Online_Support_Usage", "Sleep_Quality")
data[categorical_cols] <- lapply(data[categorical_cols], factor)

# Feature Engineering
data <- data %>%
  mutate(
    Total_Screen_Time = Technology_Usage_Hours + Social_Media_Usage_Hours + Gaming_Hours,
    Screen_Activity_Ratio = Screen_Time_Hours / (Physical_Activity_Hours + 0.1),
    Tech_Social_Ratio = Technology_Usage_Hours / (Social_Media_Usage_Hours + 0.1)
  )
```

# Data Splitting


``` r
set.seed(123)
trainIndex <- createDataPartition(data$Sleep_Quality, p = 0.7, list = FALSE)
train_data <- data[trainIndex, ]
test_data <- data[-trainIndex, ]

# Create training control
ctrl <- trainControl(
  method = "cv",
  number = 5,
  classProbs = TRUE,
  summaryFunction = multiClassSummary
)
```

# Model 1 - Random Forest


``` r
rf_model <- train(
  Sleep_Quality ~ .,
  data = train_data,
  method = "rf",
  trControl = trainControl(
    method = "cv",
    number = 5,  # Reduced from 10 to 5 for faster execution
    verboseIter = TRUE
  ),
  tuneGrid = expand.grid(mtry = sqrt(ncol(train_data)-1)), # Simplified tuning
  ntree = 100  # Reduced number of trees
)
```

```
## + Fold1: mtry=3.873 
## - Fold1: mtry=3.873 
## + Fold2: mtry=3.873 
## - Fold2: mtry=3.873 
## + Fold3: mtry=3.873 
## - Fold3: mtry=3.873 
## + Fold4: mtry=3.873 
## - Fold4: mtry=3.873 
## + Fold5: mtry=3.873 
## - Fold5: mtry=3.873 
## Aggregating results
## Fitting final model on full training set
```

# Model 2 - XGBoost


``` r
xgb_model <- train(
  Sleep_Quality ~ .,
  data = train_data,
  method = "xgbTree",
  trControl = ctrl,
  metric = "Accuracy",
  tuneGrid = expand.grid(
    nrounds = 100,
    max_depth = 6,
    eta = 0.3,
    gamma = 0,
    colsample_bytree = 1,
    min_child_weight = 1,
    subsample = 1
  ),
  verbose = FALSE
)
```

# Model 3 - Multinomial Logistic Regression


``` r
multinom_model <- train(
  Sleep_Quality ~ .,
  data = train_data,
  method = "multinom",
  trControl = ctrl,
  metric = "Accuracy",
  trace = FALSE
)
```

# Model 4 - Support Vector Machine


``` r
# First, create proper scaling and preprocessing
preProcess_range <- preProcess(train_data[,-which(names(train_data) == "Sleep_Quality")], 
                             method = c("center", "scale"))

# Apply preprocessing to both training and test data
train_data_preprocessed <- predict(preProcess_range, train_data)
test_data_preprocessed <- predict(preProcess_range, test_data)

# Create a more robust control parameter
ctrl_svm <- trainControl(
  method = "cv",
  number = 5,
  classProbs = TRUE,
  verboseIter = TRUE,
  allowParallel = TRUE
)

# Define a specific tuning grid for SVM
svm_grid <- expand.grid(
  sigma = c(0.01, 0.02, 0.025, 0.03, 0.035, 0.04, 0.05),
  C = c(0.5, 1, 2, 4, 8, 16)
)

# Train SVM with modified parameters
set.seed(123)
svm_model <- train(
  Sleep_Quality ~ .,
  data = train_data_preprocessed,
  method = "svmRadial",
  trControl = ctrl_svm,
  tuneGrid = svm_grid,
  preProcess = NULL,  # Already preprocessed
  verbose = FALSE
)
```

```
## + Fold1: sigma=0.010, C= 0.5 
## - Fold1: sigma=0.010, C= 0.5 
## + Fold1: sigma=0.020, C= 0.5 
## - Fold1: sigma=0.020, C= 0.5 
## + Fold1: sigma=0.025, C= 0.5 
## - Fold1: sigma=0.025, C= 0.5 
## + Fold1: sigma=0.030, C= 0.5 
## - Fold1: sigma=0.030, C= 0.5 
## + Fold1: sigma=0.035, C= 0.5 
## - Fold1: sigma=0.035, C= 0.5 
## + Fold1: sigma=0.040, C= 0.5 
## - Fold1: sigma=0.040, C= 0.5 
## + Fold1: sigma=0.050, C= 0.5 
## - Fold1: sigma=0.050, C= 0.5 
## + Fold1: sigma=0.010, C= 1.0 
## - Fold1: sigma=0.010, C= 1.0 
## + Fold1: sigma=0.020, C= 1.0 
## - Fold1: sigma=0.020, C= 1.0 
## + Fold1: sigma=0.025, C= 1.0 
## - Fold1: sigma=0.025, C= 1.0 
## + Fold1: sigma=0.030, C= 1.0 
## - Fold1: sigma=0.030, C= 1.0 
## + Fold1: sigma=0.035, C= 1.0 
## - Fold1: sigma=0.035, C= 1.0 
## + Fold1: sigma=0.040, C= 1.0 
## - Fold1: sigma=0.040, C= 1.0 
## + Fold1: sigma=0.050, C= 1.0 
## - Fold1: sigma=0.050, C= 1.0 
## + Fold1: sigma=0.010, C= 2.0 
## - Fold1: sigma=0.010, C= 2.0 
## + Fold1: sigma=0.020, C= 2.0 
## - Fold1: sigma=0.020, C= 2.0 
## + Fold1: sigma=0.025, C= 2.0 
## - Fold1: sigma=0.025, C= 2.0 
## + Fold1: sigma=0.030, C= 2.0 
## - Fold1: sigma=0.030, C= 2.0 
## + Fold1: sigma=0.035, C= 2.0 
## - Fold1: sigma=0.035, C= 2.0 
## + Fold1: sigma=0.040, C= 2.0 
## line search fails -1.376226 -0.1541639 1.141328e-05 4.562427e-06 -2.503397e-08 -5.97109e-10 -2.884439e-13
```

```
## Warning in method$predict(modelFit = modelFit, newdata = newdata, submodels =
## param): kernlab class prediction calculations failed; returning NAs
```

```
## Warning in method$prob(modelFit = modelFit, newdata = newdata, submodels =
## param): kernlab class probability calculations failed; returning NAs
```

```
## Warning in data.frame(..., check.names = FALSE): row names were found from a
## short variable and have been discarded
```

```
## - Fold1: sigma=0.040, C= 2.0 
## + Fold1: sigma=0.050, C= 2.0 
## - Fold1: sigma=0.050, C= 2.0 
## + Fold1: sigma=0.010, C= 4.0 
## - Fold1: sigma=0.010, C= 4.0 
## + Fold1: sigma=0.020, C= 4.0 
## - Fold1: sigma=0.020, C= 4.0 
## + Fold1: sigma=0.025, C= 4.0 
## - Fold1: sigma=0.025, C= 4.0 
## + Fold1: sigma=0.030, C= 4.0 
## - Fold1: sigma=0.030, C= 4.0 
## + Fold1: sigma=0.035, C= 4.0 
## - Fold1: sigma=0.035, C= 4.0 
## + Fold1: sigma=0.040, C= 4.0 
## - Fold1: sigma=0.040, C= 4.0 
## + Fold1: sigma=0.050, C= 4.0 
## - Fold1: sigma=0.050, C= 4.0 
## + Fold1: sigma=0.010, C= 8.0 
## - Fold1: sigma=0.010, C= 8.0 
## + Fold1: sigma=0.020, C= 8.0 
## - Fold1: sigma=0.020, C= 8.0 
## + Fold1: sigma=0.025, C= 8.0 
## - Fold1: sigma=0.025, C= 8.0 
## + Fold1: sigma=0.030, C= 8.0 
## - Fold1: sigma=0.030, C= 8.0 
## + Fold1: sigma=0.035, C= 8.0 
## line search fails -1.103827 -0.2544857 1.577235e-05 5.412526e-06 -2.465695e-08 -3.024031e-09 -4.052656e-13
```

```
## Warning in method$predict(modelFit = modelFit, newdata = newdata, submodels =
## param): kernlab class prediction calculations failed; returning NAs
```

```
## Warning in method$prob(modelFit = modelFit, newdata = newdata, submodels =
## param): kernlab class probability calculations failed; returning NAs
```

```
## Warning in data.frame(..., check.names = FALSE): row names were found from a
## short variable and have been discarded
```

```
## - Fold1: sigma=0.035, C= 8.0 
## + Fold1: sigma=0.040, C= 8.0 
## line search fails -1.086797 -0.2752001 3.672123e-05 1.235877e-05 -5.608696e-08 -6.622958e-09 -2.141434e-12
```

```
## Warning in method$predict(modelFit = modelFit, newdata = newdata, submodels =
## param): kernlab class prediction calculations failed; returning NAs
```

```
## Warning in method$prob(modelFit = modelFit, newdata = newdata, submodels =
## param): kernlab class probability calculations failed; returning NAs
```

```
## Warning in data.frame(..., check.names = FALSE): row names were found from a
## short variable and have been discarded
```

```
## - Fold1: sigma=0.040, C= 8.0 
## + Fold1: sigma=0.050, C= 8.0 
## - Fold1: sigma=0.050, C= 8.0 
## + Fold1: sigma=0.010, C=16.0 
## - Fold1: sigma=0.010, C=16.0 
## + Fold1: sigma=0.020, C=16.0 
## - Fold1: sigma=0.020, C=16.0 
## + Fold1: sigma=0.025, C=16.0 
## - Fold1: sigma=0.025, C=16.0 
## + Fold1: sigma=0.030, C=16.0 
## - Fold1: sigma=0.030, C=16.0 
## + Fold1: sigma=0.035, C=16.0 
## - Fold1: sigma=0.035, C=16.0 
## + Fold1: sigma=0.040, C=16.0 
## - Fold1: sigma=0.040, C=16.0 
## + Fold1: sigma=0.050, C=16.0 
## - Fold1: sigma=0.050, C=16.0 
## + Fold2: sigma=0.010, C= 0.5 
## - Fold2: sigma=0.010, C= 0.5 
## + Fold2: sigma=0.020, C= 0.5 
## line search fails -1.719261 0.1027207 1.436936e-05 6.306849e-06 -4.47361e-08 5.777024e-09 -6.063944e-13
```

```
## Warning in method$predict(modelFit = modelFit, newdata = newdata, submodels =
## param): kernlab class prediction calculations failed; returning NAs
```

```
## Warning in method$prob(modelFit = modelFit, newdata = newdata, submodels =
## param): kernlab class probability calculations failed; returning NAs
```

```
## Warning in data.frame(..., check.names = FALSE): row names were found from a
## short variable and have been discarded
```

```
## - Fold2: sigma=0.020, C= 0.5 
## + Fold2: sigma=0.025, C= 0.5 
## - Fold2: sigma=0.025, C= 0.5 
## + Fold2: sigma=0.030, C= 0.5 
## - Fold2: sigma=0.030, C= 0.5 
## + Fold2: sigma=0.035, C= 0.5 
## - Fold2: sigma=0.035, C= 0.5 
## + Fold2: sigma=0.040, C= 0.5 
## - Fold2: sigma=0.040, C= 0.5 
## + Fold2: sigma=0.050, C= 0.5 
## - Fold2: sigma=0.050, C= 0.5 
## + Fold2: sigma=0.010, C= 1.0 
## - Fold2: sigma=0.010, C= 1.0 
## + Fold2: sigma=0.020, C= 1.0 
## - Fold2: sigma=0.020, C= 1.0 
## + Fold2: sigma=0.025, C= 1.0 
## line search fails -1.539249 -0.009769775 2.038327e-05 8.385695e-06 -5.285689e-08 3.392399e-09 -1.048949e-12
```

```
## Warning in method$predict(modelFit = modelFit, newdata = newdata, submodels =
## param): kernlab class prediction calculations failed; returning NAs
```

```
## Warning in method$prob(modelFit = modelFit, newdata = newdata, submodels =
## param): kernlab class probability calculations failed; returning NAs
```

```
## Warning in data.frame(..., check.names = FALSE): row names were found from a
## short variable and have been discarded
```

```
## - Fold2: sigma=0.025, C= 1.0 
## + Fold2: sigma=0.030, C= 1.0 
## - Fold2: sigma=0.030, C= 1.0 
## + Fold2: sigma=0.035, C= 1.0 
## line search fails -1.571387 -0.02512287 1.526111e-05 6.403158e-06 -4.089134e-08 2.15254e-09 -6.10264e-13
```

```
## Warning in method$predict(modelFit = modelFit, newdata = newdata, submodels =
## param): kernlab class prediction calculations failed; returning NAs
```

```
## Warning in method$prob(modelFit = modelFit, newdata = newdata, submodels =
## param): kernlab class probability calculations failed; returning NAs
```

```
## Warning in data.frame(..., check.names = FALSE): row names were found from a
## short variable and have been discarded
```

```
## - Fold2: sigma=0.035, C= 1.0 
## + Fold2: sigma=0.040, C= 1.0 
## - Fold2: sigma=0.040, C= 1.0 
## + Fold2: sigma=0.050, C= 1.0 
## - Fold2: sigma=0.050, C= 1.0 
## + Fold2: sigma=0.010, C= 2.0 
## - Fold2: sigma=0.010, C= 2.0 
## + Fold2: sigma=0.020, C= 2.0 
## - Fold2: sigma=0.020, C= 2.0 
## + Fold2: sigma=0.025, C= 2.0 
## - Fold2: sigma=0.025, C= 2.0 
## + Fold2: sigma=0.030, C= 2.0 
## - Fold2: sigma=0.030, C= 2.0 
## + Fold2: sigma=0.035, C= 2.0 
## - Fold2: sigma=0.035, C= 2.0 
## + Fold2: sigma=0.040, C= 2.0 
## - Fold2: sigma=0.040, C= 2.0 
## + Fold2: sigma=0.050, C= 2.0 
## - Fold2: sigma=0.050, C= 2.0 
## + Fold2: sigma=0.010, C= 4.0 
## - Fold2: sigma=0.010, C= 4.0 
## + Fold2: sigma=0.020, C= 4.0 
## - Fold2: sigma=0.020, C= 4.0 
## + Fold2: sigma=0.025, C= 4.0 
## - Fold2: sigma=0.025, C= 4.0 
## + Fold2: sigma=0.030, C= 4.0 
## - Fold2: sigma=0.030, C= 4.0 
## + Fold2: sigma=0.035, C= 4.0 
## - Fold2: sigma=0.035, C= 4.0 
## + Fold2: sigma=0.040, C= 4.0 
## - Fold2: sigma=0.040, C= 4.0 
## + Fold2: sigma=0.050, C= 4.0 
## - Fold2: sigma=0.050, C= 4.0 
## + Fold2: sigma=0.010, C= 8.0 
## - Fold2: sigma=0.010, C= 8.0 
## + Fold2: sigma=0.020, C= 8.0 
## - Fold2: sigma=0.020, C= 8.0 
## + Fold2: sigma=0.025, C= 8.0 
## - Fold2: sigma=0.025, C= 8.0 
## + Fold2: sigma=0.030, C= 8.0 
## - Fold2: sigma=0.030, C= 8.0 
## + Fold2: sigma=0.035, C= 8.0 
## - Fold2: sigma=0.035, C= 8.0 
## + Fold2: sigma=0.040, C= 8.0 
## - Fold2: sigma=0.040, C= 8.0 
## + Fold2: sigma=0.050, C= 8.0 
## - Fold2: sigma=0.050, C= 8.0 
## + Fold2: sigma=0.010, C=16.0 
## - Fold2: sigma=0.010, C=16.0 
## + Fold2: sigma=0.020, C=16.0 
## - Fold2: sigma=0.020, C=16.0 
## + Fold2: sigma=0.025, C=16.0 
## - Fold2: sigma=0.025, C=16.0 
## + Fold2: sigma=0.030, C=16.0 
## - Fold2: sigma=0.030, C=16.0 
## + Fold2: sigma=0.035, C=16.0 
## - Fold2: sigma=0.035, C=16.0 
## + Fold2: sigma=0.040, C=16.0 
## - Fold2: sigma=0.040, C=16.0 
## + Fold2: sigma=0.050, C=16.0 
## - Fold2: sigma=0.050, C=16.0 
## + Fold3: sigma=0.010, C= 0.5 
## line search fails -1.839268 0.1941855 1.507903e-05 7.086538e-06 -5.30206e-08 8.63984e-09 -7.382725e-13
```

```
## Warning in method$predict(modelFit = modelFit, newdata = newdata, submodels =
## param): kernlab class prediction calculations failed; returning NAs
```

```
## Warning in method$prob(modelFit = modelFit, newdata = newdata, submodels =
## param): kernlab class probability calculations failed; returning NAs
```

```
## Warning in data.frame(..., check.names = FALSE): row names were found from a
## short variable and have been discarded
```

```
## - Fold3: sigma=0.010, C= 0.5 
## + Fold3: sigma=0.020, C= 0.5 
## - Fold3: sigma=0.020, C= 0.5 
## + Fold3: sigma=0.025, C= 0.5 
## - Fold3: sigma=0.025, C= 0.5 
## + Fold3: sigma=0.030, C= 0.5 
## - Fold3: sigma=0.030, C= 0.5 
## + Fold3: sigma=0.035, C= 0.5 
## - Fold3: sigma=0.035, C= 0.5 
## + Fold3: sigma=0.040, C= 0.5 
## - Fold3: sigma=0.040, C= 0.5 
## + Fold3: sigma=0.050, C= 0.5 
## - Fold3: sigma=0.050, C= 0.5 
## + Fold3: sigma=0.010, C= 1.0 
## - Fold3: sigma=0.010, C= 1.0 
## + Fold3: sigma=0.020, C= 1.0 
## - Fold3: sigma=0.020, C= 1.0 
## + Fold3: sigma=0.025, C= 1.0 
## - Fold3: sigma=0.025, C= 1.0 
## + Fold3: sigma=0.030, C= 1.0 
## - Fold3: sigma=0.030, C= 1.0 
## + Fold3: sigma=0.035, C= 1.0 
## - Fold3: sigma=0.035, C= 1.0 
## + Fold3: sigma=0.040, C= 1.0 
## - Fold3: sigma=0.040, C= 1.0 
## + Fold3: sigma=0.050, C= 1.0 
## - Fold3: sigma=0.050, C= 1.0 
## + Fold3: sigma=0.010, C= 2.0 
## - Fold3: sigma=0.010, C= 2.0 
## + Fold3: sigma=0.020, C= 2.0 
## line search fails -1.434554 -0.06069309 3.04085e-05 1.20467e-05 -7.104868e-08 2.286291e-09 -2.132941e-12
```

```
## Warning in method$predict(modelFit = modelFit, newdata = newdata, submodels =
## param): kernlab class prediction calculations failed; returning NAs
```

```
## Warning in method$prob(modelFit = modelFit, newdata = newdata, submodels =
## param): kernlab class probability calculations failed; returning NAs
```

```
## Warning in data.frame(..., check.names = FALSE): row names were found from a
## short variable and have been discarded
```

```
## - Fold3: sigma=0.020, C= 2.0 
## + Fold3: sigma=0.025, C= 2.0 
## line search fails -1.39121 -0.07592312 1.428953e-05 5.589866e-06 -3.131905e-08 8.019344e-10 -4.430517e-13
```

```
## Warning in method$predict(modelFit = modelFit, newdata = newdata, submodels =
## param): kernlab class prediction calculations failed; returning NAs
```

```
## Warning in method$prob(modelFit = modelFit, newdata = newdata, submodels =
## param): kernlab class probability calculations failed; returning NAs
```

```
## Warning in data.frame(..., check.names = FALSE): row names were found from a
## short variable and have been discarded
```

```
## - Fold3: sigma=0.025, C= 2.0 
## + Fold3: sigma=0.030, C= 2.0 
## - Fold3: sigma=0.030, C= 2.0 
## + Fold3: sigma=0.035, C= 2.0 
## - Fold3: sigma=0.035, C= 2.0 
## + Fold3: sigma=0.040, C= 2.0 
## - Fold3: sigma=0.040, C= 2.0 
## + Fold3: sigma=0.050, C= 2.0 
## - Fold3: sigma=0.050, C= 2.0 
## + Fold3: sigma=0.010, C= 4.0 
## - Fold3: sigma=0.010, C= 4.0 
## + Fold3: sigma=0.020, C= 4.0 
## - Fold3: sigma=0.020, C= 4.0 
## + Fold3: sigma=0.025, C= 4.0 
## - Fold3: sigma=0.025, C= 4.0 
## + Fold3: sigma=0.030, C= 4.0 
## - Fold3: sigma=0.030, C= 4.0 
## + Fold3: sigma=0.035, C= 4.0 
## - Fold3: sigma=0.035, C= 4.0 
## + Fold3: sigma=0.040, C= 4.0 
## - Fold3: sigma=0.040, C= 4.0 
## + Fold3: sigma=0.050, C= 4.0 
## - Fold3: sigma=0.050, C= 4.0 
## + Fold3: sigma=0.010, C= 8.0 
## - Fold3: sigma=0.010, C= 8.0 
## + Fold3: sigma=0.020, C= 8.0 
## - Fold3: sigma=0.020, C= 8.0 
## + Fold3: sigma=0.025, C= 8.0 
## - Fold3: sigma=0.025, C= 8.0 
## + Fold3: sigma=0.030, C= 8.0 
## - Fold3: sigma=0.030, C= 8.0 
## + Fold3: sigma=0.035, C= 8.0 
## - Fold3: sigma=0.035, C= 8.0 
## + Fold3: sigma=0.040, C= 8.0 
## - Fold3: sigma=0.040, C= 8.0 
## + Fold3: sigma=0.050, C= 8.0 
## - Fold3: sigma=0.050, C= 8.0 
## + Fold3: sigma=0.010, C=16.0 
## - Fold3: sigma=0.010, C=16.0 
## + Fold3: sigma=0.020, C=16.0 
## - Fold3: sigma=0.020, C=16.0 
## + Fold3: sigma=0.025, C=16.0 
## - Fold3: sigma=0.025, C=16.0 
## + Fold3: sigma=0.030, C=16.0 
## - Fold3: sigma=0.030, C=16.0 
## + Fold3: sigma=0.035, C=16.0 
## - Fold3: sigma=0.035, C=16.0 
## + Fold3: sigma=0.040, C=16.0 
## - Fold3: sigma=0.040, C=16.0 
## + Fold3: sigma=0.050, C=16.0 
## - Fold3: sigma=0.050, C=16.0 
## + Fold4: sigma=0.010, C= 0.5 
## line search fails -1.860099 0.2362963 1.594746e-05 7.44886e-06 -5.73602e-08 1.111118e-08 -8.319838e-13
```

```
## Warning in method$predict(modelFit = modelFit, newdata = newdata, submodels =
## param): kernlab class prediction calculations failed; returning NAs
```

```
## Warning in method$prob(modelFit = modelFit, newdata = newdata, submodels =
## param): kernlab class probability calculations failed; returning NAs
```

```
## Warning in data.frame(..., check.names = FALSE): row names were found from a
## short variable and have been discarded
```

```
## - Fold4: sigma=0.010, C= 0.5 
## + Fold4: sigma=0.020, C= 0.5 
## - Fold4: sigma=0.020, C= 0.5 
## + Fold4: sigma=0.025, C= 0.5 
## - Fold4: sigma=0.025, C= 0.5 
## + Fold4: sigma=0.030, C= 0.5 
## - Fold4: sigma=0.030, C= 0.5 
## + Fold4: sigma=0.035, C= 0.5 
## - Fold4: sigma=0.035, C= 0.5 
## + Fold4: sigma=0.040, C= 0.5 
## - Fold4: sigma=0.040, C= 0.5 
## + Fold4: sigma=0.050, C= 0.5 
## - Fold4: sigma=0.050, C= 0.5 
## + Fold4: sigma=0.010, C= 1.0 
## - Fold4: sigma=0.010, C= 1.0 
## + Fold4: sigma=0.020, C= 1.0 
## - Fold4: sigma=0.020, C= 1.0 
## + Fold4: sigma=0.025, C= 1.0 
## - Fold4: sigma=0.025, C= 1.0 
## + Fold4: sigma=0.030, C= 1.0 
## - Fold4: sigma=0.030, C= 1.0 
## + Fold4: sigma=0.035, C= 1.0 
## - Fold4: sigma=0.035, C= 1.0 
## + Fold4: sigma=0.040, C= 1.0 
## - Fold4: sigma=0.040, C= 1.0 
## + Fold4: sigma=0.050, C= 1.0 
## - Fold4: sigma=0.050, C= 1.0 
## + Fold4: sigma=0.010, C= 2.0 
## - Fold4: sigma=0.010, C= 2.0 
## + Fold4: sigma=0.020, C= 2.0 
## - Fold4: sigma=0.020, C= 2.0 
## + Fold4: sigma=0.025, C= 2.0 
## - Fold4: sigma=0.025, C= 2.0 
## + Fold4: sigma=0.030, C= 2.0 
## - Fold4: sigma=0.030, C= 2.0 
## + Fold4: sigma=0.035, C= 2.0 
## - Fold4: sigma=0.035, C= 2.0 
## + Fold4: sigma=0.040, C= 2.0 
## - Fold4: sigma=0.040, C= 2.0 
## + Fold4: sigma=0.050, C= 2.0 
## - Fold4: sigma=0.050, C= 2.0 
## + Fold4: sigma=0.010, C= 4.0 
## - Fold4: sigma=0.010, C= 4.0 
## + Fold4: sigma=0.020, C= 4.0 
## - Fold4: sigma=0.020, C= 4.0 
## + Fold4: sigma=0.025, C= 4.0 
## - Fold4: sigma=0.025, C= 4.0 
## + Fold4: sigma=0.030, C= 4.0 
## - Fold4: sigma=0.030, C= 4.0 
## + Fold4: sigma=0.035, C= 4.0 
## - Fold4: sigma=0.035, C= 4.0 
## + Fold4: sigma=0.040, C= 4.0 
## - Fold4: sigma=0.040, C= 4.0 
## + Fold4: sigma=0.050, C= 4.0 
## - Fold4: sigma=0.050, C= 4.0 
## + Fold4: sigma=0.010, C= 8.0 
## - Fold4: sigma=0.010, C= 8.0 
## + Fold4: sigma=0.020, C= 8.0 
## - Fold4: sigma=0.020, C= 8.0 
## + Fold4: sigma=0.025, C= 8.0 
## - Fold4: sigma=0.025, C= 8.0 
## + Fold4: sigma=0.030, C= 8.0 
## - Fold4: sigma=0.030, C= 8.0 
## + Fold4: sigma=0.035, C= 8.0 
## - Fold4: sigma=0.035, C= 8.0 
## + Fold4: sigma=0.040, C= 8.0 
## - Fold4: sigma=0.040, C= 8.0 
## + Fold4: sigma=0.050, C= 8.0 
## - Fold4: sigma=0.050, C= 8.0 
## + Fold4: sigma=0.010, C=16.0 
## - Fold4: sigma=0.010, C=16.0 
## + Fold4: sigma=0.020, C=16.0 
## - Fold4: sigma=0.020, C=16.0 
## + Fold4: sigma=0.025, C=16.0 
## - Fold4: sigma=0.025, C=16.0 
## + Fold4: sigma=0.030, C=16.0 
## - Fold4: sigma=0.030, C=16.0 
## + Fold4: sigma=0.035, C=16.0 
## line search fails -0.9384687 -0.2536188 2.632307e-05 7.968486e-06 -3.198442e-08 -2.38014e-09 -8.608942e-13
```

```
## Warning in method$predict(modelFit = modelFit, newdata = newdata, submodels =
## param): kernlab class prediction calculations failed; returning NAs
```

```
## Warning in method$prob(modelFit = modelFit, newdata = newdata, submodels =
## param): kernlab class probability calculations failed; returning NAs
```

```
## Warning in data.frame(..., check.names = FALSE): row names were found from a
## short variable and have been discarded
```

```
## - Fold4: sigma=0.035, C=16.0 
## + Fold4: sigma=0.040, C=16.0 
## - Fold4: sigma=0.040, C=16.0 
## + Fold4: sigma=0.050, C=16.0 
## - Fold4: sigma=0.050, C=16.0 
## + Fold5: sigma=0.010, C= 0.5 
## - Fold5: sigma=0.010, C= 0.5 
## + Fold5: sigma=0.020, C= 0.5 
## - Fold5: sigma=0.020, C= 0.5 
## + Fold5: sigma=0.025, C= 0.5 
## - Fold5: sigma=0.025, C= 0.5 
## + Fold5: sigma=0.030, C= 0.5 
## - Fold5: sigma=0.030, C= 0.5 
## + Fold5: sigma=0.035, C= 0.5 
## - Fold5: sigma=0.035, C= 0.5 
## + Fold5: sigma=0.040, C= 0.5 
## - Fold5: sigma=0.040, C= 0.5 
## + Fold5: sigma=0.050, C= 0.5 
## - Fold5: sigma=0.050, C= 0.5 
## + Fold5: sigma=0.010, C= 1.0 
## - Fold5: sigma=0.010, C= 1.0 
## + Fold5: sigma=0.020, C= 1.0 
## - Fold5: sigma=0.020, C= 1.0 
## + Fold5: sigma=0.025, C= 1.0 
## line search fails -1.552098 0.02674804 1.060076e-05 4.388755e-06 -2.792335e-08 2.708124e-09 -2.841234e-13
```

```
## Warning in method$predict(modelFit = modelFit, newdata = newdata, submodels =
## param): kernlab class prediction calculations failed; returning NAs
```

```
## Warning in method$prob(modelFit = modelFit, newdata = newdata, submodels =
## param): kernlab class probability calculations failed; returning NAs
```

```
## Warning in data.frame(..., check.names = FALSE): row names were found from a
## short variable and have been discarded
```

```
## - Fold5: sigma=0.025, C= 1.0 
## + Fold5: sigma=0.030, C= 1.0 
## - Fold5: sigma=0.030, C= 1.0 
## + Fold5: sigma=0.035, C= 1.0 
## - Fold5: sigma=0.035, C= 1.0 
## + Fold5: sigma=0.040, C= 1.0 
## - Fold5: sigma=0.040, C= 1.0 
## + Fold5: sigma=0.050, C= 1.0 
## - Fold5: sigma=0.050, C= 1.0 
## + Fold5: sigma=0.010, C= 2.0 
## - Fold5: sigma=0.010, C= 2.0 
## + Fold5: sigma=0.020, C= 2.0 
## - Fold5: sigma=0.020, C= 2.0 
## + Fold5: sigma=0.025, C= 2.0 
## - Fold5: sigma=0.025, C= 2.0 
## + Fold5: sigma=0.030, C= 2.0 
## line search fails -1.404061 -0.0783257 2.743968e-05 1.06054e-05 -6.21655e-08 1.980913e-09 -1.684793e-12
```

```
## Warning in method$predict(modelFit = modelFit, newdata = newdata, submodels =
## param): kernlab class prediction calculations failed; returning NAs
```

```
## Warning in method$prob(modelFit = modelFit, newdata = newdata, submodels =
## param): kernlab class probability calculations failed; returning NAs
```

```
## Warning in data.frame(..., check.names = FALSE): row names were found from a
## short variable and have been discarded
```

```
## - Fold5: sigma=0.030, C= 2.0 
## + Fold5: sigma=0.035, C= 2.0 
## line search fails -1.388707 -0.121921 1.381091e-05 5.428002e-06 -3.07104e-08 -2.599424e-11 -4.242795e-13
```

```
## Warning in method$predict(modelFit = modelFit, newdata = newdata, submodels =
## param): kernlab class prediction calculations failed; returning NAs
```

```
## Warning in method$prob(modelFit = modelFit, newdata = newdata, submodels =
## param): kernlab class probability calculations failed; returning NAs
```

```
## Warning in data.frame(..., check.names = FALSE): row names were found from a
## short variable and have been discarded
```

```
## - Fold5: sigma=0.035, C= 2.0 
## + Fold5: sigma=0.040, C= 2.0 
## - Fold5: sigma=0.040, C= 2.0 
## + Fold5: sigma=0.050, C= 2.0 
## - Fold5: sigma=0.050, C= 2.0 
## + Fold5: sigma=0.010, C= 4.0 
## - Fold5: sigma=0.010, C= 4.0 
## + Fold5: sigma=0.020, C= 4.0 
## - Fold5: sigma=0.020, C= 4.0 
## + Fold5: sigma=0.025, C= 4.0 
## - Fold5: sigma=0.025, C= 4.0 
## + Fold5: sigma=0.030, C= 4.0 
## - Fold5: sigma=0.030, C= 4.0 
## + Fold5: sigma=0.035, C= 4.0 
## - Fold5: sigma=0.035, C= 4.0 
## + Fold5: sigma=0.040, C= 4.0 
## line search fails -1.201355 -0.1955691 1.129824e-05 4.087703e-06 -2.006066e-08 -1.034554e-09 -2.308792e-13
```

```
## Warning in method$predict(modelFit = modelFit, newdata = newdata, submodels =
## param): kernlab class prediction calculations failed; returning NAs
```

```
## Warning in method$prob(modelFit = modelFit, newdata = newdata, submodels =
## param): kernlab class probability calculations failed; returning NAs
```

```
## Warning in data.frame(..., check.names = FALSE): row names were found from a
## short variable and have been discarded
```

```
## - Fold5: sigma=0.040, C= 4.0 
## + Fold5: sigma=0.050, C= 4.0 
## - Fold5: sigma=0.050, C= 4.0 
## + Fold5: sigma=0.010, C= 8.0 
## - Fold5: sigma=0.010, C= 8.0 
## + Fold5: sigma=0.020, C= 8.0 
## - Fold5: sigma=0.020, C= 8.0 
## + Fold5: sigma=0.025, C= 8.0 
## - Fold5: sigma=0.025, C= 8.0 
## + Fold5: sigma=0.030, C= 8.0 
## - Fold5: sigma=0.030, C= 8.0 
## + Fold5: sigma=0.035, C= 8.0 
## - Fold5: sigma=0.035, C= 8.0 
## + Fold5: sigma=0.040, C= 8.0 
## - Fold5: sigma=0.040, C= 8.0 
## + Fold5: sigma=0.050, C= 8.0 
## line search fails -1.031029 -0.2984478 1.22927e-05 4.260553e-06 -1.723084e-08 -2.449716e-09 -2.222507e-13
```

```
## Warning in method$predict(modelFit = modelFit, newdata = newdata, submodels =
## param): kernlab class prediction calculations failed; returning NAs
```

```
## Warning in method$prob(modelFit = modelFit, newdata = newdata, submodels =
## param): kernlab class probability calculations failed; returning NAs
```

```
## Warning in data.frame(..., check.names = FALSE): row names were found from a
## short variable and have been discarded
```

```
## - Fold5: sigma=0.050, C= 8.0 
## + Fold5: sigma=0.010, C=16.0 
## - Fold5: sigma=0.010, C=16.0 
## + Fold5: sigma=0.020, C=16.0 
## - Fold5: sigma=0.020, C=16.0 
## + Fold5: sigma=0.025, C=16.0 
## - Fold5: sigma=0.025, C=16.0 
## + Fold5: sigma=0.030, C=16.0 
## - Fold5: sigma=0.030, C=16.0 
## + Fold5: sigma=0.035, C=16.0 
## - Fold5: sigma=0.035, C=16.0 
## + Fold5: sigma=0.040, C=16.0 
## line search fails -0.8952954 -0.3050781 1.44156e-05 4.301031e-06 -1.601663e-08 -1.97941e-09 -2.394029e-13
```

```
## Warning in method$predict(modelFit = modelFit, newdata = newdata, submodels =
## param): kernlab class prediction calculations failed; returning NAs
```

```
## Warning in method$prob(modelFit = modelFit, newdata = newdata, submodels =
## param): kernlab class probability calculations failed; returning NAs
```

```
## Warning in data.frame(..., check.names = FALSE): row names were found from a
## short variable and have been discarded
```

```
## - Fold5: sigma=0.040, C=16.0 
## + Fold5: sigma=0.050, C=16.0 
## - Fold5: sigma=0.050, C=16.0
```

```
## Warning in nominalTrainWorkflow(x = x, y = y, wts = weights, info = trainInfo,
## : There were missing values in resampled performance measures.
```

```
## Aggregating results
## Selecting tuning parameters
## Fitting sigma = 0.02, C = 0.5 on full training set
```

# Model 5 - Decision Tree


``` r
dt_model <- train(
  Sleep_Quality ~ .,
  data = train_data,
  method = "rpart",
  trControl = ctrl,
  metric = "Accuracy",
  tuneLength = 10
)
```

# Model 6 - Gradient Boosting


``` r
gbm_model <- train(
  Sleep_Quality ~ .,
  data = train_data,
  method = "gbm",
  trControl = ctrl,
  metric = "Accuracy",
  tuneLength = 10,
  verbose = FALSE
)
```

# Model Evaluation


``` r
# Function to evaluate models
evaluate_model <- function(model, test_data, model_name) {
  predictions <- predict(model, test_data)
  cm <- confusionMatrix(predictions, test_data$Sleep_Quality)
  
  results <- data.frame(
    Model = model_name,
    Accuracy = cm$overall["Accuracy"],
    Kappa = cm$overall["Kappa"]
  )
  
  return(list(results = results, cm = cm))
}

# Evaluate all models
models_list <- list(
  RF = rf_model,
  XGBoost = xgb_model,
  Multinom = multinom_model,
  SVM = svm_model,
  DecisionTree = dt_model,
  GBM = gbm_model
)

results_list <- lapply(names(models_list), function(name) {
  evaluate_model(models_list[[name]], test_data, name)
})

# Combine results
all_results <- do.call(rbind, lapply(results_list, function(x) x$results))
print("Model Comparison:")
```

```
## [1] "Model Comparison:"
```

``` r
print(all_results[order(-all_results$Accuracy),])
```

```
##                  Model  Accuracy     Kappa
## Accuracy2     Multinom 0.5258419 0.1759037
## Accuracy4 DecisionTree 0.5255085 0.1681161
## Accuracy5          GBM 0.5245082 0.2077390
## Accuracy1      XGBoost 0.5225075 0.2144040
## Accuracy            RF 0.5195065 0.2050571
## Accuracy3          SVM 0.4924975 0.0000000
```

# Ensemble Model


``` r
# Create ensemble predictions
ensemble_predictions <- data.frame(
  RF = predict(rf_model, test_data),
  XGB = predict(xgb_model, test_data),
  Multinom = predict(multinom_model, test_data),
  SVM = predict(svm_model, test_data),
  DT = predict(dt_model, test_data),
  GBM = predict(gbm_model, test_data)
)

# Majority voting
ensemble_final <- apply(ensemble_predictions, 1, function(x) {
  names(which.max(table(x)))
})
ensemble_final <- factor(ensemble_final, levels = levels(test_data$Sleep_Quality))

# Evaluate ensemble
ensemble_cm <- confusionMatrix(ensemble_final, test_data$Sleep_Quality)
print("Ensemble Model Results:")
```

```
## [1] "Ensemble Model Results:"
```

``` r
print(ensemble_cm)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction Fair Good Poor
##       Fair  142   40  140
##       Good  118  252  155
##       Poor  742  228 1182
## 
## Overall Statistics
##                                           
##                Accuracy : 0.5255          
##                  95% CI : (0.5075, 0.5435)
##     No Information Rate : 0.4925          
##     P-Value [Acc > NIR] : 0.0001604       
##                                           
##                   Kappa : 0.1824          
##                                           
##  Mcnemar's Test P-Value : < 2.2e-16       
## 
## Statistics by Class:
## 
##                      Class: Fair Class: Good Class: Poor
## Sensitivity              0.14172     0.48462      0.8003
## Specificity              0.90986     0.88987      0.3627
## Pos Pred Value           0.44099     0.48000      0.5493
## Neg Pred Value           0.67874     0.89167      0.6517
## Prevalence               0.33411     0.17339      0.4925
## Detection Rate           0.04735     0.08403      0.3941
## Detection Prevalence     0.10737     0.17506      0.7176
## Balanced Accuracy        0.52579     0.68725      0.5815
```

# Feature Importance Analysis


``` r
# Random Forest importance
rf_importance <- varImp(rf_model)
# XGBoost importance
xgb_importance <- varImp(xgb_model)

# Plot feature importance
plot(rf_importance, top = 15, main = "Random Forest Feature Importance")
```

![](FinalClass_files/figure-html/unnamed-chunk-12-1.png)<!-- -->

``` r
plot(xgb_importance, top = 15, main = "XGBoost Feature Importance")
```

![](FinalClass_files/figure-html/unnamed-chunk-12-2.png)<!-- -->

# Save Best Models


``` r
# Print detailed confusion matrix for best model
best_model_name <- all_results$Model[which.max(all_results$Accuracy)]
cat("\nDetailed Results for Best Model (", best_model_name, "):\n")
```

```
## 
## Detailed Results for Best Model ( Multinom ):
```

``` r
saveRDS(models_list[[best_model_name]], paste0("best_model_", best_model_name, ".rds"))
```

# Model Performance Visualization


``` r
# Create performance plot
performance_plot <- ggplot(all_results, aes(x = reorder(Model, -Accuracy), y = Accuracy)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  geom_text(aes(label = round(Accuracy, 3)), vjust = -0.3) +
  theme_minimal() +
  labs(title = "Model Performance Comparison",
       x = "Model",
       y = "Accuracy") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

print(performance_plot)
```

![](FinalClass_files/figure-html/unnamed-chunk-14-1.png)<!-- -->


``` r
plot_confusion_matrix <- function(cm) {
  # Convert confusion matrix to data frame
  cm_d <- as.data.frame(cm$table)
  
  # Calculate accuracy
  accuracy <- round(sum(diag(cm$table)) / sum(cm$table) * 100, 2)
  
  # Create confusion matrix plot
  ggplot(data = cm_d, aes(x = Reference, y = Prediction, fill = Freq)) +
    geom_tile() +
    geom_text(aes(label = sprintf("%d", Freq)), color = "white", size = 4) +
    scale_fill_gradient(low = "#4A90E2", high = "#2E5894") +
    theme_minimal() +
    labs(title = paste("Confusion Matrix\nAccuracy:", accuracy, "%"),
         x = "Actual",
         y = "Predicted") +
    theme(
      plot.title = element_text(hjust = 0.5, size = 14, face = "bold"),
      axis.text = element_text(size = 10),
      axis.title = element_text(size = 12),
      legend.title = element_text(size = 10),
      legend.text = element_text(size = 8)
    )
}

# Get predictions from the best model
predictions <- predict(models_list[[best_model_name]], test_data)

# Create confusion matrix
cm <- confusionMatrix(predictions, test_data$Sleep_Quality)

# Plot the confusion matrix
plot_confusion_matrix(cm)
```

![](FinalClass_files/figure-html/unnamed-chunk-15-1.png)<!-- -->

``` r
# Print detailed metrics
cat("\nDetailed Performance Metrics:\n")
```

```
## 
## Detailed Performance Metrics:
```

``` r
cat("Accuracy:", round(cm$overall["Accuracy"], 4), "\n")
```

```
## Accuracy: 0.5258
```

``` r
cat("Kappa:", round(cm$overall["Kappa"], 4), "\n")
```

```
## Kappa: 0.1759
```

``` r
cat("\nClass-wise Performance:\n")
```

```
## 
## Class-wise Performance:
```

``` r
print(cm$byClass)
```

```
##             Sensitivity Specificity Pos Pred Value Neg Pred Value Precision
## Class: Fair   0.0508982   0.9298948      0.2670157      0.6613248 0.2670157
## Class: Good   0.5057692   0.8870512      0.4843462      0.8953583 0.4843462
## Class: Poor   0.8551117   0.3416557      0.5576159      0.7084469 0.5576159
##                Recall         F1 Prevalence Detection Rate Detection Prevalence
## Class: Fair 0.0508982 0.08549874  0.3341114     0.01700567            0.0636879
## Class: Good 0.5057692 0.49482596  0.1733911     0.08769590            0.1810604
## Class: Poor 0.8551117 0.67504009  0.4924975     0.42114038            0.7552518
##             Balanced Accuracy
## Class: Fair         0.4903965
## Class: Good         0.6964102
## Class: Poor         0.5983837
```
