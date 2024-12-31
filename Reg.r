# Load required libraries
library(tidyverse)
library(caret)
library(randomForest)
library(xgboost)
library(gbm)
library(kernlab)
library(e1071)

# Data Loading and Preprocessing
data <- read.csv("mental_health_and_technology_usage_2024.csv")
data <- data %>% select(-c(User_ID))
categorical_cols <- c("Gender", "Mental_Health_Status", "Work_Environment_Impact", 
                     "Support_Systems_Access", "Online_Support_Usage")
data[categorical_cols] <- lapply(data[categorical_cols], factor)
data <- data %>%
  mutate(
    Total_Screen_Time = Technology_Usage_Hours + Social_Media_Usage_Hours + Gaming_Hours,
    Screen_Activity_Ratio = Screen_Time_Hours / (Physical_Activity_Hours + 0.1),
    Tech_Social_Ratio = Technology_Usage_Hours / (Social_Media_Usage_Hours + 0.1)
  )

# Data Splitting
set.seed(123)
trainIndex <- createDataPartition(data$Sleep_Hours, p = 0.7, list = FALSE)
train_data <- data[trainIndex, ]
test_data <- data[-trainIndex, ]
ctrl <- trainControl(
  method = "cv",
  number = 5,
  summaryFunction = defaultSummary
)

# Model 1 - Linear Regression
lm_model <- train(
  Sleep_Hours ~ .,
  data = train_data,
  method = "lm",
  trControl = ctrl
)

# Model 2 - Decision Tree
dt_model <- train(
  Sleep_Hours ~ .,
  data = train_data,
  method = "rpart",
  trControl = ctrl,
  tuneLength = 10
)

# Model 3 - Random Forest
rf_model <- train(
  Sleep_Hours ~ .,
  data = train_data,
  method = "rf",
  trControl = ctrl,
  tuneLength = 10
)

# Model 4 - XGBoost
xgb_model <- train(
  Sleep_Hours ~ .,
  data = train_data,
  method = "xgbTree",
  trControl = ctrl,
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

# Model 5 - Gradient Boosting
gbm_model <- train(
  Sleep_Hours ~ .,
  data = train_data,
  method = "gbm",
  trControl = ctrl,
  tuneLength = 10,
  verbose = FALSE
)

# Model 6 - Support Vector Machine
svm_model <- train(
  Sleep_Hours ~ .,
  data = train_data,
  method = "svmRadial",
  trControl = ctrl,
  tuneLength = 10
)

# Model Evaluation
evaluate_model <- function(model, test_data, model_name) {
  predictions <- predict(model, test_data)
  rmse <- RMSE(predictions, test_data$Sleep_Hours)
  results <- data.frame(
    Model = model_name,
    RMSE = rmse
  )
  return(results)
}
models_list <- list(
  LM = lm_model,
  DecisionTree = dt_model,
  RandomForest = rf_model,
  XGBoost = xgb_model,
  GBM = gbm_model,
  SVM = svm_model
)
results_list <- lapply(names(models_list), function(name) {
  evaluate_model(models_list[[name]], test_data, name)
})
all_results <- do.call(rbind, results_list)
print("Model Comparison:")
print(all_results[order(all_results$RMSE),])

# Save Best Models
best_model_name <- all_results$Model[which.min(all_results$RMSE)]
cat("\nDetailed Results for Best Model (", best_model_name, "):\n")
saveRDS(models_list[[best_model_name]], paste0("best_model_", best_model_name, ".rds"))

# Model Performance Visualization
performance_plot <- ggplot(all_results, aes(x = reorder(Model, RMSE), y = RMSE)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  geom_text(aes(label = round(RMSE, 3)), vjust = -0.3) +
  theme_minimal() +
  labs(title = "Model Performance Comparison",
       x = "Model",
       y = "RMSE") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))
print(performance_plot)
