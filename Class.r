# Load required libraries
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
library(ggplot)

# Data Loading and Preprocessing
data <- read.csv("mental_health_with_sleep_quality.csv")
data <- data %>% select(-c(User_ID, Sleep_Hours))
categorical_cols <- c("Gender", "Mental_Health_Status", "Work_Environment_Impact", 
                     "Support_Systems_Access", "Online_Support_Usage", "Sleep_Quality")
data[categorical_cols] <- lapply(data[categorical_cols], factor)
data <- data %>%
  mutate(
    Total_Screen_Time = Technology_Usage_Hours + Social_Media_Usage_Hours + Gaming_Hours,
    Screen_Activity_Ratio = Screen_Time_Hours / (Physical_Activity_Hours + 0.1),
    Tech_Social_Ratio = Technology_Usage_Hours / (Social_Media_Usage_Hours + 0.1)
  )

# Data Splitting
set.seed(123)
trainIndex <- createDataPartition(data$Sleep_Quality, p = 0.7, list = FALSE)
train_data <- data[trainIndex, ]
test_data <- data[-trainIndex, ]
ctrl <- trainControl(
  method = "cv",
  number = 5,
  classProbs = TRUE,
  summaryFunction = multiClassSummary
)

# Model 1 - Random Forest
rf_model <- train(
  Sleep_Quality ~ .,
  data = train_data,
  method = "rf",
  trControl = trainControl(
    method = "cv",
    number = 5,
    verboseIter = TRUE
  ),
  tuneGrid = expand.grid(mtry = sqrt(ncol(train_data)-1)),
  ntree = 100
)

# Model 2 - XGBoost
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

# Model 3 - Multinomial Logistic Regression
multinom_model <- train(
  Sleep_Quality ~ .,
  data = train_data,
  method = "multinom",
  trControl = ctrl,
  metric = "Accuracy",
  trace = FALSE
)

# Model 4 - Support Vector Machine
preProcess_range <- preProcess(train_data[,-which(names(train_data) == "Sleep_Quality")], 
                             method = c("center", "scale"))
train_data_preprocessed <- predict(preProcess_range, train_data)
test_data_preprocessed <- predict(preProcess_range, test_data)
ctrl_svm <- trainControl(
  method = "cv",
  number = 5,
  classProbs = TRUE,
  verboseIter = TRUE,
  allowParallel = TRUE
)
svm_grid <- expand.grid(
  sigma = c(0.01, 0.02, 0.025, 0.03, 0.035, 0.04, 0.05),
  C = c(0.5, 1, 2, 4, 8, 16)
)
set.seed(123)
svm_model <- train(
  Sleep_Quality ~ .,
  data = train_data_preprocessed,
  method = "svmRadial",
  trControl = ctrl_svm,
  tuneGrid = svm_grid,
  preProcess = NULL,
  verbose = FALSE
)

# Model 5 - Decision Tree
dt_model <- train(
  Sleep_Quality ~ .,
  data = train_data,
  method = "rpart",
  trControl = ctrl,
  metric = "Accuracy",
  tuneLength = 10
)

# Model 6 - Gradient Boosting
gbm_model <- train(
  Sleep_Quality ~ .,
  data = train_data,
  method = "gbm",
  trControl = ctrl,
  metric = "Accuracy",
  tuneLength = 10,
  verbose = FALSE
)

# Model Evaluation
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
all_results <- do.call(rbind, lapply(results_list, function(x) x$results))
print("Model Comparison:")
print(all_results[order(-all_results$Accuracy),])

# Ensemble Model
ensemble_predictions <- data.frame(
  RF = predict(rf_model, test_data),
  XGB = predict(xgb_model, test_data),
  Multinom = predict(multinom_model, test_data),
  SVM = predict(svm_model, test_data),
  DT = predict(dt_model, test_data),
  GBM = predict(gbm_model, test_data)
)
ensemble_final <- apply(ensemble_predictions, 1, function(x) {
  names(which.max(table(x)))
})
ensemble_final <- factor(ensemble_final, levels = levels(test_data$Sleep_Quality))
ensemble_cm <- confusionMatrix(ensemble_final, test_data$Sleep_Quality)
print("Ensemble Model Results:")
print(ensemble_cm)

# Feature Importance Analysis
rf_importance <- varImp(rf_model)
xgb_importance <- varImp(xgb_model)
plot(rf_importance, top = 15, main = "Random Forest Feature Importance")
plot(xgb_importance, top = 15, main = "XGBoost Feature Importance")

# Save Best Models
best_model_name <- all_results$Model[which.max(all_results$Accuracy)]
cat("\nDetailed Results for Best Model (", best_model_name, "):\n")
saveRDS(models_list[[best_model_name]], paste0("best_model_", best_model_name, ".rds"))

# Model Performance Visualization
performance_plot <- ggplot(all_results, aes(x = reorder(Model, -Accuracy), y = Accuracy)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  geom_text(aes(label = round(Accuracy, 3)), vjust = -0.3) +
  theme_minimal() +
  labs(title = "Model Performance Comparison",
       x = "Model",
       y = "Accuracy") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))
print(performance_plot)

plot_confusion_matrix <- function(cm) {
  cm_d <- as.data.frame(cm$table)
  accuracy <- round(sum(diag(cm$table)) / sum(cm$table) * 100, 2)
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
predictions <- predict(models_list[[best_model_name]], test_data)
cm <- confusionMatrix(predictions, test_data$Sleep_Quality)
plot_confusion_matrix(cm)
cat("\nDetailed Performance Metrics:\n")
cat("Accuracy:", round(cm$overall["Accuracy"], 4), "\n")
cat("Kappa:", round(cm$overall["Kappa"], 4), "\n")
cat("\nClass-wise Performance:\n")
print(cm$byClass)
