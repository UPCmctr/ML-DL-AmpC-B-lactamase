# Load necessary libraries
library(caret)
library(randomForest)
library(e1071)
library(pROC)
library(ROCR)

set.seed(123)

# Read the optimized model from an RDS file
model <- readRDS("randomforest-best_model.rds")

# Load the test dataset
test_set <- read.csv("test_set.csv", sep = ",", row.names = 1, header = TRUE)

# Ensure 'activity' column is a factor for classification
real_activity <- as.factor(test_set$activity)

# Predict using the model
predicted_activity <- predict(model, test_set)

# Generate a confusion matrix
confusion_matrix <- confusionMatrix(
  data = as.factor(predicted_activity),
  reference = real_activity,
  positive = '1'
)

# Calculate various performance metrics
balAccuracy <- (confusion_matrix$byClass['Sensitivity'] + confusion_matrix$byClass['Specificity']) / 2

# Matthews Correlation Coefficient (MCC) calculation
TP <- confusion_matrix$table[2, 2]
FP <- confusion_matrix$table[1, 2]
TN <- confusion_matrix$table[1, 1]
FN <- confusion_matrix$table[2, 1]

mccValue <- (TP * TN - FP * FN) / sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))

# Compile results into a dataframe
results <- data.frame(
  Accuracy = confusion_matrix$overall['Accuracy'],
  BalAccuracy = balAccuracy,
  Sensitivity = confusion_matrix$byClass['Sensitivity'],
  Specificity = confusion_matrix$byClass['Specificity'],
  MCC = mccValue
)

# Calculate AUC
roc_obj <- roc(response = real_activity, predictor = as.numeric(predicted_activity))
auc_value <- auc(roc_obj)

results$AUC = auc_value

# Assign meaningful names to the rows and columns of the results dataframe
rownames(results) <- 'External'
colnames(results) <- c('Accuracy', 'BalAccuracy', 'Sensitivity', 'Specificity', 'MCC', 'AUC')

results

# Prepare prediction dataframe for export
prediction_df = cbind(as.data.frame(predicted_activity), test_set$activity)
colnames(prediction_df) = c('Predicted activity', 'Real activity')

write.csv(prediction_df, "test_set_predicted_molecules_RF.csv")

# Save the results to a CSV file
write.csv(results, "RF_performance_External.csv", row.names = TRUE)
