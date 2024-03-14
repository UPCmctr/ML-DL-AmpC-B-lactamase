# Load necessary libraries
library(caret)
library(randomForest)
library(e1071)
library(pROC)
library(ROCR)

set.seed(123)

# Read the optimized model from an RDS file1
model <- readRDS("randomforest-best_model.rds")

# Load the test dataset
test_set <- read.csv("test_set.csv", sep = ",", row.names = 1, header = TRUE)

# Ensure 'activity' column is a factor for classification
real_activity <- as.factor(test_set$activity)

# Predict using the model
# Note: It seems there was duplicated code for prediction, which is removed in this version
predicted_activity <- predict(model, test_set)

# Generate a confusion matrix
# 'reference' is the actual outcome, 'data' is the predicted outcome
# Adjust the prediction to match expected format if necessary
confusion_matrix <- confusionMatrix(
  data = as.factor(ifelse(as.numeric(predictFinal) == 1, 0, 1)),
  reference = real_activity,
  positive = '1'
)

# Calculate various performance metrics
# Average of Sensitivity and Specificity for Balanced Accuracy
balAccuracy <- (confusion_matrix$byClass['Sensitivity'] + confusion_matrix$byClass['Specificity']) / 2

# Matthews Correlation Coefficient (MCC)
mccValue <- mcc(
  TP = confusion_matrix$table[2, 2],
  FP = confusion_matrix$table[2, 1],
  TN = confusion_matrix$table[1, 1],
  FN = confusion_matrix$table[1, 2]
)
confusion_matrix

# Compile results into a dataframe
results <- data.frame(
  Accuracy = confusion_matrix$overall['Accuracy'],
  BalAccuracy = balAccuracy,
  Sensitivity = confusion_matrix$byClass['Sensitivity'],
  Specificity = confusion_matrix$byClass['Specificity'],
  MCC = mccValue
)

# Calculate AUC
roc_obj <- roc(response = real_activity, predictor = as.numeric(predictFinal)-1)
auc_value <- auc(roc_obj)

results$AUC=auc_value

# Assign meaningful names to the rows and columns of the results dataframe
rownames(results) <- 'External'
colnames(results) <- c('Accuracy', 'BalAccuracy', 'Sensitivity', 'Specificity', 'MCC','AUC')


prediction_df=cbind( as.data.frame(predicted_activity),test_set$activity)

colnames(prediction_df) = c( 'Predicted activity','Real activity')
write.csv(prediction_df, "test_set_predicted_molecules_RF.csv")

#Save the results to a CSV file
write.csv(results, "RF_performance_External.csv", row.names = TRUE)

