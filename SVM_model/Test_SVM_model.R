# Load necessary libraries
library(caret)
library(e1071)
library(pROC)
library(ROCR)

set.seed(123) # Ensure reproducibility

# SVM needs the test set to contain only the descriptors used in the model to be able to make predictions
important_12_descriptors = c('vsa_hyd',	'E_ang'	,'vsurf_D5',	'logP.o.w.',	'PEOE_VSA.0.1',	'vsurf_CW1',	'BCUT_PEOE_0',	'SlogP_VSA1',	'VAdjEq',	'GCUT_SLOGP_3',	'VAdjMa',	'SlogP')


# Read the optimized SVM model from an RDS file
model <- readRDS("svm_best_model.rds")

# Load the test dataset
test_set <- read.csv("test_set.csv", sep = ",", row.names = 1, header = TRUE)

# Ensure 'activity' column is a factor for classification
real_activity <- as.factor(test_set$activity)

test_set=test_set[,important_12_descriptors]

# Predict using the SVM model
predictFinal <- predict(model, test_set)

# Generate a confusion matrix
confusion_matrix <- confusionMatrix(
  data = as.factor(predictFinal),
  reference = real_activity,
  positive = '1'
)

# Extract FP, FN, TP, and TN from the confusion matrix
TP <- confusion_matrix$table[2, 2]
FP <- confusion_matrix$table[1, 2]
TN <- confusion_matrix$table[1, 1]
FN <- confusion_matrix$table[2, 1]

# Calculate MCC using the formula
mccValue <- (TP * TN - FP * FN) / sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))

# Compile results into a dataframe
results <- data.frame(
  Accuracy = confusion_matrix$overall['Accuracy'],
  BalAccuracy = (confusion_matrix$byClass['Sensitivity'] + confusion_matrix$byClass['Specificity']) / 2,
  Sensitivity = confusion_matrix$byClass['Sensitivity'],
  Specificity = confusion_matrix$byClass['Specificity'],
  MCC = mccValue
)

# Calculate AUC

roc_obj <- roc(response = real_activity, predictor = as.numeric(predictFinal)-1)

auc_value <- auc(roc_obj)

results$AUC <- auc_value

# Assign meaningful names to the rows and columns of the results dataframe
rownames(results) <- 'External'
colnames(results) <- c('Accuracy', 'BalAccuracy', 'Sensitivity', 'Specificity', 'MCC',"AUC")


prediction_df=cbind( as.data.frame(predictFinal),real_activity)

colnames(prediction_df) = c( 'Predicted activity','Real activity')
write.csv(prediction_df, "test_set_predicted_molecules_SVM2.csv")


# Save the results to a CSV file
write.csv(results, "svm_performance_External.csv", row.names = TRUE)
