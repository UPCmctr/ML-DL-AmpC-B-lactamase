import pandas as pd
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, roc_curve, auc

# Load the test data
test = pd.read_csv('test_set.csv').set_index("names")

# Load means and standard deviations for preprocessing
with open("mean.pkl", "rb") as f:
    mean = pickle.load(f)
with open("std.pkl", "rb") as f:
    std = pickle.load(f)

# Preprocess the test data
xtest = test.drop(['activity'], axis=1)
xtest -= mean
xtest /= std
ytest = test['activity']

# Load the pre-trained model
model = load_model("FFNN_model.hdf5")

# Make predictions
predict_results = model.predict(xtest) > 0.5

# Compute the confusion matrix
cm = confusion_matrix(ytest, predict_results)


# Function to calculate performance metrics
def perf(cm):
    TP = cm[0][0]
    TN = cm[1][1]
    FP = cm[1][0]
    FN = cm[0][1]
    specificity = TP / (TP + FN)
    sensitivity = TN / (TN + FP)
    mcc = ((TN * TP) - (FN * FP)) / np.sqrt((TN + FN) * (TN + FP) * (TP + FN) * (TP + FP))
    fpr, tpr, _ = roc_curve(ytest, model.predict(xtest).ravel())
    auc_score = auc(fpr, tpr)
    return sensitivity, specificity, mcc, auc_score


sensitivity, specificity, mcc, AUC = perf(cm)

print(f"sensitivity = {sensitivity}, specificity = {specificity}, mcc = {mcc}, AUC = {AUC}")

# Create a DataFrame with the metrics
metrics_df = pd.DataFrame({
    "Metric": ["Sensitivity", "Specificity", "MCC", "AUC"],
    "Value": [sensitivity, specificity, mcc, AUC]
})

# Save the DataFrame to a CSV file
metrics_df.to_csv("FFNN_performance_External.csv", index=False)

# Convert predictions to binary format
binary_predictions = predict_results.astype(int)


predictions_df = pd.DataFrame({
    "Molecule Name": xtest.index,
    "Predicted Class": binary_predictions.flatten(),
    "True Class": ytest
})

# Save CSV file  
predictions_df.to_csv("test_set_predicted_molecules_FFNN.csv", index=False)




