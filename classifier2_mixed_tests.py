
'''
Basic Keras Code for a convolutional neural network
'''

#this is the classifier used if using alpha or calpha as the diagnostic parameter
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import sys

import time
import numpy as np
import tensorflow as tf
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import LearningRateScheduler, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.metrics import MeanSquaredError
import matplotlib.pyplot as plt
from data_loader2_PDNet import DataGenerator2    #for this classifier we import only DataGenerator2 since we are using the alpha angle
from model import PDnet1, PDnet2, PDnet3, ensembleNet
import math
from tensorflow.keras.models import load_model, Model
import h5py
import tensorflow.keras.backend as K
import pandas as pd

#Function to diagnose DDH from the scans and patient details
def Classify2_mixed():    #again specifiy the function to be 2 and hence refere to the alpha angle
  # Import Dataset. For auto-cropped images, use width = 350 and height = 270.
  #For uncropped images, use: width and height = 256 (which is what Marta used, though the actual image size is 500x500.)
  data = DataGenerator2(width=350, height=270)

  #Extracting the relevant data for train, val and test sets and one-hot encoding the categorical data
  x_train = data.x_train
  gender_train = data.gender_train
  side_train = data.side_train
  indication_train = data.indication_train
  birthweight_train = data.birthweight_train
  y_train = data.y_train

  num_train_examples = np.shape(y_train)
  num_train_examples = num_train_examples[0]

  x_val = data.x_val
  gender_val = data.gender_val
  side_val = data.side_val
  indication_val = data.indication_val
  birthweight_val = data.birthweight_val
  y_val = data.y_val

  num_val_examples = np.shape(y_val)
  num_val_examples = num_val_examples[0]

  x_test = data.x_test
  gender_test = data.gender_test
  side_test = data.side_test
  indication_test = data.indication_test
  birthweight_test = data.birthweight_test
  y_test = data.y_test

  num_test_examples = np.shape(y_test)
  num_test_examples = num_test_examples[0]

  print("Training DataSet: " + str(x_train.shape) + " " + str(y_train.shape))
  print("Validation DataSet: " + str(x_val.shape) + " " + str(y_val.shape))
  print("Test DataSet: " + str(x_test.shape) + " " + str(y_test.shape))


  #Load ResNet model (the one developed by Marta) trained with the cropped scans
  resnetScans_model = load_model('C:\Year_4_Courses\\Masters_Project\\DDH_Project_1\\DDH_Project\\results\\classifier_croppedscans\\classifier__2021-05-17__03-09.h5', compile = False)

  #Inputting the training, validation and test datasets into the pre-trained model to get the features that will then be input into the neural network that is to be trained
  resnetScansPred_Train = np.round(resnetScans_model.predict(x_train))
  resnetScansPred_Val = np.round(resnetScans_model.predict(x_val))
  resnetScansPred_Test = np.round(resnetScans_model.predict(x_test))


  #Load PDNet1 model trained with the cropped scans and the patient details that are risk factors for DDH
  PDNet1_model = load_model('C:\Year_4_Courses\\Masters_Project\\DDH_Project_1\\DDH_Project\\results\\tf2\\classifier__2021-03-26__09-06.h5', compile = False)

  #Inputting the training, validation and test datasets into the pre-trained model to get the features that will then be input into the neural network that is to be trained
  PDNet1Pred_Train = np.round(PDNet1_model.predict({"scan" : x_train, "gender" :gender_train, "side" : side_train, "indication" : indication_train, "birthweight" : birthweight_train}))
  PDNet1Pred_Val = np.round(PDNet1_model.predict({"scan" : x_val, "gender" :gender_val, "side" : side_val, "indication" : indication_val, "birthweight" : birthweight_val}))
  PDNet1Pred_Test = np.round(PDNet1_model.predict({"scan" : x_test, "gender" :gender_test, "side" : side_test, "indication" : indication_test, "birthweight" : birthweight_test}))

  #Sum the predictions for the combination of predictions from the base model
  summed_Predictions_Train = np.add(resnetScansPred_Train, PDNet1Pred_Train)

  #Sum the predictions for the combination of predictions from the base model
  summed_Predictions_Val = np.add(resnetScansPred_Val, PDNet1Pred_Val)

  #Sum the predictions for the combination of predictions from the base model
  summed_Predictions_Test = np.add(resnetScansPred_Test, PDNet1Pred_Test)

  #Now, separate the data so that I have indices specific to the number 2s, 0s and 1s.
  twos_indices_Train = np.where(summed_Predictions_Train == 2)
  twos_indices_Train = twos_indices_Train[0]
  ones_indices_Train = np.where(summed_Predictions_Train == 1)
  ones_indices_Train = ones_indices_Train[0]
  zeros_indices_Train = np.where(summed_Predictions_Train == 0)
  zeros_indices_Train = zeros_indices_Train[0]

  print("Two train")
  print(twos_indices_Train.shape)

  print("One train")
  print(ones_indices_Train.shape)

  print("Zero train")
  print(zeros_indices_Train.shape)

  twos_indices_Val = np.where(summed_Predictions_Val == 2)
  twos_indices_Val = twos_indices_Val[0]
  ones_indices_Val = np.where(summed_Predictions_Val == 1)
  ones_indices_Val = ones_indices_Val[0]
  zeros_indices_Val = np.where(summed_Predictions_Val == 0)
  zeros_indices_Val = zeros_indices_Val[0]

  print("Two Validation")
  print(twos_indices_Val.shape)

  print("One Validation")
  print(ones_indices_Val.shape)

  print("Zero Validation")
  print(zeros_indices_Val.shape)

  twos_indices_Test = np.where(summed_Predictions_Test == 2)
  twos_indices_Test = twos_indices_Test[0]
  ones_indices_Test = np.where(summed_Predictions_Test == 1)
  ones_indices_Test = ones_indices_Test[0]
  zeros_indices_Test = np.where(summed_Predictions_Test == 0)
  zeros_indices_Test = zeros_indices_Test[0]

  print("Two Test")
  print(twos_indices_Test.shape)

  print("One Test")
  print(ones_indices_Test.shape)

  print("Zero Test")
  print(zeros_indices_Test.shape)

  #Code for the combination rule for 2s and 0s

  #first make the empty nD array for this.

  #A PAND B style mixed tests - the tests are run in parallel. A child has DDH only if both tests show up as positive
  #The resulting diagnosis should have higher specificity than either test alone and a lower sensitivity than either test alone
  train_Diagnosis_PAND = np.empty(summed_Predictions_Train.shape, dtype = float)
  train_Diagnosis_PAND[twos_indices_Train] = 1 #Both tests are positive, hence child is positive for DDH
  train_Diagnosis_PAND[zeros_indices_Train] = 0 #Both tests are negative, hence child is negative for DDH
  train_Diagnosis_PAND[ones_indices_Train] = 0 #Only one of the tests is positive, hence the child is negative for DDH

  val_Diagnosis_PAND = np.empty(summed_Predictions_Val.shape, dtype = float)
  val_Diagnosis_PAND[twos_indices_Val] = 1
  val_Diagnosis_PAND[zeros_indices_Val] = 0
  val_Diagnosis_PAND[ones_indices_Val] = 0

  test_Diagnosis_PAND = np.empty(summed_Predictions_Test.shape, dtype = float)
  test_Diagnosis_PAND[twos_indices_Test] = 1
  test_Diagnosis_PAND[zeros_indices_Test] = 0
  test_Diagnosis_PAND[ones_indices_Test] = 0

  overall_diagnosis_PAND = np.concatenate([train_Diagnosis_PAND, val_Diagnosis_PAND, test_Diagnosis_PAND])

  #A POR B style mixed tests - the tests are run in parallel. A child has DDH if at least one of the tests show up as positive
  #The resulting diagnosis should have higher sensitivity than either test alone and a lower specificity than either test alone
  train_Diagnosis_POR = np.empty(summed_Predictions_Train.shape, dtype = float)
  train_Diagnosis_POR[twos_indices_Train] = 1
  train_Diagnosis_POR[zeros_indices_Train] = 0
  train_Diagnosis_POR[ones_indices_Train] = 1 #At least one of the tests is positive, hence the child is positive for DDH

  val_Diagnosis_POR = np.empty(summed_Predictions_Val.shape, dtype = float)
  val_Diagnosis_POR[twos_indices_Val] = 1
  val_Diagnosis_POR[zeros_indices_Val] = 0
  val_Diagnosis_POR[ones_indices_Val] = 1

  test_Diagnosis_POR = np.empty(summed_Predictions_Test.shape, dtype = float)
  test_Diagnosis_POR[twos_indices_Test] = 1
  test_Diagnosis_POR[zeros_indices_Test] = 0
  test_Diagnosis_POR[ones_indices_Test] = 1

  overall_diagnosis_POR = np.concatenate([train_Diagnosis_POR, val_Diagnosis_POR, test_Diagnosis_POR])

  y_true = np.concatenate([y_train, y_val, y_test])

  print("A POR B - test data")
  print("Test Diagnostic accuracy: " + str(diagnostic_accuracy(y_test, test_Diagnosis_POR)))
  print("Test Sensitivity: " + str(sensitivity(y_test, test_Diagnosis_POR)))
  print("Test Specificity: " + str(specificity(y_test, test_Diagnosis_POR)))
  print("Test PPV: " + str(positive_predictive_value(y_test, test_Diagnosis_POR)))
  print("Test NPV: " + str(negative_predictive_value(y_test, test_Diagnosis_POR)))
  print("Test MCC: " + str(matthews_correlation_coefficient(y_test, test_Diagnosis_POR)))

  print("A POR B - overall data")
  print("Overall Diagnostic accuracy: " + str(diagnostic_accuracy(y_true, overall_diagnosis_POR)))
  print("Overall Sensitivity: " + str(sensitivity(y_true, overall_diagnosis_POR)))
  print("Overall Specificity: " + str(specificity(y_true, overall_diagnosis_POR)))
  print("Overall PPV: " + str(positive_predictive_value(y_true, overall_diagnosis_POR)))
  print("Overall NPV: " + str(negative_predictive_value(y_true, overall_diagnosis_POR)))
  print("Overall MCC: " + str(matthews_correlation_coefficient(y_true, overall_diagnosis_POR)))

  print("A PAND B - test data")
  print("Test Diagnostic accuracy: " + str(diagnostic_accuracy(y_test, test_Diagnosis_PAND)))
  print("Test Sensitivity: " + str(sensitivity(y_test, test_Diagnosis_PAND)))
  print("Test Specificity: " + str(specificity(y_test, test_Diagnosis_PAND)))
  print("Test PPV: " + str(positive_predictive_value(y_test, test_Diagnosis_PAND)))
  print("Test NPV: " + str(negative_predictive_value(y_test, test_Diagnosis_PAND)))
  print("Test MCC: " + str(matthews_correlation_coefficient(y_test, test_Diagnosis_PAND)))

  print("A PAND B - overall data")
  print("Overall Diagnostic accuracy: " + str(diagnostic_accuracy(y_true, overall_diagnosis_PAND)))
  print("Overall Sensitivity: " + str(sensitivity(y_true, overall_diagnosis_PAND)))
  print("Overall Specificity: " + str(specificity(y_true, overall_diagnosis_PAND)))
  print("Overall PPV: " + str(positive_predictive_value(y_true, overall_diagnosis_PAND)))
  print("Overall NPV: " + str(negative_predictive_value(y_true, overall_diagnosis_PAND)))
  print("Overall MCC: " + str(matthews_correlation_coefficient(y_true, overall_diagnosis_PAND)))
  """
  #Make a confusion matrix for A POR B diagnosis for test data
  data_POR_test = {'y_Actual': y_test,
        'y_Predicted': test_Diagnosis_POR
        }

  df_POR_test = pd.DataFrame(data_POR_test, columns=['y_Actual','y_Predicted'])

  confusion_matrix_POR_test = pd.crosstab(df_POR_test['y_Actual'], df_POR_test['y_Predicted'], rownames=['Actual'], colnames=['Predicted'])
  print("Confusion matrix for A POR B on the test data")
  print(confusion_matrix_POR_test)

  #Make a confusion matrix for A PAND B diagnosis for test data
  data_PAND_test = {'y_Actual': y_test,
        'y_Predicted': test_Diagnosis_PAND
        }

  df_PAND_test = pd.DataFrame(data_PAND_test, columns=['y_Actual','y_Predicted'])

  confusion_matrix_PAND_test = pd.crosstab(df_PAND_test['y_Actual'], df_PAND_test['y_Predicted'], rownames=['Actual'], colnames=['Predicted'])
  print("Confusion matrix for A PAND B on the test data")
  print(confusion_matrix_PAND_test)

  #Make a confusion matrix for A POR B diagnosis for entire data
  data_POR = {'y_Actual': y_true,
        'y_Predicted': overall_diagnosis_POR
        }

  df_POR = pd.DataFrame(data_POR, columns=['y_Actual','y_Predicted'])

  confusion_matrix_POR = pd.crosstab(df_POR['y_Actual'], df_POR['y_Predicted'], rownames=['Actual'], colnames=['Predicted'])
  print("Confusion matrix for A POR B on entire data")
  print(confusion_matrix_POR)

  #Make a confusion matrix for A PAND B diagnosis for entire data
  data_PAND = {'y_Actual': y_test,
        'y_Predicted': overall_diagnosis_PAND
        }

  df_PAND = pd.DataFrame(data_PAND, columns=['y_Actual','y_Predicted'])

  confusion_matrix_PAND = pd.crosstab(df_PAND['y_Actual'], df_PAND['y_Predicted'], rownames=['Actual'], colnames=['Predicted'])
  print("Confusion matrix for A PAND B on entire data")
  print(confusion_matrix_PAND)
  """
#Definitions of additional custom metrics that are used to evaluate the model - https://medium.com/analytics-vidhya/custom-metrics-for-keras-tensorflow-ae7036654e05
def specificity(y_true, y_pred):
    tn = K.sum(K.round(K.clip((1 - y_true) * (1 - y_pred), 0, 1)))
    fp = K.sum(K.round(K.clip((1 - y_true) * y_pred, 0, 1)))
    return tn / (tn + fp + K.epsilon())

def sensitivity(y_true, y_pred):
    tp = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    fn = K.sum(K.round(K.clip(y_true * (1 - y_pred), 0, 1)))
    return tp / (tp + fn + K.epsilon())

def negative_predictive_value(y_true, y_pred):
    tn = K.sum(K.round(K.clip((1 - y_true) * (1 - y_pred), 0, 1)))
    fn = K.sum(K.round(K.clip(y_true * (1 - y_pred), 0, 1)))
    return tn / (tn + fn + K.epsilon())

def positive_predictive_value(y_true, y_pred):
    tp = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    fp = K.sum(K.round(K.clip((1 - y_true) * y_pred, 0, 1)))
    return tp / (tp + fp + K.epsilon())

def matthews_correlation_coefficient(y_true, y_pred):
    tp = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    tn = K.sum(K.round(K.clip((1 - y_true) * (1 - y_pred), 0, 1)))
    fp = K.sum(K.round(K.clip((1 - y_true) * y_pred, 0, 1)))
    fn = K.sum(K.round(K.clip(y_true * (1 - y_pred), 0, 1)))

    num = tp * tn - fp * fn
    den = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    return num / K.sqrt(den + K.epsilon())

def diagnostic_accuracy(y_true, y_pred):
    tp = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    tn = K.sum(K.round(K.clip((1 - y_true) * (1 - y_pred), 0, 1)))
    fp = K.sum(K.round(K.clip((1 - y_true) * y_pred, 0, 1)))
    fn = K.sum(K.round(K.clip(y_true * (1 - y_pred), 0, 1)))

    num = tp + tn
    den = tp + fp + fn + tn
    return num / den
