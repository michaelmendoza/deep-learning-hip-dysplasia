
'''
Basic Keras Code for a convolutional neural network
'''

#this is the regression network used to autocrop the scan
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import sys

import time
import numpy as np
import tensorflow as tf
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import LearningRateScheduler, ReduceLROnPlateau
from tensorflow.keras.metrics import MeanSquaredError
import matplotlib.pyplot as plt
from data_loader2 import DataGenerator2    #for this classifier we import only DataGenerator2 since we are using the alpha angle
from model import conv0, conv1, conv2, conv3, resnet, resnet2, patientDetModel, resnet2Autocrop
import math
from tensorflow.keras.models import load_model, Model
import h5py
from data_loader2_autocrop import DataGeneratorCrop
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint

#Function to generate and train a neural network to autocrop the scans
def AutoCrop():

  #Training parameters
  epochs =7
  batch_size = 16
  val_batch_size =8
  test_batch_size =8

  data = DataGeneratorCrop(width = 500, height = 500)

  #Extracting the relevant data for train, val and test sets
  x_train = data.x_train
  x_coordinate_train = data.x_coordinate_train
  y_coordinate_train = data.y_coordinate_train

  num_train_examples = np.shape(y_coordinate_train)
  num_train_examples = num_train_examples[0]

  x_val = data.x_val
  x_coordinate_val = data.x_coordinate_val
  y_coordinate_val = data.y_coordinate_val

  num_val_examples = np.shape(y_coordinate_val)
  num_val_examples = num_val_examples[0]

  x_test = data.x_test
  x_coordinate_test = data.x_coordinate_test
  y_coordinate_test = data.y_coordinate_test

  num_test_examples = np.shape(y_coordinate_test)
  num_test_examples = num_test_examples[0]

  #Remember: The x coordinates and y coordinates have been whitened to speed up training. To undo whitening to get the real pixel coordinates, will need to unwhiten using the mean and std
  x_Mean = data.x_coordinate_mean
  x_Std = data.x_coordinate_std
  y_Mean = data.y_coordinate_mean
  y_Std = data.y_coordinate_std

  print("\nTraining DataSet: " + str(x_train.shape) + " " + str(x_coordinate_train.shape) + " " + str(y_coordinate_train.shape))
  print("\nValidation DataSet: " + str(x_val.shape) + " " + str(x_coordinate_val.shape) + " " + str(y_coordinate_val.shape))
  print("\nTest DataSet: " + str(x_test.shape) + " " + str(x_coordinate_test.shape) + " " + str(y_coordinate_test.shape))

  train_dataset = tf.data.Dataset.from_tensor_slices(({"scan" : x_train}, {"x_coordinate" : x_coordinate_train, "y_coordinate" : y_coordinate_train})).batch(batch_size).shuffle(1000)
  train_dataset = train_dataset.repeat()

  val_dataset = tf.data.Dataset.from_tensor_slices(({"scan" : x_val}, {"x_coordinate" : x_coordinate_val, "y_coordinate" : y_coordinate_val})).batch(val_batch_size).shuffle(1000)
  val_dataset = val_dataset.repeat()

  test_dataset = tf.data.Dataset.from_tensor_slices(({"scan" : x_test}, {"x_coordinate" : x_coordinate_test, "y_coordinate" : y_coordinate_test})).batch(test_batch_size).shuffle(1000)
  test_dataset = test_dataset.repeat()

  # Network Parameters
  WIDTH = data.WIDTH
  HEIGHT = data.HEIGHT
  CHANNELS = 1

  model = resnet2Autocrop(HEIGHT, WIDTH, CHANNELS);

  #The loss function will optimize the model - this is the function that will get minimized by the optimizer. A metric is used to judge the performance of your model. This is only for you to look at and has nothing to
  #do with the optimization process
  #A metric is a function used to judge the performance of your model. Metric functions are similar to loss functions, except that the results from evaluating metrics
  #are not used when training the model.
  model.compile(optimizer=Adam(learning_rate=0.0001), loss={"x_coordinate" : "mse", "y_coordinate" : "mse"}, metrics={"x_coordinate" : "mse", "y_coordinate" : "mse"})  #very important line about model characteristics

  model.summary() #prints information about the model that was trained

  start = time.time();
  import datetime
  file_time = datetime.datetime.today().strftime('_%Y-%m-%d__%I-%M')  #get date and time for today for filename

  #This is a way to save the model weights after every epoch in case training is interrupted
  checkpoint = ModelCheckpoint("results/autocrop/saveweights/best_model" + file_time + ".hdf5", monitor='loss', verbose=1,
    save_best_only=True, save_weights_only = True, mode='min', period=1)

  #FIT TRAINS THE MODEL FOR A FIXED NUMBER OF EPOCHS (ITERATIONS ON A DATASET)
  history = model.fit(train_dataset,
          epochs=epochs,
          steps_per_epoch= int((num_train_examples/batch_size)*2),
          validation_data=val_dataset,
          validation_steps = math.ceil((num_val_examples/val_batch_size)*2),
          callbacks = checkpoint)


  #This tests the model that was trained in the previous step - https://www.machinecurve.com/index.php/2020/11/03/how-to-evaluate-a-keras-model-with-model-evaluate/
  #Note: verbose shows the progress bar if 1 and doesn't show anything if it is 0
  #steps tells you the total number of batches fed forward before evaluating is considered to be complete
  evaluation = model.evaluate(test_dataset, verbose=1, steps = math.ceil((num_test_examples/test_batch_size)*2))    #after training and validation end (50 epochs), we finish with testing
  end = time.time()

  #Just testing to see if the model is predicting sensible values
  predictions = model.predict(x_test)

  x_predictions = predictions["x_coordinate"]
  y_predictions = predictions["y_coordinate"]

  print("\n PREDICTIONS")
  print((y_predictions*y_Std) + y_Mean)
  print((x_predictions*x_Std) + x_Mean)
  print((x_coordinate_test*x_Std) + x_Mean)
  print((y_coordinate_test*y_Std) + y_Mean)
  print("\n PREDICTIONS")
  #Below are just details printed onto the screen for the user to inform them about the model's accuracy, etc.
  print('Classify Summary: Test X Coordinate MSE: %.2f Time Elapsed: %.2f seconds' % (evaluation[3], (end - start)) )
  print('Classify Summary: Test Y Coordinate MSE: %.2f Time Elapsed: %.2f seconds' % (evaluation[4], (end - start)) )


  model.save('results/autocrop/classifier_' + file_time + '.h5')     #save model weights in h5 file

  #converting the metrics (x MSE, x val MSE, y MSE, y val MSE) into numpy arrays
  X_Coordinate_Array = np.array(history.history["x_coordinate_mse"])
  Val_X_Coordinate_Array = np.array(history.history["val_x_coordinate_mse"])
  Y_Coordinate_Array = np.array(history.history["y_coordinate_mse"])
  Val_Y_Coordinate_Array = np.array(history.history["val_y_coordinate_mse"])

  #saving the numpy arrays containing the metrics into .h5 files for later reference
  #MSE for the x coordinate prediction
  f_x = h5py.File('results/autocrop/x_mse' + file_time + '.h5', 'w')
  f_x.create_dataset('mse_x', data = X_Coordinate_Array)
  f_x.close()

  #MSE for the validation x coordinate prediction
  f_x_val = h5py.File('results/autocrop/x_Val_MSE' + file_time + '.h5', 'w')
  f_x_val.create_dataset('mse_x_val', data = Val_X_Coordinate_Array)
  f_x_val.close()

  #MSE for the y coordinate prediction
  f_y = h5py.File('results/autocrop/y_MSE' + file_time + '.h5', 'w')
  f_y.create_dataset('mse_y', data = Y_Coordinate_Array)
  f_y.close()

  #MSE for the validation y coordinate prediction
  f_y_val = h5py.File('results/autocrop/y_Val_MSE' + file_time + '.h5', 'w')
  f_y_val.create_dataset('mse_y_val', data = Val_Y_Coordinate_Array)
  f_y_val.close()

  # Plot MSE for the x coordinate
  plt.plot(history.history["x_coordinate_mse"])
  plt.plot(history.history["val_x_coordinate_mse"])
  plt.ylabel("X Coordinate MSE")
  plt.xlabel("Epochs")
  plt.title("Classify Summary: Test Loss: %.2f Time Elapsed: %.2f seconds" % (evaluation[3], (end - start)))
  plt.legend(["Train MSE", "Validation MSE"], loc="upper left")
  plt.savefig('results/autocrop/x_mse_' + file_time + '.png') #save MSE graph
  plt.close()

  # Plot MSE for the y coordinate
  plt.plot(history.history["y_coordinate_mse"])
  plt.plot(history.history["val_y_coordinate_mse"])
  plt.ylabel("Y Coordinate MSE")
  plt.xlabel("Epochs")
  plt.title("Classify Summary: Test Loss: %.2f Time Elapsed: %.2f seconds" % (evaluation[4], (end - start)))
  plt.legend(["Train MSE", "Validation MSE"], loc="upper left")
  plt.savefig('results/autocrop/y_mse_' + file_time + '.png') #save loss graph
  plt.close()
