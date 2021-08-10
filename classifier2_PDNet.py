
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
from model import PDnet1, PDnet2, PDnet3
import math
from tensorflow.keras.models import load_model, Model
import h5py
import tensorflow.keras.backend as K

#Function to diagnose DDH from the scans and patient details
def Classify2_PDNet():    #again specifiy the function to be 2 and hence refere to the alpha angle

  # Training Parameters
  epochs = 40 #going through the dataset 50 times
  batch_size = 32 #Number of samples passed through CNN at one time for training data
  test_batch_size = 16 #batch size for testing data
  val_batch_size = 16 #batch size for validation data

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

  """
  #Load Marta's model
  MartaModel = load_model('C:\Year_4_Courses\Masters_Project\Deep_learning_DDH\deep-learning-hip-dysplasia\\results\\tf2\\classifier__2020-06-04__08-55.h5')

  #Cut the last output layer to get only the features from the second last dense layer
  modelCut = Model(inputs=MartaModel.input, outputs=MartaModel.layers[-2].output)

  #Inputting the training, validation and test datasets into the pre-trained model to get the features that will then be input into the neural network that is to be trained
  outcomePred_Train = modelCut.predict(x_train)
  outcomePred_Val = modelCut.predict(x_val)
  outcomePred_Test = modelCut.predict(x_test)
  """

  #shuffling the dataset batches to help training converge faster, reduce bias, and preventing model from learning the order of the training

  train_dataset = tf.data.Dataset.from_tensor_slices(({"scan" : x_train, "gender" :gender_train, \
   "side" : side_train, "indication" : indication_train, "birthweight" : birthweight_train}, y_train)).batch(batch_size).shuffle(1000)
  train_dataset = train_dataset.repeat()

  val_dataset = tf.data.Dataset.from_tensor_slices(({"scan" : x_val, "gender" :gender_val, \
   "side" : side_val, "indication" : indication_val, "birthweight" : birthweight_val}, y_val)).batch(val_batch_size).shuffle(1000)
  val_dataset = val_dataset.repeat()


  test_dataset = tf.data.Dataset.from_tensor_slices(({"scan" : x_test, "gender" :gender_test, \
   "side" : side_test, "indication" : indication_test, "birthweight" : birthweight_test}, y_test)).batch(test_batch_size).shuffle(1000)
  test_dataset = test_dataset.repeat()



  """
  def lr_schedule(epoch):   #this is currently not being used


      #Learning Rate Schedule. Learning rate is scheduled to be reduced
      #after 80, 120, 160, 180 epochs. Called automatically every epoch as
      #part of callbacks during training.

      lr = 1e-2
      if epoch < 10:
          lr *= 1e-2
      elif epoch < 20:
          lr *= 1e-3
      elif epoch < 30:
          lr *= 1e-4
      elif epoch >= 40:
          lr *= 0.5e-4
      print('Learning rate: ', lr)
      return lr
      """

  # Network Parameters
  WIDTH = data.WIDTH
  HEIGHT = data.HEIGHT
  CHANNELS = 1
#  NUM_FEATURES = 1000 #This is the number of nodes in the last layer of Marta's network
  NUM_OUTPUTS = 1
  NUM_SIDE = data.SIDE
  NUM_GENDER = data.GENDER
  NUM_INDICATION = data.INDICATION


  #This section of code below chooses the model and compiles it with the necessary hyperparameters
  model = PDnet2(HEIGHT, WIDTH, CHANNELS, NUM_OUTPUTS, NUM_SIDE, NUM_GENDER, NUM_INDICATION);  #model chosen is patientDetModel


  #Note: compile configures the model for training BUT DOESN'T TRAIN IT
  #Note: recall is sensitivity while precision is positive predictive value
  model.compile(optimizer=Adam(learning_rate=0.000999), loss='binary_crossentropy', metrics=['binary_accuracy', tf.keras.metrics.Recall(), specificity, tf.keras.metrics.AUC(), negative_predictive_value, positive_predictive_value, matthews_correlation_coefficient]) #very important line about model characteristics
  model.summary() #prints information about the model that was trained

  # Prepare callbacks
  #lr_scheduler = LearningRateScheduler(lr_schedule)
  #lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=1, patience=5, min_lr=0.5e-6)
  #callbacks = [lr_reducer, lr_scheduler]

  start = time.time();
  import datetime
  file_time = datetime.datetime.today().strftime('_%Y-%m-%d__%I-%M')  #get date and time for today for filename
  #This is a way to save the model weights after every epoch in case training is interrupted
  checkpoint = ModelCheckpoint("results/PDNet2/saveweights/best_model" + file_time + ".hdf5", monitor='val_binary_accuracy', verbose=1,
    save_best_only=True, save_weights_only = True, mode='max', save_freq='epoch')
  #FIT TRAINS THE MODEL FOR A FIXED NUMBER OF EPOCHS (ITERATIONS ON A DATASET)
  #print(math.ceil((num_val_examples/val_batch_size)*2))
  history = model.fit(train_dataset,
          epochs=epochs,
          steps_per_epoch= int((num_train_examples/batch_size)*2),
          validation_data=val_dataset,
          validation_steps = math.ceil((num_val_examples/val_batch_size)*2),
          callbacks = [checkpoint])


  #This tests the model that was trained in the previous step - https://www.machinecurve.com/index.php/2020/11/03/how-to-evaluate-a-keras-model-with-model-evaluate/
  #Note: verbose shows the progress bar if 1 and doesn't show anything if it is 0
  #steps tells you the total number of batches fed forward before evaluating is considered to be complete
  evaluation = model.evaluate(test_dataset, verbose=1, steps = math.ceil((num_test_examples/test_batch_size)*2))    #after training and validation end (50 epochs), we finish with testing
  end = time.time()

  #Below are just details printed onto the screen for the user to inform them about the model's accuracy, etc.
  print('Classify Summary: Test Accuracy: %.2f Time Elapsed: %.2f seconds' % (evaluation[1], (end - start)) )
  print('Classify Summary: Test Loss: %.2f Time Elapsed: %.2f seconds' % (evaluation[0], (end - start)) )
  print('Classify Summary: Test Sensitivity: %.2f Time Elapsed: %.2f seconds' % (evaluation[2], (end - start)) )   #note sensitivity=recall
  print('Classify Summary: Test Specificity: %.2f Time Elapsed: %.2f seconds' % (evaluation[3], (end - start)) )
  print('Classify Summary: Test AUC: %.2f Time Elapsed: %.2f seconds' % (evaluation[4], (end - start)) )
  print('Classify Summary: Test NPV: %.2f Time Elapsed: %.2f seconds' % (evaluation[5], (end - start)) )   #note sensitivity=recall
  print('Classify Summary: Test PPV: %.2f Time Elapsed: %.2f seconds' % (evaluation[6], (end - start)) )   #note sensitivity=recall
  print('Classify Summary: Test MCC: %.2f Time Elapsed: %.2f seconds' % (evaluation[7], (end - start)) )   #note sensitivity=recall

  model.save('results/PDNet2/classifier_' + file_time + '.h5')     #save model weights in h5 file

  plt.close()

  #Creating subplots so that we only have 1 big figure instead of 8 figures
  fig = plt.figure(1)

  ax1 = fig.add_subplot(2, 2, 1)
  ax1.plot(history.history["binary_accuracy"])
  ax1.plot(history.history["val_binary_accuracy"])
  ax1.set_ylabel("Accuracy")
  ax1.set_xlabel("Epochs")
  ax1.title.set_text('Test Accuracy: %.2f' % (evaluation[1]))
  ax1.legend(["Train", "Val"], loc="upper left")

  ax2 = fig.add_subplot(2, 2, 2)
  ax2.plot(history.history["loss"])
  ax2.plot(history.history["val_loss"])
  ax2.set_ylabel("Loss")
  ax2.set_xlabel("Epochs")
  ax2.title.set_text("Test Loss: %.2f" % (evaluation[0]))
  ax2.legend(["Train", "Val"], loc="upper left")

  ax3 = fig.add_subplot(2, 2, 3)
  ax3.plot(history.history["recall"])
  ax3.plot(history.history["val_recall"])
  ax3.set_ylabel("Sensitivity")
  ax3.set_xlabel("Epochs")
  ax3.title.set_text("Test Sensitivity: %.2f" % (evaluation[2]))
  ax3.legend(["Train", "Val"], loc="upper left")

  ax4 = fig.add_subplot(2, 2, 4)
  ax4.plot(history.history["specificity"])
  ax4.plot(history.history["val_specificity"])
  ax4.set_ylabel("Specificity")
  ax4.set_xlabel("Epochs")
  ax4.title.set_text("Test Specificity: %.2f" % (evaluation[3]))
  ax4.legend(["Train", "Val"], loc="upper left")

  fig.suptitle("Classification Time Elapsed: %.2f seconds" % ((end - start)))
  plt.subplots_adjust(top=0.85)
  plt.subplots_adjust(wspace=0.5, hspace=0.5)
  fig.savefig('results/PDNet2/classifier1_' + file_time + '.png')

  plt.close()

  fig1 = plt.figure(2)
  fig1.tight_layout(h_pad = 2)

  ax5 = fig1.add_subplot(2, 2, 1)
  ax5.plot(history.history["auc"])
  ax5.plot(history.history["val_auc"])
  ax5.set_ylabel("AUC")
  ax5.set_xlabel("Epochs")
  ax5.title.set_text("Test AUC: %.2f" % (evaluation[4]))
  ax5.legend(["Train", "Val"], loc="upper left")

  ax6 = fig1.add_subplot(2, 2, 2)
  ax6.plot(history.history["negative_predictive_value"])
  ax6.plot(history.history["val_negative_predictive_value"])
  ax6.set_ylabel("NPV")
  ax6.set_xlabel("Epochs")
  ax6.title.set_text("Test NPV: %.2f" % (evaluation[5]))
  ax6.legend(["Train", "Val"], loc="upper left")

  ax7 = fig1.add_subplot(2, 2, 3)
  ax7.plot(history.history["positive_predictive_value"])
  ax7.plot(history.history["val_positive_predictive_value"])
  ax7.set_ylabel("PPV")
  ax7.set_xlabel("Epochs")
  ax7.title.set_text("Test PPV: %.2f" % (evaluation[6]))
  ax7.legend(["Train", "Val"], loc="upper left")

  ax8 = fig1.add_subplot(2, 2, 4)
  ax8.plot(history.history["matthews_correlation_coefficient"])
  ax8.plot(history.history["val_matthews_correlation_coefficient"])
  ax8.set_ylabel("MCC")
  ax8.set_xlabel("Epochs")
  ax8.title.set_text("Test MCC: %.2f" % (evaluation[7]))
  ax8.legend(["Train", "Val"], loc="upper left")

  fig1.suptitle("Classification Time Elapsed: %.2f seconds" % ((end - start)))
  plt.subplots_adjust(top=0.85)
  plt.subplots_adjust(wspace=0.5, hspace=0.5)
  fig1.savefig('results/PDNet2/classifier2_' + file_time + '.png')


  plt.close()
  # Plot Accuracy
  plt.plot(history.history["binary_accuracy"])
  plt.plot(history.history["val_binary_accuracy"])
  plt.ylabel("Accuracy")
  plt.xlabel("Epochs")
  plt.title('Classify Summary: Test Accuracy: %.2f Time Elapsed: %.2f seconds' % (evaluation[1], (end - start)))
  plt.legend(["Train Accuracy", "Validation Accuracy"], loc="upper left")
  plt.savefig('results/PDNet2/classifier_' + file_time + '.png')   #save graph
  plt.close()

  print(model.metrics_names)

  # Plot Loss
  plt.plot(history.history["loss"])
  plt.plot(history.history["val_loss"])
  plt.ylabel("Loss")
  plt.xlabel("Epochs")
  plt.title("Classify Summary: Test Loss: %.2f Time Elapsed: %.2f seconds" % (evaluation[0], (end - start)))
  plt.legend(["Train Loss", "Validation Loss"], loc="upper left")
  plt.savefig('results/PDNet2/loss_' + file_time + '.png') #save loss graph
  plt.close()

  # Plot Sensitivity
  plt.plot(history.history["recall"])
  plt.plot(history.history["val_recall"])
  plt.ylabel("Sensitivity")
  plt.xlabel("Epochs")
  plt.title("Classify Summary: Test Sensitivity: %.2f Time Elapsed: %.2f seconds" % (evaluation[2], (end - start)))
  plt.legend(["Train Sensitivity", "Validation Sensitivity"], loc="upper left")
  plt.savefig('results/PDNet2/sensitivity_' + file_time + '.png')  #save sensitivity graph
  plt.close()

  # Plot Specificity
  plt.plot(history.history["specificity"])
  plt.plot(history.history["val_specificity"])
  plt.ylabel("Specificity")
  plt.xlabel("Epochs")
  plt.title("Classify Summary: Test Specificity: %.2f Time Elapsed: %.2f seconds" % (evaluation[3], (end - start)))
  plt.legend(["Train Specificity", "Validation Specificity"], loc="upper left")
  plt.savefig('results/PDNet2/specificity_' + file_time + '.png')  #save specificity graph
  plt.close()

  # Plot AUC
  plt.plot(history.history["auc"])
  plt.plot(history.history["val_auc"])
  plt.ylabel("AUC")
  plt.xlabel("Epochs")
  plt.title("Classify Summary: Test AUC: %.2f Time Elapsed: %.2f seconds" % (evaluation[4], (end - start)))
  plt.legend(["Train AUC", "Validation AUC"], loc="upper left")
  plt.savefig('results/PDNet2/auc_' + file_time + '.png')  #save AUC graph
  plt.close()

  # Plot NPV
  plt.plot(history.history["negative_predictive_value"])
  plt.plot(history.history["val_negative_predictive_value"])
  plt.ylabel("Negative Predictive Value")
  plt.xlabel("Epochs")
  plt.title("Classify Summary: Test NPV: %.2f Time Elapsed: %.2f seconds" % (evaluation[5], (end - start)))
  plt.legend(["Train NPV", "Validation NPV"], loc="upper left")
  plt.savefig('results/PDNet2/npv_' + file_time + '.png')  #save NPV graph
  plt.close()

  # Plot PPV
  plt.plot(history.history["positive_predictive_value"])
  plt.plot(history.history["val_positive_predictive_value"])
  plt.ylabel("Positive Predictive Value")
  plt.xlabel("Epochs")
  plt.title("Classify Summary: Test PPV: %.2f Time Elapsed: %.2f seconds" % (evaluation[6], (end - start)))
  plt.legend(["Train PPV", "Validation PPV"], loc="upper left")
  plt.savefig('results/PDNet2/ppv_' + file_time + '.png')  #save PPV graph
  plt.close()

  # Plot MCC
  plt.plot(history.history["matthews_correlation_coefficient"])
  plt.plot(history.history["val_matthews_correlation_coefficient"])
  plt.ylabel("Matthews Correlation Coefficient")
  plt.xlabel("Epochs")
  plt.title("Classify Summary: Test PPV: %.2f Time Elapsed: %.2f seconds" % (evaluation[7], (end - start)))
  plt.legend(["Train MCC", "Validation MCC"], loc="upper left")
  plt.savefig('results/PDNet2/mcc_' + file_time + '.png')  #save MCC graph
  plt.close()


#Definitions of additional custom metrics that are used to evaluate the model - https://medium.com/analytics-vidhya/custom-metrics-for-keras-tensorflow-ae7036654e05
def specificity(y_true, y_pred):
    tn = K.sum(K.round(K.clip((1 - y_true) * (1 - y_pred), 0, 1)))
    fp = K.sum(K.round(K.clip((1 - y_true) * y_pred, 0, 1)))
    return tn / (tn + fp + K.epsilon())

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
