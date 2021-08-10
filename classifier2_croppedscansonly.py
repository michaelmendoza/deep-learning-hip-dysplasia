'''
Basic Keras Code for a convolutional neural network
'''

#this is the classifier used if using alpha or calpha as the diagnostic parameter
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import time
import numpy as np
import tensorflow as tf
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import LearningRateScheduler, ReduceLROnPlateau, ModelCheckpoint
import matplotlib.pyplot as plt
import math
from data_loader2_croppedscansonly import DataGenerator2   #for this classifier we import only DataGenerator2 since we are using the alpha angle
from model import conv0, conv1, conv2, conv3, resnet, resnet2
import tensorflow.keras.backend as K

def Classify2_croppedScans():    #again specifiy the function to be 2 and hence refere to the alpha angle

  # Training Parameters
  epochs = 40
  batch_size = 32
  test_batch_size = 16
  val_batch_size = 16

  #Note: to train Marta's network with cropped scans, use width = 350 and height = 270
  data = DataGenerator2(width=350, height=270)  #in this case we have specified the width and height to be 256, larger than the standard in the dataloader file
  x_train = data.x_train
  y_train = data.y_train
  x_val = data.x_val
  y_val = data.y_val
  x_test = data.x_test
  y_test = data.y_test

  num_train_examples = np.shape(y_train)
  num_train_examples = num_train_examples[0]

  num_val_examples = np.shape(y_val)
  num_val_examples = num_val_examples[0]

  num_test_examples = np.shape(y_test)
  num_test_examples = num_test_examples[0]

  print("Training DataSet: " + str(x_train.shape) + " " + str(y_train.shape))
  print("Validation DataSet: " + str(x_val.shape) + " " + str(y_val.shape))
  print("Test DataSet: " + str(x_test.shape) + " " + str(y_test.shape))

  train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size).shuffle(1000)
  train_dataset = train_dataset.repeat()

  val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(val_batch_size).shuffle(1000)
  val_dataset = val_dataset.repeat()

  test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(test_batch_size).shuffle(1000)
  test_dataset = test_dataset.repeat()

  def lr_schedule(epoch):   #this is currently not being used
      """Learning Rate Schedule. Learning rate is scheduled to be reduced
      after 80, 120, 160, 180 epochs. Called automatically every epoch as
      part of callbacks during training. """

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

  # Network Parameters
  WIDTH = data.WIDTH
  HEIGHT = data.HEIGHT
  CHANNELS = 1
  NUM_OUTPUTS = 1

  model = resnet2(HEIGHT, WIDTH, CHANNELS, NUM_OUTPUTS);  #model chosen is resnet2
  model.compile(optimizer=Adam(learning_rate=0.0034), loss='binary_crossentropy', metrics=['binary_accuracy', tf.keras.metrics.Recall(), specificity, tf.keras.metrics.AUC(), negative_predictive_value, positive_predictive_value, matthews_correlation_coefficient]) #very important line about model characteristics
  model.summary()


  #Change the folders if you are working with uncropped scans to classifier_uncroppedscans

  start = time.time();
  import datetime
  file_time = datetime.datetime.today().strftime('_%Y-%m-%d__%I-%M')  #get date and time for today for filename
  #This is a way to save the model weights after every epoch in case training is interrupted
  checkpoint = ModelCheckpoint("results/classifier_croppedscans/saveweights/best_model" + file_time + ".hdf5", monitor='val_binary_accuracy', verbose=1,
      save_best_only=True, save_weights_only = True, mode='max', save_freq='epoch')
  history = model.fit(train_dataset,
      epochs=epochs,
      steps_per_epoch=int((num_train_examples/batch_size)*2),    #108 since training dataset is 864 aprox, 864/batchsize = 864/16 = 54, and this is usually x2 hence 108
      validation_data=val_dataset,
      validation_steps = math.ceil((num_val_examples/val_batch_size)*2),
      callbacks = [checkpoint])    #27 since validation dataset is 108 aprox, 108/batchsize = 108/8 = 13.5, and this is usually x2 hence 27

  evaluation = model.evaluate(x_test, y_test, verbose=1, steps = math.ceil((num_test_examples/test_batch_size)*2))    #after training and validation end (50 epochs), we finish with testing
  end = time.time()

  print('Classify Summary: Test Accuracy: %.2f Time Elapsed: %.2f seconds' % (evaluation[1], (end - start)) )
  print('Classify Summary: Test Loss: %.2f Time Elapsed: %.2f seconds' % (evaluation[0], (end - start)) )
  print('Classify Summary: Sensitivity: %.2f Time Elapsed: %.2f seconds' % (evaluation[2], (end - start)) )   #note sensitivity=recall
  print('Classify Summary: Test Specificity: %.2f Time Elapsed: %.2f seconds' % (evaluation[3], (end - start)) )
  print('Classify Summary: Test AUC: %.2f Time Elapsed: %.2f seconds' % (evaluation[4], (end - start)) )
  print('Classify Summary: Test NPV: %.2f Time Elapsed: %.2f seconds' % (evaluation[5], (end - start)) )   #note sensitivity=recall
  print('Classify Summary: Test PPV: %.2f Time Elapsed: %.2f seconds' % (evaluation[6], (end - start)) )   #note sensitivity=recall
  print('Classify Summary: Test MCC: %.2f Time Elapsed: %.2f seconds' % (evaluation[7], (end - start)) )   #note sensitivity=recall

  model.save('results/classifier_croppedscans/classifier_' + file_time + '.h5')     #save model weights in h5 file

  plt.close()

  #Creating subplots so that we only have 2 big figures instead of 8 figures
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
  fig.savefig('results/classifier_croppedscans/classifier1_' + file_time + '.png')

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
  fig1.savefig('results/classifier_croppedscans/classifier2_' + file_time + '.png')


  plt.close()
  # Plot Accuracy
  plt.plot(history.history["binary_accuracy"])
  plt.plot(history.history["val_binary_accuracy"])
  plt.ylabel("Accuracy")
  plt.xlabel("Epochs")
  plt.title('Classify Summary: Test Accuracy: %.2f Time Elapsed: %.2f seconds' % (evaluation[1], (end - start)))
  plt.legend(["Train Accuracy", "Validation Accuracy"], loc="upper left")
  plt.savefig('results/classifier_croppedscans/classifier_' + file_time + '.png')   #save graph
  plt.close()

  print(model.metrics_names)

  # Plot Loss
  plt.plot(history.history["loss"])
  plt.plot(history.history["val_loss"])
  plt.ylabel("Loss")
  plt.xlabel("Epochs")
  plt.title("Classify Summary: Test Loss: %.2f Time Elapsed: %.2f seconds" % (evaluation[0], (end - start)))
  plt.legend(["Train Loss", "Validation Loss"], loc="upper left")
  plt.savefig('results/classifier_croppedscans/loss_' + file_time + '.png') #save loss graph
  plt.close()

  # Plot Sensitivity
  plt.plot(history.history["recall"])
  plt.plot(history.history["val_recall"])
  plt.ylabel("Sensitivity")
  plt.xlabel("Epochs")
  plt.title("Classify Summary: Test Sensitivity: %.2f Time Elapsed: %.2f seconds" % (evaluation[2], (end - start)))
  plt.legend(["Train Sensitivity", "Validation Sensitivity"], loc="upper left")
  plt.savefig('results/classifier_croppedscans/sensitivity_' + file_time + '.png')  #save sensitivity graph
  plt.close()

  # Plot Specificity
  plt.plot(history.history["specificity"])
  plt.plot(history.history["val_specificity"])
  plt.ylabel("Specificity")
  plt.xlabel("Epochs")
  plt.title("Classify Summary: Test Specificity: %.2f Time Elapsed: %.2f seconds" % (evaluation[3], (end - start)))
  plt.legend(["Train Specificity", "Validation Specificity"], loc="upper left")
  plt.savefig('results/classifier_croppedscans/specificity_' + file_time + '.png')  #save specificity graph
  plt.close()

  # Plot AUC
  plt.plot(history.history["auc"])
  plt.plot(history.history["val_auc"])
  plt.ylabel("AUC")
  plt.xlabel("Epochs")
  plt.title("Classify Summary: Test AUC: %.2f Time Elapsed: %.2f seconds" % (evaluation[4], (end - start)))
  plt.legend(["Train AUC", "Validation AUC"], loc="upper left")
  plt.savefig('results/classifier_croppedscans/auc_' + file_time + '.png')  #save AUC graph
  plt.close()

  # Plot NPV
  plt.plot(history.history["negative_predictive_value"])
  plt.plot(history.history["val_negative_predictive_value"])
  plt.ylabel("Negative Predictive Value")
  plt.xlabel("Epochs")
  plt.title("Classify Summary: Test NPV: %.2f Time Elapsed: %.2f seconds" % (evaluation[5], (end - start)))
  plt.legend(["Train NPV", "Validation NPV"], loc="upper left")
  plt.savefig('results/classifier_croppedscans/npv_' + file_time + '.png')  #save NPV graph
  plt.close()

  # Plot PPV
  plt.plot(history.history["positive_predictive_value"])
  plt.plot(history.history["val_positive_predictive_value"])
  plt.ylabel("Positive Predictive Value")
  plt.xlabel("Epochs")
  plt.title("Classify Summary: Test PPV: %.2f Time Elapsed: %.2f seconds" % (evaluation[6], (end - start)))
  plt.legend(["Train PPV", "Validation PPV"], loc="upper left")
  plt.savefig('results/classifier_croppedscans/ppv_' + file_time + '.png')  #save PPV graph
  plt.close()

  # Plot MCC
  plt.plot(history.history["matthews_correlation_coefficient"])
  plt.plot(history.history["val_matthews_correlation_coefficient"])
  plt.ylabel("Matthews Correlation Coefficient")
  plt.xlabel("Epochs")
  plt.title("Classify Summary: Test PPV: %.2f Time Elapsed: %.2f seconds" % (evaluation[7], (end - start)))
  plt.legend(["Train MCC", "Validation MCC"], loc="upper left")
  plt.savefig('results/classifier_croppedscans/mcc_' + file_time + '.png')  #save MCC graph
  plt.close()

  #if more parameters are to be evaluated the evaluation summary should be brought to find how these are ordered
  """
  # Plot Accuracy
  plt.plot(history.history["binary_accuracy"])
  plt.plot(history.history["val_binary_accuracy"])
  plt.ylabel("Accuracy")
  plt.xlabel("Epochs")
  plt.title('Classify Summary: Test Accuracy: %.2f Time Elapsed: %.2f seconds' % (evaluation[1], (end - start)))
  plt.legend(["Train Accuracy", "Validation Accuracy"], loc="upper left")


  plt.savefig('results/classifier_croppedscans/classifier_' + file_time + '.png')   #save graph
  model.save('results/classifier_croppedscans/classifier_' + file_time + '.h5')     #save model weights in h5 file
  plt.close()


  # Plot Loss
  plt.plot(history.history["loss"])
  plt.plot(history.history["val_loss"])
  plt.ylabel("Loss")
  plt.xlabel("Epochs")
  plt.title("Classify Summary: Test Loss: %.2f Time Elapsed: %.2f seconds" % (evaluation[0], (end - start)))
  plt.legend(["Train Loss", "Validation Loss"], loc="upper left")
  plt.savefig('results/classifier_croppedscans/loss_' + file_time + '.png') #save loss graph
  plt.close()

  # Plot Sensitivity
  plt.plot(history.history["recall"])
  plt.plot(history.history["val_recall"])
  plt.ylabel("Sensitivity")
  plt.xlabel("Epochs")
  plt.title("Classify Summary: Test Sensitivity: %.2f Time Elapsed: %.2f seconds" % (evaluation[2], (end - start)))
  plt.legend(["Train Sensitivity", "Validation Sensitivity"], loc="upper left")
  plt.savefig('results/classifier_croppedscans/sensitivity_' + file_time + '.png')  #save sensitivity graph
  """
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
