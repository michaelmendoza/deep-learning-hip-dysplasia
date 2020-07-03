
'''
Basic Keras Code for a convolutional neural network
'''

#this is the classifier used if using outcome as the diagnostic parameter
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import time
import numpy as np 
import tensorflow as tf
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import LearningRateScheduler, ReduceLROnPlateau
import matplotlib.pyplot as plt
from ..data_loader import DataGenerator   #for this classifier we import only DataGenerator since we are using the outcome
from .model import conv0, conv1, conv2, conv3, resnet, resnet2

def Classify():
  
  # Training Parameters
  epochs = 50
  batch_size = 16 
  test_batch_size = 8
  val_batch_size = 8

  # Import Dataset, in this case the y label is the outcome
  data = DataGenerator(width=256, height=256)   #in this case we have specified the width to be 256, larger than the standard in the dataloader file
  x_train = data.x_train  
  y_train = data.y_train
  x_val = data.x_val
  x_test = data.x_test
  y_test = data.y_test
  y_val = data.y_val
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
      """ Learning Rate Schedule. Learning rate is scheduled to be reduced 
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

  model = resnet2(HEIGHT, WIDTH, CHANNELS, NUM_OUTPUTS);    #model chosen is resnet2
  model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['binary_accuracy', tf.keras.metrics.Recall()]) #very important line about model characteristics
  model.summary()

  # Prepare callbacks
  #lr_scheduler = LearningRateScheduler(lr_schedule)
  #lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=1, patience=5, min_lr=0.5e-6)
  #callbacks = [lr_reducer, lr_scheduler]

  start = time.time();
  history = model.fit(train_dataset, 
          epochs=epochs, 
          steps_per_epoch=108,    #108 since training dataset is 864 aprox, 864/batchsize = 864/16 = 54, and this is usually x2 hence 108
          validation_data=val_dataset,
          validation_steps = 27)    #27 since validation dataset is 108 aprox, 108/batchsize = 108/8 = 13.5, and this is usually x2 hence 27
  
  evaluation = model.evaluate(x_test, y_test, verbose=1)    #after training and validation end (50 epochs), we finish with testing
  end = time.time()

  print('Classify Summary: Accuracy: %.2f Time Elapsed: %.2f seconds' % (evaluation[1], (end - start)) )
  print('Classify Summary: Test Loss: %.2f Time Elapsed: %.2f seconds' % (evaluation[0], (end - start)) )
  print('Classify Summary: Sensitivity: %.2f Time Elapsed: %.2f seconds' % (evaluation[2], (end - start)) )   #note sensitivity=recall
  
  #if more parameters are to be evaluated the evaluation summary should be brought to find how these are ordered

  # Plot Accuracy 
  plt.plot(history.history["binary_accuracy"])
  plt.plot(history.history["val_binary_accuracy"])
  plt.ylabel("Accuracy")
  plt.xlabel("Epochs")
  plt.title('Classify Summary: Test Accuracy: %.2f Time Elapsed: %.2f seconds' % (evaluation[1], (end - start)))
  plt.legend(["Train Accuracy", "Validation Accuracy"], loc="upper left")

  import datetime
  file_time = datetime.datetime.today().strftime('_%Y-%m-%d__%I-%M')  #get date today for filename
  plt.savefig('results/tf2/classifier_' + file_time + '.png')   #save graph
  model.save('results/tf2/classifier_' + file_time + '.h5')   #save model weights in h5 file
  plt.close()

  print(model.metrics_names)
  
  # Plot Loss
  plt.plot(history.history["loss"])
  plt.plot(history.history["val_loss"])
  plt.ylabel("Loss")
  plt.xlabel("Epochs")
  plt.title("Classify Summary: Test Loss: %.2f Time Elapsed: %.2f seconds" % (evaluation[0], (end - start)))
  plt.legend(["Train Loss", "Validation Loss"], loc="upper left")
  plt.savefig('results/tf2/loss_' + file_time + '.png')   #save loss graph
  plt.close()
  
  # Plot Sensitivity
  plt.plot(history.history["recall"])
  plt.plot(history.history["val_recall"])
  plt.ylabel("Sensitivity")
  plt.xlabel("Epochs")
  plt.title("Classify Summary: Test Sensitivity: %.2f Time Elapsed: %.2f seconds" % (evaluation[2], (end - start)))
  plt.legend(["Train Sensitivity", "Validation Sensitivity"], loc="upper left")
  plt.savefig('results/tf2/sensitivity_' + file_time + '.png')  #save graph
