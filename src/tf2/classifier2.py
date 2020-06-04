
'''
Basic Keras Code for a convolutional neural network
'''
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import time
import numpy as np 
import tensorflow as tf
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import LearningRateScheduler, ReduceLROnPlateau
import matplotlib.pyplot as plt
from ..data_loader2 import DataGenerator2
from .model import conv0, conv1, conv2, conv3, resnet, resnet2

def Classify2():
  
  # Training Parameters
  epochs = 50
  batch_size = 16
  test_batch_size = 8
  val_batch_size = 8

  # Import Dataset
  data = DataGenerator2(width=256, height=256)
  x_train = data.x_train
  y_train = data.y_train
  x_val = data.x_val
  y_val = data.y_val
  x_test = data.x_test
  y_test = data.y_test
  print("Training DataSet: " + str(x_train.shape) + " " + str(y_train.shape))
  print("Validation DataSet: " + str(x_val.shape) + " " + str(y_val.shape))
  print("Test DataSet: " + str(x_test.shape) + " " + str(y_test.shape))
  
  train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size).shuffle(1000)
  train_dataset = train_dataset.repeat()
    
  val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(val_batch_size).shuffle(1000)
  val_dataset = val_dataset.repeat()

  test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(test_batch_size).shuffle(1000)
  test_dataset = test_dataset.repeat()

  def lr_schedule(epoch):
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

  model = resnet2(HEIGHT, WIDTH, CHANNELS, NUM_OUTPUTS);
  model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['binary_accuracy', tf.keras.metrics.Recall()])
  model.summary()

  # Prepare callbacks
  #lr_scheduler = LearningRateScheduler(lr_schedule)
  #lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=1, patience=5, min_lr=0.5e-6)
  #callbacks = [lr_reducer, lr_scheduler]

  start = time.time();
  history = model.fit(train_dataset, 
          epochs=epochs, 
          steps_per_epoch=108,
          validation_data=val_dataset,
          validation_steps = 27)
  
  evaluation = model.evaluate(x_test, y_test, verbose=1)
  end = time.time()

  print('Classify Summary: Test Accuracy: %.2f Time Elapsed: %.2f seconds' % (evaluation[1], (end - start)) )
  print('Classify Summary: Test Loss: %.2f Time Elapsed: %.2f seconds' % (evaluation[0], (end - start)) )
  print('Classify Summary: Sensitivity: %.2f Time Elapsed: %.2f seconds' % (evaluation[2], (end - start)) )
  # Plot Accuracy 
  plt.plot(history.history["binary_accuracy"])
  plt.plot(history.history["val_binary_accuracy"])
  plt.ylabel("Accuracy")
  plt.xlabel("Epochs")
  plt.title('Classify Summary: Test Accuracy: %.2f Time Elapsed: %.2f seconds' % (evaluation[1], (end - start)))
  plt.legend(["Train Accuracy", "Validation Accuracy"], loc="upper left")

  import datetime
  file_time = datetime.datetime.today().strftime('_%Y-%m-%d__%I-%M')
  plt.savefig('results/tf2/classifier_' + file_time + '.png')
  model.save('results/tf2/classifier_' + file_time + '.h5') 
  plt.close()

  print(model.metrics_names)
  plt.plot(history.history["loss"])
  plt.plot(history.history["val_loss"])
  plt.ylabel("Loss")
  plt.xlabel("Epochs")
  plt.title("Classify Summary: Test Loss: %.2f Time Elapsed: %.2f seconds" % (evaluation[0], (end - start)))
  plt.legend(["Train Loss", "Validation Loss"], loc="upper left")
  plt.savefig('results/tf2/loss_' + file_time + '.png')

  plt.close()

  plt.plot(history.history["recall"])
  plt.plot(history.history["val_recall"])
  plt.ylabel("Sensitivity")
  plt.xlabel("Epochs")
  plt.title("Classify Summary: Test Sensitivity: %.2f Time Elapsed: %.2f seconds" % (evaluation[2], (end - start)))
  plt.legend(["Train Sensitivity", "Validation Sensitivity"], loc="upper left")
  plt.savefig('results/tf2/sensitivity_' + file_time + '.png') 

 