
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
from ..data_loader import DataGenerator
from .model import conv0, conv1, conv2, conv3, resnet, resnet2

def Classify():
  
  # Training Parameters
  epochs = 100
  batch_size = 16 
  test_batch_size = 8

  # Import Dataset
  data = DataGenerator(width=256, height=256)
  x_train = data.x_train
  y_train = data.y_train
  x_test = data.x_test
  y_test = data.y_test
  print("Training DataSet: " + str(x_train.shape) + " " + str(y_train.shape))
  print("Test DataSet: " + str(x_test.shape) + " " + str(y_test.shape))
  
  train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size).shuffle(1000)
  train_dataset = train_dataset.repeat()
    
  valid_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(test_batch_size).shuffle(1000)
  valid_dataset = valid_dataset.repeat()

  def lr_schedule(epoch):
      """ Learning Rate Schedule. Learning rate is scheduled to be reduced 
      after 80, 120, 160, 180 epochs. Called automatically every epoch as 
      part of callbacks during training. """

      lr = 1e-3
      if epoch > 180:
          lr *= 0.5e-3
      elif epoch > 160:
          lr *= 1e-3
      elif epoch > 120:
          lr *= 1e-2
      elif epoch > 80:
          lr *= 1e-1
      print('Learning rate: ', lr)
      return lr

  # Network Parameters
  WIDTH = data.WIDTH
  HEIGHT = data.HEIGHT
  CHANNELS = 1
  NUM_OUTPUTS = 2

  model = resnet2(HEIGHT, WIDTH, CHANNELS, NUM_OUTPUTS);
  model.compile(optimizer=Adam(learning_rate=lr_schedule(0)), loss='categorical_crossentropy', metrics=['categorical_accuracy'])
  model.summary()

  # Prepare callbacks
  lr_scheduler = LearningRateScheduler(lr_schedule)
  lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=1, patience=5, min_lr=0.5e-6)
  callbacks = [lr_reducer, lr_scheduler]

  start = time.time();
  history = model.fit(train_dataset, 
          epochs=epochs, 
          steps_per_epoch=200,
          validation_data=valid_dataset,
          validation_steps = 10, 
          callbacks=callbacks)
  
  evaluation = model.evaluate(x_test, y_test, verbose=1)
  end = time.time()

  print('Classify Summary: Accuracy: %.2f Time Elapsed: %.2f seconds' % (evaluation[1], (end - start)) )

  # Plot Accuracy 
  plt.plot(history.history["categorical_accuracy"])
  plt.plot(history.history["val_categorical_accuracy"])
  plt.ylabel("Accuracy")
  plt.xlabel("Epochs")
  plt.title('Classify Summary: Accuracy: %.2f Time Elapsed: %.2f seconds' % (evaluation[1], (end - start)))
  plt.legend(["Train Accuracy", "Test Accuracy"], loc="upper left")

  import datetime
  file_time = datetime.datetime.today().strftime('_%Y-%m-%d__%I-%M')
  plt.savefig('results/tf2/classifier_' + file_time + '.png')
  model.save('results/tf2/classifier_' + file_time + '.h5') 