
'''
Basic Keras Code for a convolutional neural network
'''
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import time
import tensorflow as tf
from tensorflow.keras import regularizers
import matplotlib.pyplot as plt
from ..data_loader import DataGenerator
from .model import conv0, conv1, conv2, conv3, resnet

def Classify():
  
  # Training Parameters
  epochs = 50 
  batch_size = 32 
  test_batch_size = 32

  # Import Dataset
  data = DataGenerator(width=256, height=256)
  x_train = data.x_train
  y_train = data.y_train
  x_test = data.x_test
  y_test = data.y_test
  print("Training DataSet: " + str(x_train.shape) + " " + str(y_train.shape))
  print("Test DataSet: " + str(x_test.shape) + " " + str(y_test.shape))
  
  train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size).shuffle(1000)
  train_dataset = train_dataset.map(lambda x, y: (tf.image.random_flip_left_right(x), y))
  train_dataset = train_dataset.repeat()
  
  valid_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(test_batch_size).shuffle(1000)
  valid_dataset = valid_dataset.repeat()

  # Network Parameters
  WIDTH = data.WIDTH
  HEIGHT = data.HEIGHT
  CHANNELS = 1
  NUM_OUTPUTS = 2

  model = resnet(HEIGHT, WIDTH, CHANNELS, NUM_OUTPUTS);
  model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
  model.summary()

  start = time.time();
  history = model.fit(train_dataset, 
          epochs=epochs, 
          steps_per_epoch=200,
          validation_data=valid_dataset,
          validation_steps=3)
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