
'''
Basic Keras Code for a convolutional neural network
'''
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import time
import tensorflow as tf
import matplotlib.pyplot as plt
from ..data_loader import DataGenerator

def Classify():
  
  # Training Parameters
  epochs = 10
  batch_size = 16
  validation_split = 0.2
  shuffle = True

  # Import Dataset
  data = DataGenerator()
  x_train = data.x_train
  y_train = data.y_train
  x_test = data.x_test
  y_test = data.y_test

  # Network Parameters
  WIDTH = data.WIDTH
  HEIGHT = data.HEIGHT
  CHANNELS = 1
  NUM_OUTPUTS = 2

  model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(64, (3, 3), padding="same", activation=tf.nn.relu, input_shape=(HEIGHT, WIDTH, CHANNELS)),
    tf.keras.layers.Conv2D(64, (3, 3), padding="same", activation=tf.nn.relu),
    tf.keras.layers.MaxPooling2D(padding="same", strides=(2, 2), pool_size=(2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(NUM_OUTPUTS, activation=tf.nn.softmax)
  ])

  model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
  model.summary()

  start = time.time();
  history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=validation_split, shuffle=shuffle)
  evaluation = model.evaluate(x_test, y_test, verbose=1)
  end = time.time()
  print("Training Complete in " + "{0:.2f}".format(end - start) + " secs" )

  # Plot Accuracy 
  plt.plot(history.history["acc"]);
  plt.plot(history.history["val_acc"]);
  plt.ylabel("Accuracy")
  plt.xlabel("Epochs")
  plt.legend(["Train Accuracy", "Test Accuracy"], loc="upper left")
  plt.show();
