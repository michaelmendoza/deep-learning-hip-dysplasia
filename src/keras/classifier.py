
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

def Classify():
  
  # Training Parameters
  epochs = 500 
  batch_size = 32 
  validation_split = 0.2
  shuffle = True
  
  # Import Dataset
  data = DataGenerator(width=128, height=128, useRandomOrder = False)
  x_train = data.x_train
  y_train = data.y_train
  x_test = data.x_test
  y_test = data.y_test
  print("Training DataSet: " + str(x_train.shape) + " " + str(y_train.shape))

  # Network Parameters
  WIDTH = data.WIDTH
  HEIGHT = data.HEIGHT
  CHANNELS = 1
  NUM_OUTPUTS = 2

  weight_decay = 1e-4
  model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), padding="same", activation=tf.nn.relu, input_shape=(HEIGHT, WIDTH, CHANNELS)),
    tf.keras.layers.Conv2D(32, (3, 3), padding="same", activation=tf.nn.relu),
    tf.keras.layers.MaxPooling2D(padding="same", strides=(2, 2), pool_size=(2, 2)),
    
    tf.keras.layers.Conv2D(64, (3, 3), padding="same", activation=tf.nn.relu),
    tf.keras.layers.Conv2D(64, (3, 3), padding="same", activation=tf.nn.relu),
    tf.keras.layers.MaxPooling2D(padding="same", strides=(2, 2), pool_size=(2, 2)),

    tf.keras.layers.Conv2D(128, (3, 3), padding="same", activation=tf.nn.relu),
    tf.keras.layers.Conv2D(128, (3, 3), padding="same", activation=tf.nn.relu),
    tf.keras.layers.MaxPooling2D(padding="same", strides=(2, 2), pool_size=(2, 2)),
    
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(2048, activation=tf.nn.relu),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(2048, activation=tf.nn.relu),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(NUM_OUTPUTS, activation=tf.nn.softmax)
  ])
  
  model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
  model.summary()

  start = time.time();
  history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=validation_split, shuffle=shuffle)
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
  plt.savefig('results/keras/classifier_' + file_time + '.png')
  model.save('results/keras/classifier_' + file_time + '.h5')