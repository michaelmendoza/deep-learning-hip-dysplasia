
'''
Tensorflow Convolution Angle Estimator
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import numpy as np
#import matplotlib as plt
#import h5py

from ..data_loader import DataGenerator
from .model import conv_network_1, conv_network_2, unet_like_network, VGG

class Classifier:

  def __init__(self,
    num_steps = 10000,
    batch_size = 128,
    display_step = 100,
    learning_rate = 0.001):

    # Training Parameters
    self.num_steps = num_steps    
    self.batch_size = batch_size
    self.display_step = display_step
    self.learning_rate = learning_rate

    # Import Dataset
    self.data = DataGenerator()

    # Network Parameters
    self.WIDTH = self.data.WIDTH
    self.HEIGHT = self.data.HEIGHT
    self.CHANNELS = 1
    self.NUM_OUTPUTS = 2

    self.initNetwork()
    self.train()
    #self.showResults()

  def initNetwork(self):

    # Network Varibles and placeholders
    self.X = tf.placeholder(tf.float32, [None, self.HEIGHT, self.WIDTH, self.CHANNELS])  # Input
    self.Y = tf.placeholder(tf.float32, [None, self.NUM_OUTPUTS]) # Truth Data - Output

    # Define loss and optimizer 
    self.logits = VGG(self.X, self.NUM_OUTPUTS) 
    self.prediction =  tf.nn.softmax(self.logits)

    self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.Y))
    self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
    self.trainer = self.optimizer.minimize(self.loss)

    # Evaluate model
    self.correct_pred = tf.equal(tf.argmax(self.prediction, 1), tf.argmax(self.Y, 1))
    self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))

    # Initalize varibles, and run network
    init = tf.global_variables_initializer()
    self.sess = tf.Session() 
    self.sess.run(init)

  def train(self):
    print ('Start Training: BatchSize:', self.batch_size,
                          ' LearningRate:', self.learning_rate, 
                          ' Total Steps:', self.num_steps)

    import time
    start_time = time.time()

    # Train network
    _steps = []
    _loss_train = []
    _loss_test = []
    _acc = []

    for step in range(self.num_steps):
        batch_xs, batch_ys = self.data.next_batch(self.batch_size)
        self.sess.run( self.trainer, feed_dict={self.X: batch_xs, self.Y: batch_ys} )

        if(step % self.display_step == 0):
          train_loss = self.sess.run(self.loss, feed_dict={ self.X:batch_xs, self.Y: batch_ys })  
          test_loss = self.sess.run(self.loss, feed_dict={ self.X:self.data.x_test, self.Y:self.data.y_test })
          acc = self.sess.run(self.accuracy, feed_dict={ self.X:self.data.x_test, self.Y:self.data.y_test })
          
          _steps.append(step)
          _loss_train.append(train_loss)
          _loss_test.append(test_loss)
          _acc.append(acc)

          time_elapsed = time.time() - start_time
          print("Step: " + str(step) + " Train Loss: " + str(train_loss) + " Test Loss: " + str(test_loss) + " Test Accuracy: " + str(acc) +  " Time Elapsed: " + str(round(time_elapsed)) + " secs") 

    # Get prediction after network is trained
    pred_train = self.sess.run(1 - tf.argmax(self.prediction, 1), feed_dict={ self.X: batch_xs })
    pred_test = self.sess.run(1 - tf.argmax(self.prediction, 1), feed_dict={ self.X: self.data.x_test })
    
    # Wrap results in results object
    self.results = lambda: None
    self.results.steps = _steps;
    self.results.loss_train = _loss_train
    self.results.loss_test = _loss_test
    self.results.accuracy = _acc
    self.results.pred_train = pred_train
    self.results.pred_test = pred_test
    self.results.x_train = batch_xs
    self.results.y_train = batch_ys
    self.results.x_test = self.data.x_test
    self.results.y_test = self.data.y_test

    # Save data in h5py
    #hf = h5py.File('results/data.h5', 'w')
    #hf.create_dataset('results', data=self.results)
    #hf.close() 
'''  
  def showResults(self):
    
    data = self.data
    results = self.results

    # Plot Loss
    plt.pyplot.figure()
    plt.pyplot.plot(results.steps, results.loss_train, label="Training Loss")
    plt.pyplot.plot(results.steps, results.loss_test, label="Test Loss")
    plt.pyplot.legend()
    plt.pyplot.xlabel("Steps")
    plt.pyplot.ylabel("Loss")
    plt.pyplot.title("Loss for Angle Estimatation")
    plt.pyplot.savefig('results/Loss.png')

    # Plot Accuracy
    plt.pyplot.figure() 
    plt.pyplot.plot(results.steps, results.accuracy, label="Training Accuracy")
    plt.pyplot.legend()
    plt.pyplot.xlabel("Steps")
    plt.pyplot.ylabel("Accuracy")
    plt.pyplot.title("Accuracy for Angle Estimatation")
    plt.pyplot.savefig('results/Accuracy.png')

    print(results.pred_train.shape)
    print(results.y_train.shape)

    plt.pyplot.figure()
    plt.pyplot.plot(results.pred_train, label="Predicted Values")
    plt.pyplot.plot(results.y_train[:, 0], label="Truth Values")
    plt.pyplot.legend()
    plt.pyplot.xlabel("Steps")
    plt.pyplot.ylabel("Loss")
    plt.pyplot.title("Loss for Angle Estimatation - Training Data")
    plt.pyplot.savefig('results/Results-Training.png')

    plt.pyplot.figure()
    plt.pyplot.plot(results.pred_test, label="Predicted Values")
    plt.pyplot.plot(data.y_test[:, 0], label="Truth Values")
    plt.pyplot.legend()
    plt.pyplot.xlabel("Steps")
    plt.pyplot.ylabel("Loss")
    plt.pyplot.title("Loss for Angle Estimatation - Test Data" )
    plt.pyplot.savefig('results/Results.png')
    '''
