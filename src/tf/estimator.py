
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
import matplotlib.pyplot as plt
import h5py

from ..data_loader import DataGenerator
from .model import conv_network_1, conv_network_2, unet_like_network

class Estimator:

  def __init__(self,
    num_steps = 10000,
    batch_size = 128,
    display_step = 100,
    learning_rate = 0.001,
    save_step = 1000,
    doRestore = False):

    # Training Parameters
    self.num_steps = num_steps    
    self.batch_size = batch_size
    self.display_step = display_step
    self.learning_rate = learning_rate
    self.save_step = save_step
    self.doRestore = doRestore

    # Import Dataset
    self.data = DataGenerator(useBinaryClassify = False)

    # Network Parameters
    self.WIDTH = self.data.WIDTH
    self.HEIGHT = self.data.HEIGHT
    self.CHANNELS = 1
    self.NUM_OUTPUTS = 1

    self.initNetwork()
    self.train()
    self.showResults()

  def initNetwork(self):

    # Network Varibles and placeholders
    self.X = tf.placeholder(tf.float32, [None, self.HEIGHT, self.WIDTH, self.CHANNELS])  # Input
    self.Y = tf.placeholder(tf.float32, [None, self.NUM_OUTPUTS]) # Truth Data - Output
    self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')

    # Define loss and optimizer 
    self.prediction = conv_network_2(self.X) 
    self.loss = tf.reduce_mean(tf.square(self.prediction - self.Y))
    self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
    self.trainer = self.optimizer.minimize(self.loss, global_step=self.global_step)

    # Setup Saver
    self.saver = tf.train.Saver()

    # Initalize varibles, and run network
    init = tf.global_variables_initializer()
    self.sess = tf.Session() 
    self.sess.run(init)

    if(self.doRestore):
      ckpt = tf.train.get_checkpoint_state('./checkpoints/estimator/' )
      if(ckpt and ckpt.model_checkpoint_path):
        print('Restoring Prev. Model ....')
        self.saver.restore(self.sess, ckpt.model_checkpoint_path)
        print('Model Loaded....')

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
    for _ in range(self.num_steps):
        batch_xs, batch_ys = self.data.next_batch(self.batch_size)
        self.sess.run( self.trainer, feed_dict={self.X: batch_xs, self.Y: batch_ys} )

        step = self.sess.run(self.global_step)

        if(step % self.display_step == 0):
          train_loss = self.sess.run(self.loss, feed_dict={ self.X:batch_xs, self.Y: batch_ys })  
          test_loss = self.sess.run(self.loss, feed_dict={ self.X:self.data.x_test, self.Y:self.data.y_test })

          _steps.append(step)
          _loss_train.append(train_loss)
          _loss_test.append(test_loss)

          time_elapsed = time.time() - start_time
          print("Step: " + str(step) + " Train Loss: " + str(train_loss) + " Test Loss: " + str(test_loss) + " Time Elapsed: " + str(round(time_elapsed)) + " secs") 

        if(step % self.save_step == 0):
          self.saver.save(self.sess, './checkpoints/estimator/hip', global_step=self.global_step)

    # Get prediction after network is trained
    pred_train = self.sess.run(self.prediction, feed_dict={ self.X: batch_xs })
    pred_test = self.sess.run(self.prediction, feed_dict={ self.X: self.data.x_test })

    # Wrap results in results object
    self.results = lambda: None
    self.results.steps = _steps;
    self.results.loss_train = _loss_train
    self.results.loss_test = _loss_test
    self.results.pred_train = pred_train
    self.results.pred_test = pred_test
    self.results.x_train = batch_xs
    self.results.y_train = batch_ys
    self.results.x_test = self.data.x_test
    self.results.y_test = self.data.y_test
    
  def showResults(self):
    
    data = self.data
    results = self.results

    # Plot Accuracy 
    plt.figure()
    plt.plot(results.steps, 10 * np.log(results.loss_train), label="Training Loss")
    plt.plot(results.steps, 10 * np.log(results.loss_test), label="Test Loss")
    plt.legend()
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.title("Loss for Angle Estimatation")
    plt.savefig('results/Loss.png')

    if(data.useWhitening):
      results.y_train     = data.undo_whitening(results.y_train,     data.ang_mean, data.ang_std)
      results.pred_train  = data.undo_whitening(results.pred_train,  data.ang_mean, data.ang_std)
      data.y_test         = data.undo_whitening(data.y_test,         data.ang_mean, data.ang_std)
      results.pred_test   = data.undo_whitening(results.pred_test,   data.ang_mean, data.ang_std)

    plt.figure()
    plt.plot(results.pred_train[:, 0], label="Predicted Values")
    plt.plot(results.y_train[:, 0], label="Truth Values")
    plt.legend()
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.title("Loss for Angle Estimatation - Training Data")
    plt.savefig('results/Results-Training.png')

    plt.figure()
    plt.plot(results.pred_test[:, 0], label="Predicted Values")
    plt.plot(data.y_test[:, 0], label="Truth Values")
    plt.legend()
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.title("Loss for Angle Estimatation - Test Data" )
    plt.savefig('results/Results.png')
