
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

# Import Dataset
data = DataGenerator()

# Training Parameters
learning_rate = 0.001
num_steps = 40000
batch_size = 16 #16 #128
display_step = 1000 #100

# Network Parameters
WIDTH = data.WIDTH
HEIGHT = data.HEIGHT
CHANNELS = 1
NUM_OUTPUTS = 1

# Network Varibles and placeholders
X = tf.placeholder(tf.float32, [None, HEIGHT, WIDTH, CHANNELS])  # Input
Y = tf.placeholder(tf.float32, [None, NUM_OUTPUTS]) # Truth Data - Output

# Define loss and optimizer 
prediction = conv_network_2(X)  # unet_like_network(X)
loss = tf.reduce_mean(tf.square(prediction - Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
trainer = optimizer.minimize(loss)

# Initalize varibles, and run network
init = tf.global_variables_initializer()
sess = tf.Session() #sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
sess.run(init)

print ('Start Training: BatchSize:', batch_size,' LearningRate:', learning_rate, 'Total Steps:', num_steps)

import time
start_time = time.time()

# Train network
_step = []
_loss_train = []
_loss_test = []
for step in range(num_steps):
    batch_xs, batch_ys = data.next_batch(batch_size)
    sess.run( trainer, feed_dict={X: batch_xs, Y: batch_ys} )

    if(step % display_step == 0):
      train_loss = sess.run(loss, feed_dict={X: batch_xs, Y:batch_ys})  
      test_loss = sess.run(loss, feed_dict={X: data.x_test, Y:data.y_test})

      _step.append(step)
      _loss_train.append(train_loss)
      _loss_test.append(test_loss)

      time_elapsed = time.time() - start_time
      print("Step: " + str(step) + " Train Loss: " + str(train_loss) + " Test Loss: " + str(test_loss) + " Time Elapsed: " + str(round(time_elapsed)) + " secs") 

# Get prediction after network is trained
pred_train = sess.run(prediction, feed_dict={X: batch_xs})
pred_test = sess.run(prediction, feed_dict={X: data.x_test})

# Save data in h5py
hf = h5py.File('results/data.h5', 'w')
hf.create_dataset('x_test', data=data.x_test)
hf.create_dataset('y_test', data=data.y_test)
hf.create_dataset('results', data=pred_test)
hf.close()

# Plot Accuracy 
plt.figure()
plt.plot(_step, 10 * np.log(_loss_train), label="Training Accuracy")
plt.plot(_step, 10 * np.log(_loss_test), label="Test Accuracy")
plt.legend()
plt.xlabel("Steps")
plt.ylabel("Loss")
plt.title("Loss for Angle Estimatation")
plt.savefig('results/Loss.png')
#plt.show()

if(data.useWhitening):
  batch_ys    = data.undo_whitening(batch_ys,    data.ang_mean, data.ang_std)
  pred_train  = data.undo_whitening(pred_train,  data.ang_mean, data.ang_std)
  data.y_test = data.undo_whitening(data.y_test, data.ang_mean, data.ang_std)
  pred_test   = data.undo_whitening(pred_test,   data.ang_mean, data.ang_std)

plt.figure()
plt.plot(pred_train[:, 0], label="Predicted Values")
plt.plot(batch_ys[:, 0], label="Truth Values")
plt.legend()
plt.xlabel("Steps")
plt.ylabel("Loss")
plt.title("Loss for Angle Estimatation - Training Data")
plt.savefig('results/Results-Training.png')
#plt.show()

plt.figure()
plt.plot(pred_test[:, 0], label="Predicted Values")
plt.plot(data.y_test[:, 0], label="Truth Values")
plt.legend()
plt.xlabel("Steps")
plt.ylabel("Loss")
plt.title("Loss for Angle Estimatation - Test Data" )
plt.savefig('results/Results.png')
#plt.show()
