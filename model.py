
import tensorflow as tf

# Network Parameters
NUM_OUTPUTS = 1

# Network Architecture
def conv_network_1(x):

    # Convolutional layers and max pool
    he_init = tf.contrib.layers.variance_scaling_initializer()
    conv1 = tf.layers.conv2d(x,     16, [3, 3], padding="SAME", activation=tf.nn.relu, kernel_initializer=he_init, name='h1')
    conv2 = tf.layers.conv2d(conv1, 16, [3, 3], padding="SAME", activation=tf.nn.relu, kernel_initializer=he_init, name='h2')
    pool1 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
    
    # Reshape to fit to fully connected layer input
    flatten = tf.contrib.layers.flatten(pool1)

    # Fully-connected layers 
    fc1 = tf.layers.dense(flatten, 512, activation=tf.nn.relu, kernel_initializer=he_init, name='fc1')  
    fc2 = tf.layers.dense(fc1,     256, activation=tf.nn.relu, kernel_initializer=he_init, name='fc2')
    out = tf.layers.dense(fc2,     NUM_OUTPUTS,  activation=None, name='logits')  
    return out

def conv_network_2(x): 

    with tf.device('/device:GPU:0'):

        # Convolutional layers and max pool
        he_init = tf.contrib.layers.variance_scaling_initializer()
        conv1 = tf.layers.conv2d(x,     32, [3, 3], padding="SAME", activation=tf.nn.relu, kernel_initializer=he_init, name='h1')
        conv2 = tf.layers.conv2d(conv1, 32, [3, 3], padding="SAME", activation=tf.nn.relu, kernel_initializer=he_init, name='h2')
        pool1 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
        
        he_init = tf.contrib.layers.variance_scaling_initializer()
        conv3 = tf.layers.conv2d(pool1, 64, [3, 3], padding="SAME", activation=tf.nn.relu, kernel_initializer=he_init, name='h3')
        conv4 = tf.layers.conv2d(conv3, 64, [3, 3], padding="SAME", activation=tf.nn.relu, kernel_initializer=he_init, name='h4')
        pool2 = tf.layers.max_pooling2d(inputs=conv4, pool_size=[2, 2], strides=2)


    with tf.device('/device:GPU:1'):

        he_init = tf.contrib.layers.variance_scaling_initializer()
        conv5 = tf.layers.conv2d(pool2, 128, [3, 3], padding="SAME", activation=tf.nn.relu, kernel_initializer=he_init, name='h5')
        conv6 = tf.layers.conv2d(conv5, 128, [3, 3], padding="SAME", activation=tf.nn.relu, kernel_initializer=he_init, name='h6')
        pool3 = tf.layers.max_pooling2d(inputs=conv6, pool_size=[2, 2], strides=2)

        # Reshape to fit to fully connected layer input
        flatten = tf.contrib.layers.flatten(pool3)

        # Fully-connected layers 
        fc1 = tf.layers.dense(flatten, 2048, activation=tf.nn.relu, kernel_initializer=he_init, name='fc1')  
        fc2 = tf.layers.dense(fc1,     2048, activation=tf.nn.relu, kernel_initializer=he_init, name='fc2')
        out = tf.layers.dense(fc2,     NUM_OUTPUTS,  activation=None, name='logits')  
    return out 
