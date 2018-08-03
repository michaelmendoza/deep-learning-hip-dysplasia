
import tensorflow as tf

# Network Architecture
def conv_network_1(x, NUM_OUTPUTS = 1):

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

def conv_network_2(x, NUM_OUTPUTS = 1): 
    
    # Convolutional layers and max pool
    he_init = tf.contrib.layers.variance_scaling_initializer()
    conv1 = tf.layers.conv2d(x,     32, [3, 3], padding="SAME", activation=tf.nn.relu, kernel_initializer=he_init, name='h1')
    conv2 = tf.layers.conv2d(conv1, 32, [3, 3], padding="SAME", activation=tf.nn.relu, kernel_initializer=he_init, name='h2')
    pool1 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
    
    he_init = tf.contrib.layers.variance_scaling_initializer()
    conv3 = tf.layers.conv2d(pool1, 64, [3, 3], padding="SAME", activation=tf.nn.relu, kernel_initializer=he_init, name='h3')
    conv4 = tf.layers.conv2d(conv3, 64, [3, 3], padding="SAME", activation=tf.nn.relu, kernel_initializer=he_init, name='h4')
    pool2 = tf.layers.max_pooling2d(inputs=conv4, pool_size=[2, 2], strides=2)

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

def unet_like_network(x, NUM_OUTPUTS = 1):
    he_init = tf.contrib.layers.variance_scaling_initializer()
    conv1 = tf.layers.conv2d(x,     32, [3, 3], padding="SAME", activation=tf.nn.relu, kernel_initializer=he_init, name='Conv1')
    conv1 = tf.layers.conv2d(conv1, 32, [3, 3], padding="SAME", activation=tf.nn.relu, kernel_initializer=he_init, name='Conv1-2')
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
    
    conv2 = tf.layers.conv2d(pool1, 64, [3, 3], padding="SAME", activation=tf.nn.relu, kernel_initializer=he_init, name='Conv2')
    conv2 = tf.layers.conv2d(conv2, 64, [3, 3], padding="SAME", activation=tf.nn.relu, kernel_initializer=he_init, name='Conv2-2')
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

    conv3 = tf.layers.conv2d(pool2, 128, [3, 3], padding="SAME", activation=tf.nn.relu, kernel_initializer=he_init, name='Conv3')
    conv3 = tf.layers.conv2d(conv3, 128, [3, 3], padding="SAME", activation=tf.nn.relu, kernel_initializer=he_init, name='Conv3-3')
    pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], strides=2)    

    conv4 = tf.layers.conv2d(pool3, 256, [3, 3], padding="SAME", activation=tf.nn.relu, kernel_initializer=he_init, name='Conv4')
    conv4 = tf.layers.conv2d(conv4, 256, [3, 3], padding="SAME", activation=tf.nn.relu, kernel_initializer=he_init, name='Conv4-2')
    pool4 = tf.layers.max_pooling2d(inputs=conv4, pool_size=[2, 2], strides=2)    

    conv5 = tf.layers.conv2d(pool4, 512, [3, 3], padding="SAME", activation=tf.nn.relu, kernel_initializer=he_init, name='Conv5')
    conv5 = tf.layers.conv2d(conv5, 512, [3, 3], padding="SAME", activation=tf.nn.relu, kernel_initializer=he_init, name='Conv5-2')
 
    up6 = tf.layers.conv2d_transpose(conv5, 256, [3, 3], strides=2, padding="SAME", name='Up6')
    up6 = tf.concat([up6, conv4], 3)
    conv6 = tf.layers.conv2d(up6, 256, [3, 3], padding="SAME", activation=tf.nn.relu, kernel_initializer=he_init, name='Conv6')
    conv6 = tf.layers.conv2d(conv6, 256, [3, 3], padding="SAME", activation=tf.nn.relu, kernel_initializer=he_init, name='Conv6-2')
    
    up7 = tf.layers.conv2d_transpose(conv6, 128, [3, 3], strides=2, padding="SAME", name='Up7')
    up7 = tf.concat([up7, conv3], 3)
    conv7 = tf.layers.conv2d(up7, 128, [3, 3], padding="SAME", activation=tf.nn.relu, kernel_initializer=he_init, name='Conv7')
    conv7 = tf.layers.conv2d(conv7, 128, [3, 3], padding="SAME", activation=tf.nn.relu, kernel_initializer=he_init, name='Conv7-2')

    up8 = tf.layers.conv2d_transpose(conv7, 64, [3, 3], strides=2, padding="SAME", name='Up8')
    up8 = tf.concat([up8, conv2], 3)
    conv8 = tf.layers.conv2d(up8, 64, [3, 3], padding="SAME", activation=tf.nn.relu, kernel_initializer=he_init, name='Conv8')
    conv8 = tf.layers.conv2d(conv8, 64, [3, 3], padding="SAME", activation=tf.nn.relu, kernel_initializer=he_init, name='Conv8-2')
 
    up9 = tf.layers.conv2d_transpose(conv8, 32, [3, 3], strides=2, padding="SAME", name='Up9')
    up9 = tf.concat([up9, conv1], 3)
    conv9 = tf.layers.conv2d(up9, 32, [3, 3], padding="SAME", activation=tf.nn.relu, kernel_initializer=he_init, name='Conv9')
    conv9 = tf.layers.conv2d(conv9, 32, [3, 3], padding="SAME", activation=tf.nn.relu, kernel_initializer=he_init, name='Conv9-2')
    
    # Reshape to fit to fully connected layer input
    flatten = tf.contrib.layers.flatten(conv9) 

    # Fully-connected layers 
    fc1 = tf.layers.dense(flatten, 1024, activation=tf.nn.relu, kernel_initializer=he_init, name='fc1') 
    fc2 = tf.layers.dense(fc1,     1024, activation=tf.nn.relu, kernel_initializer=he_init, name='fc2') 
    out = tf.layers.dense(fc2,     NUM_OUTPUTS,  activation=None, name='logits') 
    return out

def VGG(x, NUM_OUTPUTS = 1):
	he_init = tf.contrib.layers.variance_scaling_initializer()
    conv1 = tf.layers.conv2d(x,     64, [3, 3], padding="SAME", activation=tf.nn.relu, kernel_initializer=he_init, name='Conv1')
    conv1 = tf.layers.conv2d(conv1, 64, [3, 3], padding="SAME", activation=tf.nn.relu, kernel_initializer=he_init, name='Conv1-2')
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
    
    conv2 = tf.layers.conv2d(pool1, 128, [3, 3], padding="SAME", activation=tf.nn.relu, kernel_initializer=he_init, name='Conv2')
    conv2 = tf.layers.conv2d(conv2, 128, [3, 3], padding="SAME", activation=tf.nn.relu, kernel_initializer=he_init, name='Conv2-2')
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

    conv3 = tf.layers.conv2d(pool2, 256, [3, 3], padding="SAME", activation=tf.nn.relu, kernel_initializer=he_init, name='Conv3')
    conv3 = tf.layers.conv2d(conv3, 256, [3, 3], padding="SAME", activation=tf.nn.relu, kernel_initializer=he_init, name='Conv3-2')
    conv3 = tf.layers.conv2d(conv3, 256, [3, 3], padding="SAME", activation=tf.nn.relu, kernel_initializer=he_init, name='Conv3-3')
    pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], strides=2)    

    conv4 = tf.layers.conv2d(pool3, 512, [3, 3], padding="SAME", activation=tf.nn.relu, kernel_initializer=he_init, name='Conv4')
    conv4 = tf.layers.conv2d(conv4, 512, [3, 3], padding="SAME", activation=tf.nn.relu, kernel_initializer=he_init, name='Conv4-2')
    conv4 = tf.layers.conv2d(conv4, 512, [3, 3], padding="SAME", activation=tf.nn.relu, kernel_initializer=he_init, name='Conv4-3')
    pool4 = tf.layers.max_pooling2d(inputs=conv4, pool_size=[2, 2], strides=2) 

    conv5 = tf.layers.conv2d(pool4, 512, [3, 3], padding="SAME", activation=tf.nn.relu, kernel_initializer=he_init, name='Conv5')
    conv5 = tf.layers.conv2d(conv5, 512, [3, 3], padding="SAME", activation=tf.nn.relu, kernel_initializer=he_init, name='Conv5-2')
    conv5 = tf.layers.conv2d(conv5, 512, [3, 3], padding="SAME", activation=tf.nn.relu, kernel_initializer=he_init, name='Conv5-3')
    pool5 = tf.layers.max_pooling2d(inputs=conv5, pool_size=[2, 2], strides=2)       

    # Reshape to fit to fully connected layer input
    flatten = tf.contrib.layers.flatten(pool5)

    # Fully-connected layers 
    fc1 = tf.layers.dense(flatten, 4096, activation=tf.nn.relu, kernel_initializer=he_init, name='fc1')  
    fc2 = tf.layers.dense(fc1,     4096, activation=tf.nn.relu, kernel_initializer=he_init, name='fc2')
    out = tf.layers.dense(fc2,     NUM_OUTPUTS,  activation=None, name='logits')  
    return out 

def resnet_34(x, NUM_OUTPUTS = 1):
	he_init = tf.contrib.layers.variance_scaling_initializer()
	conv1 = tf.layers.conv2d(x,     64, [7, 7], strides=2, padding="SAME", activation=tf.nn.relu, kernel_initializer=he_init, name='Conv1')
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[3, 3], strides=2)



    conv2_1 = tf.layers.conv2d(pool1, 64, [3, 3], padding="SAME", activation=tf.nn.relu, kernel_initializer=he_init, name='Conv2-1')
    conv2_1 = tf.layers.conv2d(conv2, 64, [3, 3], padding="SAME", activation=tf.nn.relu, kernel_initializer=he_init, name='Conv2-1-2')

    conv2_2 = tf.layers.conv2d(conv2_1, 64, [3, 3], padding="SAME", activation=tf.nn.relu, kernel_initializer=he_init, name='Conv2-2')
    conv2_2 = tf.layers.conv2d(conv2_2, 64, [3, 3], padding="SAME", activation=tf.nn.relu, kernel_initializer=he_init, name='Conv2-2-2')

    conv2_3 = tf.layers.conv2d(conv2_2, 64, [3, 3], padding="SAME", activation=tf.nn.relu, kernel_initializer=he_init, name='Conv2-3')
    conv2_3 = tf.layers.conv2d(conv2_3, 64, [3, 3], padding="SAME", activation=tf.nn.relu, kernel_initializer=he_init, name='Conv2-3-2')



    conv3_1 = tf.layers.conv2d(conv2_3, 128, [3, 3], strides=2, padding="SAME", activation=tf.nn.relu, kernel_initializer=he_init, name='Conv3-1')
    conv3_1 = tf.layers.conv2d(conv3_1, 128, [3, 3], padding="SAME", activation=tf.nn.relu, kernel_initializer=he_init, name='Conv3-1-2')

    conv3_2 = tf.layers.conv2d(conv3_1, 128, [3, 3], padding="SAME", activation=tf.nn.relu, kernel_initializer=he_init, name='Conv3-2')
    conv3_2 = tf.layers.conv2d(conv3_2, 128, [3, 3], padding="SAME", activation=tf.nn.relu, kernel_initializer=he_init, name='Conv3-2-2')

    conv3_3 = tf.layers.conv2d(conv3_2, 128, [3, 3], padding="SAME", activation=tf.nn.relu, kernel_initializer=he_init, name='Conv3-3')
    conv3_3 = tf.layers.conv2d(conv3_3, 128, [3, 3], padding="SAME", activation=tf.nn.relu, kernel_initializer=he_init, name='Conv3-3-2')

    conv3_4 = tf.layers.conv2d(conv3_3, 128, [3, 3], padding="SAME", activation=tf.nn.relu, kernel_initializer=he_init, name='Conv3-4')
    conv3_4 = tf.layers.conv2d(conv3_4, 128, [3, 3], padding="SAME", activation=tf.nn.relu, kernel_initializer=he_init, name='Conv3-4-2')



    conv4_1 = tf.layers.conv2d(conv3_4, 256, [3, 3], strides=2, padding="SAME", activation=tf.nn.relu, kernel_initializer=he_init, name='Conv4-1')
    conv4_1 = tf.layers.conv2d(conv4_1, 256, [3, 3], padding="SAME", activation=tf.nn.relu, kernel_initializer=he_init, name='Conv4-1-2')

    conv4_2 = tf.layers.conv2d(conv4_1, 256, [3, 3], padding="SAME", activation=tf.nn.relu, kernel_initializer=he_init, name='Conv4-2')
    conv4_2 = tf.layers.conv2d(conv4_2, 256, [3, 3], padding="SAME", activation=tf.nn.relu, kernel_initializer=he_init, name='Conv4-2-2')

    conv4_3 = tf.layers.conv2d(conv4_2, 256, [3, 3], padding="SAME", activation=tf.nn.relu, kernel_initializer=he_init, name='Conv4-3')
    conv4_3 = tf.layers.conv2d(conv4_3, 256, [3, 3], padding="SAME", activation=tf.nn.relu, kernel_initializer=he_init, name='Conv4-3-2')

    conv4_4 = tf.layers.conv2d(conv4_3, 256, [3, 3], padding="SAME", activation=tf.nn.relu, kernel_initializer=he_init, name='Conv4-4')
    conv4_4 = tf.layers.conv2d(conv4_4, 256, [3, 3], padding="SAME", activation=tf.nn.relu, kernel_initializer=he_init, name='Conv4-4-2')

    conv4_5 = tf.layers.conv2d(conv4_4, 256, [3, 3], padding="SAME", activation=tf.nn.relu, kernel_initializer=he_init, name='Conv4-5')
    conv4_5 = tf.layers.conv2d(conv4_5, 256, [3, 3], padding="SAME", activation=tf.nn.relu, kernel_initializer=he_init, name='Conv4-5-2')

    conv4_6 = tf.layers.conv2d(conv4_5, 256, [3, 3], padding="SAME", activation=tf.nn.relu, kernel_initializer=he_init, name='Conv4-6')
    conv4_6 = tf.layers.conv2d(conv4_6, 256, [3, 3], padding="SAME", activation=tf.nn.relu, kernel_initializer=he_init, name='Conv4-6-2')



    conv5_1 = tf.layers.conv2d(conv4_6, 512, [3, 3], strides=2, padding="SAME", activation=tf.nn.relu, kernel_initializer=he_init, name='Conv5-1')
    conv5_1 = tf.layers.conv2d(conv5_1, 512, [3, 3], padding="SAME", activation=tf.nn.relu, kernel_initializer=he_init, name='Conv5-1-2')

    conv5_2 = tf.layers.conv2d(conv5_1, 512, [3, 3], padding="SAME", activation=tf.nn.relu, kernel_initializer=he_init, name='Conv5-2')
    conv5_2 = tf.layers.conv2d(conv5_2, 512, [3, 3], padding="SAME", activation=tf.nn.relu, kernel_initializer=he_init, name='Conv5-2-2')

    conv5_3 = tf.layers.conv2d(conv5_2, 512, [3, 3], padding="SAME", activation=tf.nn.relu, kernel_initializer=he_init, name='Conv5-3')
    conv5_3 = tf.layers.conv2d(conv5_3, 512, [3, 3], padding="SAME", activation=tf.nn.relu, kernel_initializer=he_init, name='Conv5-3-2')



    flatten = tf.contrib.layers.flatten(conv5_3) 

    fc1 = tf.layers.dense(flatten, 1024, activation=tf.nn.relu, kernel_initializer=he_init, name='fc1') 
    fc2 = tf.layers.dense(fc1,     1024, activation=tf.nn.relu, kernel_initializer=he_init, name='fc2') 
    out = tf.layers.dense(fc2,     NUM_OUTPUTS,  activation=None, name='logits') 
    return out

