
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def conv0(HEIGHT, WIDTH, CHANNELS, NUM_OUTPUTS):
    inputs = keras.Input(shape=(HEIGHT, WIDTH, CHANNELS))
    x = layers.Conv2D(32, 3, padding="same", activation='relu')(inputs)
    x = layers.Conv2D(64, 3, padding="same", activation='relu')(x)
    x = layers.MaxPooling2D(padding="same", strides=(2, 2), pool_size=(2, 2))(x)

    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dense(128, activation='relu')(x)
    outputs = layers.Dense(NUM_OUTPUTS, activation=tf.nn.softmax)(x)
    
    model = keras.Model(inputs, outputs)
    return model

def conv1(HEIGHT, WIDTH, CHANNELS, NUM_OUTPUTS):
    return tf.keras.models.Sequential([
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
        tf.keras.layers.Dense(2048, activation=tf.nn.relu),
        tf.keras.layers.Dense(NUM_OUTPUTS, activation=tf.nn.softmax)
    ])

def conv2(HEIGHT, WIDTH, CHANNELS, NUM_OUTPUTS):
    return tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(64, (3, 3), padding="same", activation=tf.nn.relu, input_shape=(HEIGHT, WIDTH, CHANNELS)),
        tf.keras.layers.Conv2D(64, (3, 3), padding="same", activation=tf.nn.relu),
        tf.keras.layers.MaxPooling2D(padding="same", strides=(2, 2), pool_size=(2, 2)),

        tf.keras.layers.Conv2D(128, (3, 3), padding="same", activation=tf.nn.relu),
        tf.keras.layers.Conv2D(128, (3, 3), padding="same", activation=tf.nn.relu),
        tf.keras.layers.MaxPooling2D(padding="same", strides=(2, 2), pool_size=(2, 2)),

        tf.keras.layers.Conv2D(256, (3, 3), padding="same", activation=tf.nn.relu),
        tf.keras.layers.Conv2D(256, (3, 3), padding="same", activation=tf.nn.relu),
        tf.keras.layers.MaxPooling2D(padding="same", strides=(2, 2), pool_size=(2, 2)),

        tf.keras.layers.Conv2D(512, (3, 3), padding="same", activation=tf.nn.relu),
        tf.keras.layers.Conv2D(512, (3, 3), padding="same", activation=tf.nn.relu),
        tf.keras.layers.MaxPooling2D(padding="same", strides=(2, 2), pool_size=(2, 2)),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(4096, activation=tf.nn.relu),
        tf.keras.layers.Dense(4096, activation=tf.nn.relu),
        tf.keras.layers.Dense(NUM_OUTPUTS, activation=tf.nn.softmax)
    ])

def conv3(HEIGHT, WIDTH, CHANNELS, NUM_OUTPUTS):
    return tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(64, (3, 3), padding="same", activation=tf.nn.relu, input_shape=(HEIGHT, WIDTH, CHANNELS)),
        tf.keras.layers.Conv2D(64, (3, 3), padding="same", activation=tf.nn.relu),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(padding="same", strides=(2, 2), pool_size=(2, 2)),

        tf.keras.layers.Conv2D(128, (3, 3), padding="same", activation=tf.nn.relu),
        tf.keras.layers.Conv2D(128, (3, 3), padding="same", activation=tf.nn.relu),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(padding="same", strides=(2, 2), pool_size=(2, 2)),

        tf.keras.layers.Conv2D(256, (3, 3), padding="same", activation=tf.nn.relu),
        tf.keras.layers.Conv2D(256, (3, 3), padding="same", activation=tf.nn.relu),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(padding="same", strides=(2, 2), pool_size=(2, 2)),

        tf.keras.layers.Conv2D(512, (3, 3), padding="same", activation=tf.nn.relu),
        tf.keras.layers.Conv2D(512, (3, 3), padding="same", activation=tf.nn.relu),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(padding="same", strides=(2, 2), pool_size=(2, 2)),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(4096, activation=tf.nn.relu),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(4096, activation=tf.nn.relu),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(NUM_OUTPUTS, activation=tf.nn.softmax)
    ])

def resnet(HEIGHT, WIDTH, CHANNELS, NUM_OUTPUTS):

    def res_net_block(input_data, filters, conv_size):
        x = layers.Conv2D(filters, conv_size, activation='relu', padding='same')(input_data)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(filters, conv_size, activation=None, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Add()([x, input_data])
        x = layers.Activation('relu')(x)
        return x

    inputs = keras.Input(shape=(HEIGHT, WIDTH, CHANNELS))
    x = layers.Conv2D(32, 3, activation='relu')(inputs)
    x = layers.Conv2D(64, 3, activation='relu')(x)
    x = layers.MaxPooling2D(3)(x)

    num_res_net_blocks = 10
    for i in range(num_res_net_blocks):
        x = res_net_block(x, 64, 3)

    x = layers.Conv2D(64, 3, activation='relu')(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(NUM_OUTPUTS, activation='softmax')(x)

    model = keras.Model(inputs, outputs)
    return model

def resnet2(HEIGHT, WIDTH, CHANNELS, NUM_OUTPUTS):

    he_init = keras.initializers.he_normal(seed=None)

    def res_net_block(input_data, filters, kernel_size, strides=1):
        x = layers.Conv2D(filters, kernel_size, strides=strides, padding='same', activation='relu', kernel_initializer=he_init)(input_data)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(filters, kernel_size, padding='same', activation='relu', kernel_initializer=he_init)(x)
        x = layers.BatchNormalization()(x)
        
        if(strides == 2):  # add linear projection residual shortcut connection to match changed dims
            input_data = layers.Conv2D(filters, 1, strides=strides, padding='same', activation='relu', kernel_initializer=he_init)(input_data)
        x = layers.Add()([x, input_data])
        x = layers.Activation('relu')(x)
        return x

    inputs = keras.Input(shape=(HEIGHT, WIDTH, CHANNELS))
    x = layers.Conv2D(32, 3, padding='same', activation='relu', kernel_initializer=he_init)(inputs)
    x = layers.Conv2D(64, 3, padding='same', activation='relu', kernel_initializer=he_init)(x)
    x = layers.MaxPooling2D(3)(x)

    num_filters = 64
    for stage in range(3):
        if(stage == 0):
            x = res_net_block(x, num_filters, 3)
        else:
            x = res_net_block(x, num_filters, 3, strides=2)
        x = res_net_block(x, num_filters, 3)
        x = res_net_block(x, num_filters, 3)
        x = res_net_block(x, num_filters, 3)
        x = res_net_block(x, num_filters, 3)
        num_filters *= 2

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(1000, activation='relu', kernel_initializer=he_init)(x)
    outputs = layers.Dense(NUM_OUTPUTS, activation='sigmoid')(x)

    model = keras.Model(inputs, outputs)
    return model
