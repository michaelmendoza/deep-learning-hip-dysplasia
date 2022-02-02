
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

'''
https://medium.com/swlh/hands-on-the-cifar-10-dataset-with-transfer-learning-2e768fd6c318
'''

def resnet2(HEIGHT, WIDTH, CHANNELS, NUM_OUTPUTS):

    he_init = keras.initializers.he_normal(seed=None)

    def res_net_block(input_data, filters, kernel_size, strides=1):
        x = layers.Conv2D(filters, kernel_size, strides=strides, padding='same', activation='relu', kernel_initializer=he_init)(input_data)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(filters, kernel_size, padding='same', activation=None, kernel_initializer=he_init)(x)
        x = layers.BatchNormalization()(x)
        
        if(strides == 2):  # add linear projection residual shortcut connection to match changed dims
            input_data = layers.Conv2D(filters, 1, strides=strides, padding='same', activation=None, kernel_initializer=he_init)(input_data)
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
    outputs = layers.Dense(NUM_OUTPUTS, activation='softmax')(x)

    model = keras.Model(inputs, outputs)

    layer_count = sum([model.layers[i].name.count('conv') for i in range(len(model.layers))])
    print("Number of conv layers in the model: ", layer_count)
    return model

def resnet50_keras(HEIGHT, WIDTH, CHANNELS, NUM_OUTPUTS):
    Network = tf.keras.applications.ResNet50V2
    base_model = Network(weights = None, 
                        include_top = False, 
                        input_shape = (HEIGHT, WIDTH, CHANNELS),
                        pooling='max')
    
    for layer in base_model.layers:
        layer.trainable = True
    x = layers.Flatten()(base_model.output)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(NUM_OUTPUTS, activation='softmax')(x)
    model = keras.Model(inputs = base_model.input, outputs = x)
    model.base_model = base_model # Save ref to base_model 
        
    layer_count = sum([model.layers[i].name.count('conv2d') for i in range(len(model.layers))])
    print("Number of conv2d layers in the model: ", layer_count)
    print("Number of layers in the base model: ", len(base_model.layers))
    return model

def get_model_names():
    return ["VGG16", "VGG19", "ResNet50V2", "ResNet152V2", "Xception", "InceptionResNetV2", "DenseNet121", "DenseNet169"] #, "EfficientNetB7"]

def transfer_learned_model(model_name, HEIGHT, WIDTH, CHANNELS, NUM_OUTPUTS):

    MODELS = {
        "VGG16": tf.keras.applications.VGG16,
        "VGG19": tf.keras.applications.VGG19,
        "ResNet50V2": tf.keras.applications.ResNet50V2,
        "ResNet152V2": tf.keras.applications.ResNet152V2,
        "Xception": tf.keras.applications.Xception, 
        "InceptionResNetV2": tf.keras.applications.InceptionResNetV2,
        "DenseNet121": tf.keras.applications.DenseNet121,
        "DenseNet169": tf.keras.applications.DenseNet169,
        #"EfficientNetB7": tf.keras.applications.EfficientNetB7
    }
    Network = MODELS[model_name]

    base_model = Network(weights = 'imagenet', 
                        include_top = False, 
                        input_shape = (HEIGHT, WIDTH, CHANNELS),
                        pooling='max')

    for layer in base_model.layers:
        layer.trainable = False
    x = layers.Flatten()(base_model.output)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(NUM_OUTPUTS, activation='softmax')(x)
    model = keras.Model(inputs = base_model.input, outputs = x)
    model.base_model = base_model # Save ref to base_model 
        
    print("Number of layers in the base model: ", len(base_model.layers))
    return model

def fine_tune_model(model_name, model):
    
    MODELS = {
        "ResNet50V2": 25,
        "ResNet152V2": 50,
        "Xception":  25, 
        "InceptionResNetV2": 50,
        "DenseNet121":  40,
        "DenseNet169": 60,
        #"EfficientNetB7":  6,
    }

    base_model = model.base_model
    base_model.trainable = True

    # Let's take a look to see how many layers are in the base model
    print("Number of layers in the base model: ", len(base_model.layers))

    # Fine-tune from this layer onwards
    fine_tune_at =  len(base_model.layers) - MODELS[model_name]
    print("FineTune at ", fine_tune_at)

    # Freeze all the layers before the `fine_tune_at` layer
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable =  False

    model.compile(optimizer=keras.optimizers.SGD(lr=0.0001, momentum=0.9),   
                 loss='categorical_crossentropy', metrics=['acc']) 
    return model 