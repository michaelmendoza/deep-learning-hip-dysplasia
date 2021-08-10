
#Several CNN structures are contained withtin this file, currently the ResNet2 network is the one used. As refered to in the report, this corresponds to Residual Block B

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import load_model, Model

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

#This is the ResNet model architecture trained by Marta.
def resnet2(HEIGHT, WIDTH, CHANNELS, NUM_OUTPUTS):

    he_init = keras.initializers.he_normal(seed=None)

    def res_net_block(input_data, filters, kernel_size, strides=1):     #this is the structure of the residual block, it contains 2 convolutional layers
        x = layers.Conv2D(filters, kernel_size, strides=strides, padding='same', activation='relu', kernel_initializer=he_init)(input_data) #first layer does have its activation
        x = layers.BatchNormalization()(x)      #batch normalisation after each conv layer
        x = layers.Conv2D(filters, kernel_size, padding='same', activation=None, kernel_initializer=he_init)(x) #no activation, if residual block A is wanted, the activation here should be changed to relu
        x = layers.BatchNormalization()(x)

        if(strides == 2):  # add linear projection residual shortcut connection to match changed dims
            input_data = layers.Conv2D(filters, 1, strides=strides, padding='same', activation=None, kernel_initializer=he_init)(input_data)
        x = layers.Add()([x, input_data])       #add the input with the current transformation the residual block has done
        x = layers.Activation('relu')(x)        #do the activation after the addition for residual block B; if doing residual block A this line can be deleted
        return x

    inputs = keras.Input(shape=(HEIGHT, WIDTH, CHANNELS))
    x = layers.Conv2D(32, 3, padding='same', activation='relu', kernel_initializer=he_init)(inputs)     #this is the code for the initial layers
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
    x = layers.Dense(1000, activation='relu', kernel_initializer=he_init)(x)        #1000 fully connected layer at the end
    outputs = layers.Dense(NUM_OUTPUTS, activation='sigmoid')(x)        #last layer with sigmoid activatio since binary, corresponds to either a 1 or 0

    model = keras.Model(inputs, outputs)
    return model

#This is the model for the auto-cropping network, with input being the scan and
#the outputs being the x and y coordinates of the bounding box. The network is a ResNet
def resnet2Autocrop(HEIGHT, WIDTH, CHANNELS):

    #The He initializer samples from a truncated normal distribution centred around 0 with a std (2/...)
    he_init = keras.initializers.he_normal(seed=None)

    #Below is a function definition for a resnet block
    def res_net_block(input_data, filters, kernel_size, strides=1):     #this is the structure of the residual block, it contains 2 convolutional layers
        x = layers.Conv2D(filters, kernel_size, strides=strides, padding='same', activation='relu', kernel_initializer=he_init)(input_data) #first layer does have its activation
        x = layers.BatchNormalization()(x)      #batch normalisation after each conv layer
        x = layers.Conv2D(filters, kernel_size, padding='same', activation=None, kernel_initializer=he_init)(x) #no activation, if residual block A is wanted, the activation here should be changed to relu
        x = layers.BatchNormalization()(x)

        if(strides == 2):  # add linear projection residual shortcut connection to match changed dims
            input_data = layers.Conv2D(filters, 1, strides=strides, padding='same', activation=None, kernel_initializer=he_init)(input_data)
        x = layers.Add()([x, input_data])       #add the input with the current transformation the residual block has done
        x = layers.Activation('relu')(x)        #do the activation after the addition for residual block B; if doing residual block A this line can be deleted
        return x

    inputs = keras.Input(shape=(HEIGHT, WIDTH, CHANNELS), name = "scan")

    #specifies what the input is - the scans
    x = layers.Conv2D(32, 3, padding='same', activation='relu', kernel_initializer=he_init)(inputs)     #this is the code for the initial layers
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
    x = layers.Dense(1000, activation = 'relu', kernel_initializer = he_init)(x)        #1000 fully connected layer at the end
    x = layers.Dense(1000, activation = 'relu', kernel_initializer = he_init)(x)
    x = layers.Dense(1000, activation = 'relu', kernel_initializer = he_init)(x)
    x = layers.Dense(1000, activation = 'relu', kernel_initializer = he_init)(x)

    #Output layers
    x_coordinate = layers.Dense(1, name = "x_coordinate")(x)  #predicts the x coordinate
    y_coordinate = layers.Dense(1, name = "y_coordinate")(x)  #predicts the y coordinate
    model = keras.Model({"scan" : inputs},  {"x_coordinate": x_coordinate, "y_coordinate": y_coordinate})

    return model

#This is the first network architecture for the patient details + scans network
#without dense layers being placed specifically for the patient details
#Note: here, you are retraining Marta's model with the cropped scans as well, and not using the pretrained network
def PDnet1(HEIGHT, WIDTH, CHANNELS, NUM_OUTPUTS, NUM_SIDE, NUM_GENDER, NUM_INDICATION):

    #The He initializer samples from a truncated normal distribution centred around 0 with a std (2/...)
    he_init = keras.initializers.he_normal(seed=None)

    #Below is a function definition for a resnet block
    def res_net_block(input_data, filters, kernel_size, strides=1):     #this is the structure of the residual block, it contains 2 convolutional layers
        x = layers.Conv2D(filters, kernel_size, strides=strides, padding='same', activation='relu', kernel_initializer=he_init)(input_data) #first layer does have its activation
        x = layers.BatchNormalization()(x)      #batch normalisation after each conv layer
        x = layers.Conv2D(filters, kernel_size, padding='same', activation=None, kernel_initializer=he_init)(x) #no activation, if residual block A is wanted, the activation here should be changed to relu
        x = layers.BatchNormalization()(x)

        if(strides == 2):  # add linear projection residual shortcut connection to match changed dims
            input_data = layers.Conv2D(filters, 1, strides=strides, padding='same', activation=None, kernel_initializer=he_init)(input_data)
        x = layers.Add()([x, input_data])       #add the input with the current transformation the residual block has done
        x = layers.Activation('relu')(x)        #do the activation after the addition for residual block B; if doing residual block A this line can be deleted
        return x

    inputs = keras.Input(shape=(HEIGHT, WIDTH, CHANNELS), name = "scan")
    inputGender = keras.Input(shape=(NUM_GENDER), name = "gender")
    inputSide = keras.Input(shape=(NUM_SIDE), name = "side")
    inputIndication = keras.Input(shape=(NUM_INDICATION), name = "indication")
    inputBirthweight = keras.Input(shape=(1), name = "birthweight")

    #specifies what the input is - the scans
    x = layers.Conv2D(32, 3, padding='same', activation='relu', kernel_initializer=he_init)(inputs)     #this is the code for the initial layers
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
    x = layers.Dense(1000, activation='relu', kernel_initializer=he_init)(x)        #1000 fully connected layer at the end

    #At this point, add in the extra patient details
    z_patient = layers.Concatenate(axis = 1)([x, inputBirthweight, inputSide, inputGender, inputIndication])

    #run z_patient through 3 fully connected 1000 neuron layers with relu activation
    x = layers.Dense(1000, activation = 'relu', kernel_initializer = he_init)(z_patient)

    for layer in range(2):
        x = layers.Dense(1000, activation = 'relu', kernel_initializer = he_init)(x)

    #Output layer
    outputs = layers.Dense(NUM_OUTPUTS, activation = 'sigmoid')(x)  #last layer with sigmoid activatio since binary, corresponds to either a 1 or 0
    model = keras.Model({"scan" : inputs, "gender" :inputGender, \
     "side" : inputSide, "indication" : inputIndication, "birthweight" : inputBirthweight},  outputs) #[inputs, inputGender, inputSide, inputIndication, inputBirthweight, inputAlpha, inputBeta]

    return model

#This is the second network architecture for the patient details + scans network
#with dense layers being placed specifically for the patient details
#Note: here, you are retraining Marta's model with the cropped scans as well, and not using the pretrained network
def PDnet2(HEIGHT, WIDTH, CHANNELS, NUM_OUTPUTS, NUM_SIDE, NUM_GENDER, NUM_INDICATION):

    #The He initializer samples from a truncated normal distribution centred around 0 with a std (2/...)
    he_init = keras.initializers.he_normal(seed=None)

    #Below is a function definition for a resnet block
    def res_net_block(input_data, filters, kernel_size, strides=1):     #this is the structure of the residual block, it contains 2 convolutional layers
        x = layers.Conv2D(filters, kernel_size, strides=strides, padding='same', activation='relu', kernel_initializer=he_init)(input_data) #first layer does have its activation
        x = layers.BatchNormalization()(x)      #batch normalisation after each conv layer
        x = layers.Conv2D(filters, kernel_size, padding='same', activation=None, kernel_initializer=he_init)(x) #no activation, if residual block A is wanted, the activation here should be changed to relu
        x = layers.BatchNormalization()(x)

        if(strides == 2):  # add linear projection residual shortcut connection to match changed dims
            input_data = layers.Conv2D(filters, 1, strides=strides, padding='same', activation=None, kernel_initializer=he_init)(input_data)
        x = layers.Add()([x, input_data])       #add the input with the current transformation the residual block has done
        x = layers.Activation('relu')(x)        #do the activation after the addition for residual block B; if doing residual block A this line can be deleted
        return x

    inputs = keras.Input(shape=(HEIGHT, WIDTH, CHANNELS), name = "scan")
    inputGender = keras.Input(shape=(NUM_GENDER), name = "gender")
    inputSide = keras.Input(shape=(NUM_SIDE), name = "side")
    inputIndication = keras.Input(shape=(NUM_INDICATION), name = "indication")
    inputBirthweight = keras.Input(shape=(1), name = "birthweight")

    #specifies what the input is - the scans
    x = layers.Conv2D(32, 3, padding='same', activation='relu', kernel_initializer=he_init)(inputs)     #this is the code for the initial layers
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
    x = layers.Dense(1000, activation='relu', kernel_initializer=he_init)(x)        #1000 fully connected layer at the end

    y = layers.Concatenate(axis = 1)([inputBirthweight, inputSide, inputGender, inputIndication])
    y = layers.Dense(100, activation = 'relu', kernel_initializer = he_init)(y)
    y = layers.Dense(100, activation = 'relu', kernel_initializer = he_init)(y)
    y = layers.Dense(100, activation = 'relu', kernel_initializer = he_init)(y)
    y = layers.Dense(100, activation = 'relu', kernel_initializer = he_init)(y)

    #At this point, add in the extra patient details
    z_patient = layers.Concatenate(axis = 1)([x, y])

    #run z_patient through 3 fully connected 1000 neuron layers with relu activation
    x = layers.Dense(1000, activation = 'relu', kernel_initializer = he_init)(z_patient)

    for layer in range(2):
        x = layers.Dense(1000, activation = 'relu', kernel_initializer = he_init)(x)

    #Output layer
    outputs = layers.Dense(NUM_OUTPUTS, activation = 'sigmoid')(x)  #last layer with sigmoid activatio since binary, corresponds to either a 1 or 0
    model = keras.Model({"scan" : inputs, "gender" :inputGender, \
     "side" : inputSide, "indication" : inputIndication, "birthweight" : inputBirthweight},  outputs) #[inputs, inputGender, inputSide, inputIndication, inputBirthweight, inputAlpha, inputBeta]

    return model

#This is the second network architecture for the patient details + scans network
#with dense layers being placed specifically for the patient details
#Note: here, you are using Marta's pre-trained model without the cropped scans
def PDnet3(NUM_FEATURES, NUM_OUTPUTS, NUM_SIDE, NUM_GENDER, NUM_INDICATION):
    he_init = keras.initializers.he_normal(seed=None)

    #Input with the numerical and categorical data
    alphaPred = keras.Input(shape=(NUM_FEATURES), name = "outcome_pred") #This is the input of features calculated from Marta's network
    #inputAlpha = keras.Input(shape=(1), name = "alpha")
    #inputBeta = keras.Input(shape=(1), name = "beta")
    inputBirthweight = keras.Input(shape=(1), name = "birthweight")
    inputSide = keras.Input(shape=(NUM_SIDE), name = "side")
    inputGender = keras.Input(shape=(NUM_GENDER), name = "gender")
    inputIndication = keras.Input(shape=(NUM_INDICATION), name = "indication")

    #At this point, add in the extra patient details
    z_patient = layers.Concatenate(axis = 1)([alphaPred, inputBirthweight, inputSide, inputGender, inputIndication])

    #run z_patient through 3 fully connected 1000 neuron layers with relu activation
    x = layers.Dense(1000, activation = 'relu', kernel_initializer = he_init)(z_patient)

    for layer in range(2):
        x = layers.Dense(1000, activation = 'relu', kernel_initializer = he_init)(x)

    #Output layer
    outputs = layers.Dense(NUM_OUTPUTS, activation = 'sigmoid')(x)  #last layer with sigmoid activatio since binary, corresponds to either a 1 or 0
    model = keras.Model({"outcome_pred" : alphaPred, "gender" :inputGender, \
     "side" : inputSide, "indication" : inputIndication, "birthweight" : inputBirthweight},  outputs) #[inputs, inputGender, inputSide, inputIndication, inputBirthweight, inputAlpha, inputBeta]

    return model

def ensembleNet():
    he_init = keras.initializers.he_normal(seed=None)

    #The inputs for this network are the predictions gotten from the resnet trained with cropped images and the trained PDNet1
    inputScan = keras.Input(shape = (1), name = 'scansOnlyModel')
    inputPDScan = keras.Input(shape = (1), name = 'scansPDModel')

    x = layers.Concatenate(axis = 1)([inputScan, inputPDScan])
    output = layers.Dense(1, activation = 'sigmoid')(x)

    model = keras.Model({"scansOnlyModel" : inputScan, "scansPDModel" :inputPDScan}, output)

    return model
