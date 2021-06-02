import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import time
import numpy as np 
import tensorflow as tf
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import LearningRateScheduler, ReduceLROnPlateau
import matplotlib.pyplot as plt
from .data_loader import DataGenerator

# Training Parameters
epochs = 50
batch_size = 16 

# Network Parameters
WIDTH = 256
HEIGHT = 256
CHANNELS = 1
NUM_OUTPUTS = 2

def Train():
    imagedir = '../data/hip_images_marta/'
    csvfilename = '../data/hip_images_marta/final_data.csv'

    # Generate datasets 
    data = DataGenerator(width=WIDTH, height=HEIGHT, imagedir=imagedir, csvfilename=csvfilename)
    dataset = (data.train_dataset, data.valid_dataset, data.test_dataset)
    TrainAll(dataset)

def TrainAll(dataset):
    from .models import get_model_names
    for model_name in get_model_names():
        print("Running ... " + model_name)

        # Train model 
        start = time.time()
        model, history, metrics = TrainOne(model_name, dataset)
        end = time.time()

        # Plot and Output results
        summary = 'Loss: %.4f, Accuracy: %.4f, Time: %.2fs' % (metrics[0], metrics[1], (end - start))
        print(model_name + ': ' + summary)
        plot([history], model_name, summary)

        # Save metric history 
        metrics = { 'loss': metrics[0], 'acc': metrics[1], 'time': (end - start) }
        load_and_save(model_name, history, metrics)


def TrainOne(model_name, dataset):
    (train_dataset, valid_dataset, test_dataset) = dataset

    earlyStopping = keras.callbacks.EarlyStopping(monitor='val_acc', verbose=1, patience=5)

    from .models import transfer_learned_model 
    model = transfer_learned_model(model_name, HEIGHT, WIDTH, CHANNELS, NUM_OUTPUTS)
    model.compile(optimizer=keras.optimizers.Adam(),
                loss='sparse_categorical_crossentropy',
                metrics=['acc'])
    model.summary()
    
    # Train and Evaluate model
    history = model.fit(train_dataset, epochs=epochs, steps_per_epoch=200,
            validation_data=valid_dataset,
            validation_steps=20,
            callbacks=[earlyStopping])

    # Fine-Tune Model 
    from .models import fine_tune_model
    model = fine_tune_model(model_name, model)
    history_finetune = model.fit(train_dataset, epochs=epochs, steps_per_epoch=200,
            validation_data=valid_dataset,
            validation_steps=20,
            callbacks=[earlyStopping])
    
    # Add finetune history 
    history.history['acc'] += history_finetune.history['acc']
    history.history['val_acc'] += history_finetune.history['val_acc']
    history.history['loss'] += history_finetune.history['loss']
    history.history['val_loss'] += history_finetune.history['val_loss']

    # Evaluate Test Data 
    loss_and_metrics = model.evaluate(test_dataset)

    return model, history, loss_and_metrics

'''
Plots metric history data for training a model 
'''
def plot(data, model_name, summary):
    # Plot Accuracy / Loss 
    fig, axs = plt.subplots(2)
    fig.suptitle(model_name + ': ' + summary)

    axs[0].plot(data[0].history['acc'])
    axs[0].plot(data[0].history['val_acc'])
    axs[0].set_ylabel('acc')
    axs[0].legend(["Train", "Test"], loc="lower right")

    axs[1].plot(data[0].history['loss'])
    axs[1].plot(data[0].history['val_loss'])
    axs[1].set_ylabel('loss')
    axs[1].legend(["Train", "Test"], loc="upper right")
    
    #plt.show()
    plt.savefig('cifar_' + model_name +'.png')


'''
Saves training and test metrics to metrics.npy
'''
def load_and_save(model_name, history, metrics):
    fname = 'metrics.npy' 
    fileExists = os.path.isfile(fname)
    if(fileExists):
        data = np.load(fname, allow_pickle=True)[()]
        data[model_name] = { 'history': history.history, 'metrics': metrics }
    else:
        data = { model_name: { 'history': history.history, 'metrics': metrics } }
    
    np.save(fname, data)