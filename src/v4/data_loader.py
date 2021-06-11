
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import matplotlib.pyplot as plt
import math
import csv
import tensorflow as tf
from skimage import io, transform
from tqdm import trange, tqdm

class DataGenerator:

    def __init__(self,
                imagedir = '../../../data/hip_images_marta/',
                csvfilename = '../../../data/hip_images_marta/final_data.csv',
                width = 256,
                height = 256,
                channels = 1,
                ratio = 0.8, # training size / data size ratio,
                batch_size = 16, #128,
                useBinaryClassify = True,
                binaryThreshold = 60):

        self.imagedir = imagedir
        self.csvfilename = csvfilename
        self.index = 0
        self.WIDTH = width
        self.HEIGHT = height
        self.CHANNELS = channels
        self.ratio = ratio
        self.batch_size = batch_size

        self.useBinaryClassify = useBinaryClassify
        self.binaryThreshold = binaryThreshold

        print("Loading and formating image data ....")
        self.train_dataset, self.valid_dataset, self.test_dataset = self.generate()
        print("Loading and formating image data: Complete")

    def generate(self):
        angles, files = self.loadCSV()
        images = self.loadImages(files)
        
        # Generate classifcation data if needed 
        labels = angles
        if(self.useBinaryClassify):
            labels = self.threshold(labels)   
                 
        images, mean, std = self.StandardScaler(images) 

        (x_train, y_train), (x_test, y_test) = self.split_data(images, labels)
        self.train_size = x_train.shape[0]
        self.test_size = x_test.shape[0]
        print("Training data size: Input Data", x_train.shape, " Truth Data:", y_train.shape)
        print("Test data size: Input Data", x_test.shape, " Truth Data:", y_test.shape)

        train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(self.batch_size).shuffle(1000)
        train_dataset = train_dataset.map(lambda x, y: (tf.image.resize_with_pad(x, self.HEIGHT, self.WIDTH), y))  # upscale to prevent overfitting
        train_dataset = train_dataset.map(lambda x, y: (tf.image.grayscale_to_rgb(x), y)) if self.CHANNELS == 3 else train_dataset # Add Channels if needed
        train_dataset = train_dataset.repeat()

        valid_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(self.batch_size)
        valid_dataset = valid_dataset.map(lambda x, y: (tf.image.resize_with_pad(x, self.HEIGHT, self.WIDTH), y))  # upscale to prevent overfitting
        valid_dataset = valid_dataset.map(lambda x, y: (tf.image.grayscale_to_rgb(x), y)) if self.CHANNELS == 3 else train_dataset # Add Channels if needed
        valid_dataset = valid_dataset.repeat()

        test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(self.batch_size)
        test_dataset = test_dataset.map(lambda x, y: (tf.image.resize_with_pad(x, self.HEIGHT, self.WIDTH), y))
        test_dataset = test_dataset.map(lambda x, y: (tf.image.grayscale_to_rgb(x), y)) if self.CHANNELS == 3 else train_dataset # Add Channels if needed
        
        return train_dataset, valid_dataset, test_dataset

    '''
    Loads CSV and returns an array of angle data and an array of corresponding files 
    '''
    def loadCSV(self):
        angles_dict = {}

        with open(self.csvfilename) as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                filename = row['Match 1']

                if row['Alpha'] != '' and row['Alpha'] != 'cm':
                    alpha = float(row['Alpha'])
                    beta = float(row['Beta'])
                    angles_dict[filename] = [alpha, beta]
                
        all_files = os.listdir(self.imagedir)

        files = []
        angles = []
        for f in all_files:
            if f in angles_dict:
                angles.append(angles_dict[f])
                files.append(f)

        angles = np.array(angles)
        angles = np.reshape(angles[:,0], (-1, 1)) # Grab first angle

        return angles, files
        
    '''
    Load image data and returns an numpy array of image data 
    '''
    def loadImages(self, files):
        for f in tqdm(files):
            img = io.imread(self.imagedir + f)
            img = np.reshape(img, (img.shape[0], img.shape[1], 1))
            
            if f == files[0]:
                imgs = img[None,:]
            else:
                imgs = np.concatenate((imgs, img[None,:]), axis=0)
        
        #print(imgs.shape)
        #imgs = imgs[:, :, :, None] * np.ones(3, dtype=int)[None, None, None, :]
        #print(imgs.shape)

        return imgs

    def StandardScaler(self, data):
        mean = np.mean(data)
        std = np.std(data)
        return (data - mean) / std, mean, std

    def StandardScalerByImage(self, data):
        mean = np.mean(data)
        std = np.std(data)
        return (data - mean) / std, mean, std

    '''
    Generates a onehot encoded label based on angle thresholding. Encoding: [data < threshod, data > threshold]
    '''
    def threshold(self, data): 
        threshold = (data > self.binaryThreshold) * 1
        onehot = np.concatenate( (1 - threshold, threshold), axis = 1)  # OneHot Encoding
        return onehot

    '''
    Split data into training and testing data sets 
    '''
    def split_data(self, images, labels):

        # Split data into test/training sets 
        index = int(self.ratio * len(images)) # Split index
        x_train = images[0:index, :]
        x_test = images[index:, :] 

        if self.useBinaryClassify:
            y_train = labels[0:index, :]
            y_test = labels[index:, :]
        else:
            y_train = labels[0:index]
            y_test = labels[index:]

        return (x_train, y_train), (x_test, y_test)


if __name__ == '__main__':
    data = DataGenerator()