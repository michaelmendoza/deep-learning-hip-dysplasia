

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
import math
import csv

import os
from skimage import io, transform
from tqdm import trange, tqdm

# Constants
COLOR_CHANNELS = 3

class DataGenerator:

    def __init__(self):
        self.ratio = 0.8
        self.WIDTH = 196
        self.HEIGHT = 196
        self.CHANNELS = 1

        print("Loading and formating image data ....")
        self.generate()
        print("Loading and formating image data: Complete")
        print("Data size: Input Data", self.x_train.shape, " Truth Data:", self.y_train.shape);

    def loadCSV(self):
        angle_dict = {}
        with open('./data/FinalLinkedData.csv') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                key = row['Current Standardized Name']
                angle_dict[key] = [row['Alpha'], row['Beta']]
        return angle_dict 

    def formatAngleData(self):
        fdir = 'data/images/'
        files = os.listdir(fdir)

        angle_data = []
        for f in files:
            angle_data.append(self.angle_dict[f])

        return np.array(angle_data)

    def loadImageData(self):
        fdir = 'data/images/'
        files = os.listdir(fdir)

        for f in tqdm(files):
            img = transform.resize(io.imread(fdir + f), (self.HEIGHT, self.WIDTH, COLOR_CHANNELS), mode='constant')
            img = img[:,:,1]

            if f == files[0]:
                imgs = img[None,:]
            else:
                imgs = np.concatenate( (imgs, img[None,:]), axis=0)

        imgs = np.reshape( imgs, (-1, self.HEIGHT, self.WIDTH, self.CHANNELS))
        return imgs

    def generate(self):
        self.angle_dict = self.loadCSV()
        self.image_data = self.loadImageData()
        self.angle_data = self.formatAngleData() #np.zeros(self.image_data.shape[0])

        # Grab alpha values:
        self.angle_data = np.reshape(self.angle_data[:,0], (-1, 1))

        # Split data into test/training sets
        index = int(self.ratio * len(self.image_data)) # Split index
        self.x_train = self.image_data[0:index, :]
        self.y_train = self.angle_data[0:index]
        self.x_test = self.image_data[index:,:]
        self.y_test = self.angle_data[index:]

    def next_batch(self, batch_size):
        length = self.x_train.shape[0]
        indices = np.random.randint(0, length, batch_size) # Grab batch_size values randomly
        return [self.x_train[indices], self.y_train[indices]]

if __name__ == '__main__':
    data = DataGenerator()

