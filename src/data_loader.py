
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

class Stats:
    def __init__(self, min, max, mean, std):
        self.min = min
        self.max = max
        self.mean = mean
        self.std = std

class DataGenerator:

    def __init__(self):
        self.useNormalization = False 
        self.useWhitening = True
        self.useRandomOrder = True

        self.ratio = 0.8
        self.WIDTH = 196 #128
        self.HEIGHT = 196 #128
        self.CHANNELS = 1

        self.imagedir = 'data/images/'  # 'data/images-small_set/'
        self.anglecsv = './data/FinalLinkedData.csv'

        print("Loading and formating image data ....")
        self.generate()
        print("Loading and formating image data: Complete")
        print("Data size: Input Data", self.x_train.shape, " Truth Data:", self.y_train.shape);

    def loadCSV(self):
        angle_dict = {}
        with open(self.anglecsv) as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                key = row['Current Standardized Name']
                if row['Alpha'] != '' and row['Alpha'] != 'cm':
                    alpha = float(row['Alpha'])
                    beta = float(row['Beta'])
                    angle_dict[key] = [alpha, beta]
        return angle_dict 

    def formatAngleData(self):
        files = os.listdir(self.imagedir)

        files_with_angle = []
        angle_data = []
        for f in files:
            if f in self.angle_dict:
                angle_data.append(self.angle_dict[f])
                files_with_angle.append(f)

        return np.array(angle_data), files_with_angle
        
    def loadImageData(self):
        files = self.files

        for f in tqdm(files):
            img = io.imread(self.imagedir + f)
            img = transform.resize(img, (self.HEIGHT, self.WIDTH, COLOR_CHANNELS), mode='constant')
            img = img[:,:,1] 

            if f == files[0]:
                imgs = img[None,:]
            else:
                imgs = np.concatenate( (imgs, img[None,:]), axis=0)

        imgs = np.reshape( imgs, (-1, self.HEIGHT, self.WIDTH, self.CHANNELS))
        return imgs

    def generate(self):
        self.angle_dict = self.loadCSV()
        self.angle_data, self.files = self.formatAngleData() 
        
        # Load image data
        self.image_data = self.loadImageData()

        # Grab angle data:
        self.angle_data = np.reshape(self.angle_data[:,0], (-1, 1))

        # Randomize data order
        if self.useRandomOrder:
            indices = [_ for _ in range(len(self.angle_data))]
            self.image_data = self.image_data[indices]
            self.angle_data = self.angle_data[indices]

        # Data preprocessing
        if self.useNormalization:
            self.image_data, self.img_min, self.img_max = self.normalize(self.image_data)
            self.angle_data, self.ang_min, self.ang_max = self.normalize(self.angle_data)

        if self.useWhitening:
            self.image_data, self.img_mean, self.img_std = self.whiten(self.image_data)
            self.angle_data, self.ang_mean, self.ang_std = self.whiten(self.angle_data)

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

    def normalize(self, data):
        max = np.max(data)
        min = np.min(data)
        return (data - min) / (max - min), min, max

    def whiten(self, data):
        mean = np.mean(data)
        std = np.std(data)
        return (data - mean) / std, mean, std

    def denormalize(self, data, min, max):
        return data * (max - min) + min

    def undo_whitening(self, data, mean, std):
        return data * std + mean

    def undo_whitening_and_denormalize(self, data, stats):
        return (data * stats.std + stats.mean) * (stats.max - stats.min) + stats.min

    def plot(self, index):
        plt.imshow(self.image_data[index,:,:,0])
        plt.show()

if __name__ == '__main__':
    data = DataGenerator()
    xs, ys = data.next_batch(16)
    print(ys)

    plt.figure(1)
    data.plot(1)

    plt.figure(2)
    plt.hist(data.y_train)
    
    