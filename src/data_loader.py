
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

    def __init__(self, 
        imagedir = 'hip_images_marta/', #'data2/cropped/', #'data/images/', 
        anglecsv =  'hip_images_marta/final_data.csv', #'./data2/final_data.csv', #'./data/FinalLinkedData.csv', 
        width = 128, #196, 
        height = 128, #196, 
        ratio1 = 0.8, 
        ratio2 = 0.1,
        useBinaryClassify = True, 
        binaryThreshold = "Discharged", #60.0,
        useNormalization = True,       #useNormalization = False
        useWhitening = True, 
        useRandomOrder = False):        #useRandomOrder=True):
        
        self.imagedir = imagedir
        self.anglecsv = anglecsv
        self.WIDTH = width
        self.HEIGHT = height
        self.CHANNELS = 1
        self.ratio1 = ratio1
        self.ratio2 = ratio2
        
        self.useBinaryClassify = useBinaryClassify
        self.binaryThreshold = binaryThreshold
        self.useCropping = False

        self.useNormalization = useNormalization
        self.useWhitening = useWhitening
        self.useRandomOrder = useRandomOrder

        print("Loading and formating image data ....")
        self.generate()
        print("Loading and formating image data: Complete")
        print("Data size: Input Data", self.x_train.shape, " Truth Data:", self.y_train.shape);

    def loadOldCSV(self):
        angle_dict = {}
        with open(self.anglecsv) as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                # key = row['Current Standardized Name']
                
                # Get key and format key
                key = row['linked_images'] 
                key = int(key)
                if key < 10000:
                    key ='0' + str(key) + '.png'
                else:
                    key = str(key) + '.png'

                if row['Alpha'] != '' and row['Alpha'] != 'cm' and row['Alpha'] != 'Ia':
                    alpha = float(row['Alpha'])
                    beta = float(row['Beta'])
                    angle_dict[key] = [alpha, beta]
        return angle_dict 

    def loadMartaCSV(self):
        outcome_dict = {} #angle_dict = {}
        with open(self.anglecsv) as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                
                # Get key and format key
                key = row['Match 1'] 
                outcome_dict[key] = row['Outcome']

                #if row['C Alpha'] != '' and row['C Alpha'] != 'cm' and row['C Alpha'] != 'X' and row['C Alpha'] != 'no' and row['C Alpha'] != 'No ' and row['C Beta'] != 'cm' and row['C Beta'] != 'X':
                    #alpha = float(row['C Alpha'])
                    #beta = float(row['C Beta'])
                    #angle_dict[key] = [alpha, beta]
        return outcome_dict #angle_dict 

    def formatAngleData(self):
        files = os.listdir(self.imagedir)

        files_with_angle = []
        outcome = [] #angle_data = []
        for f in files:
            if f in self.outcome_dict:  #self.angle_dict
                outcome.append(self.outcome_dict[f]) #angle_data.append(self.angle_dict[f])
                files_with_angle.append(f)

        return outcome, files_with_angle #np.array(angle_data), files_with_angle
     
    def cropimages(self, img):

        # Do cropping 

        return img

    def loadImageData(self):
        files = self.files
        
        for f in tqdm(files):
            img = io.imread(self.imagedir + f)
            if(self.useCropping): 
                img = cropimages(img)
            img = transform.resize(img, (self.HEIGHT, self.WIDTH, COLOR_CHANNELS), mode='constant')
            img = img[:,:,1] 

            if f == files[0]:
                imgs = img[None,:]
            else:
                imgs = np.concatenate( (imgs, img[None,:]), axis=0)

        imgs = np.reshape( imgs, (-1, self.HEIGHT, self.WIDTH, self.CHANNELS))
        return imgs

    def generate(self):
        self.outcome_dict = self.loadMartaCSV() #self.angle_dict = self.loadMartaCSV()
        self.outcome, self.files = self.formatAngleData()  #self.angle_data, self.files = self.formatAngleData() 
        
        # Load image data
        self.image_data = self.loadImageData()

        # Grab angle data:
        #self.angle_data = np.reshape(self.angle_data[:,0], (-1, 1))

        # Randomize data order
        if self.useRandomOrder:
            indices = [_ for _ in range(len(self.outcome))] #indices = [_ for _ in range(len(self.angle_data))]
            self.image_data = self.image_data[indices]
            self.outcome = self.outcome[indices] #self.angle_data = self.angle_data[indices]

        # Data preprocessing
        if self.useNormalization:
            self.image_data, self.img_min, self.img_max = self.normalize(self.image_data)

        if self.useWhitening:
            self.image_data, self.img_mean, self.img_std = self.whiten(self.image_data)

        if self.useBinaryClassify:
            self.outcome = self.threshold(self.outcome) #self.angle_data = self.threshold(self.angle_data)
        #else:
            #if self.useNormalization:
                #self.angle_data, self.ang_min, self.ang_max = self.normalize(self.angle_data)
            #if self.useWhitening:
                #self.angle_data, self.ang_mean, self.ang_std = self.whiten(self.angle_data)

        # Split data into test/training sets 
        index1 = int(self.ratio1 * len(self.image_data)) # Split index
        self.x_train = self.image_data[0:index1, :]
        index2 = int((self.ratio1+self.ratio2) * len(self.image_data))
        self.x_val = self.image_data[index1:index2, :] 
        self.x_test = self.image_data[index2:, :] 

        if self.useBinaryClassify:
            self.y_train = self.outcome[0:index1]
            self.y_val = self.outcome[index1:index2]
            self.y_test = self.outcome[index2:]

        else:
            self.y_train = self.outcome[0:index1]
            self.y_test = self.outcome[index1:index2]
            self.y_test = self.outcome[index2:]


    def threshold(self, data): 
        threshold = np.zeros(len(data))
        #threshold = (data > self.binaryThreshold) * 1
        for i in range(len(data)):
            threshold[i]= (data[i]!=self.binaryThreshold)*1
        # OneHot Encoding #onehot = np.concatenate( (1 - threshold, threshold), axis = 1)  # OneHot Encoding
        return threshold #return onehot

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
    xs, ys = data.next_batch(128)

    #plt.figure(1)
    #data.plot(1)

    plt.figure()
    plt.hist(ys)
    plt.show()

    plt.figure()
    plt.hist(data.y_train)
    plt.show()