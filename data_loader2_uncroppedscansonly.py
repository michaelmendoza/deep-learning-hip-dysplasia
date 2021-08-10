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

class DataGenerator2:       #data generator2 is to use alpha or calpha as the diagnostic parameter

    def __init__(self,
        #Note: to get cropped scans, change the directory below and the width and height when calling this to 350x270
        imagedir = '/home/krithika/DDH_Project/DDH_Project/uncropped_hip_images/', #insert here the directory where you store the hip images
        anglecsv =  'final_data.csv', #insert here the file location of the csv with the patient data
        width = 256,    #insert here the image width
        height = 256,   #insert here the image height
        ratio1 = 0.8,   #this is the percentage for training, in this case 80%
        ratio2 = 0.1,   #this is the percentage for validation, 10% and hence the remaining 10% for testing
        useBinaryClassify = True,   #we will be using a binary classification, 1 or 0
        binaryThreshold = 60.0,     #above 60 is healthy, below 60 is diseased
        useNormalization = False,
        useWhitening = True,
        useRandomOrder = True):

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
        self.generate()     #first function that leads onto the rest, go down to find it
        print("Loading and formating image data: Complete")
        print("Data size: Input Data", self.x_train.shape, " Truth Data:", self.y_train.shape);

    def loadOldCSV(self):       #this was the loading process for the old csv file Michael had, not being used now
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

    def loadMartaCSV(self):     #this is the loading process I used, and the one currently used
        angle_dict = {}
        with open(self.anglecsv) as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:      #as seen below you can use calpha or alpha, in this case the standard one used is calpha

                #If you are using cropped scans, Use Match 2. If you are using uncropped scans, use Match 1
                # Get key and format key
                key = row['Match 1']    #Match 1 contains the image name of the nonannotated images
                #if row['Alpha'] != '' and row['Alpha'] != 'cm' and row['Alpha'] != 'X' and row['Alpha'] != 'No ' and row['Beta'] != 'cm' and row['Alpha'] != 'no':
                if row['C Alpha'] != '' and row['C Alpha'] != 'cm' and row['C Alpha'] != 'X' and row['C Alpha'] != 'No ' and row['Beta'] != 'cm' and row['C Alpha'] != 'no':
                    alpha = float(row['C Alpha'])
                    #alpha = float(row['Alpha'])
                    beta = float(row['Beta'])       #the beta value was initially also saved though not used
                    angle_dict[key] = [alpha, beta]
        return angle_dict

    def formatAngleData(self):
        files = os.listdir(self.imagedir)

        files_with_angle = []
        angle_data = []
        for f in files:
            if f in self.angle_dict:        #this is to ensure all file names in Match 1 are also present in the image directory chosen
                angle_data.append(self.angle_dict[f])       #if this is the case the angle data will be saved for that file
                files_with_angle.append(f)      ##and the file name will also be stored

        return np.array(angle_data), files_with_angle   #note in this case the angle data list is converted into a numpy array

    def cropimages(self, img):

        #Have done it previously so not necessary anymore

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

    def generate(self):     #this is the initial function used to lead onto the others
        self.angle_dict = self.loadMartaCSV()       #run the function to get the angle values
        self.angle_data, self.files = self.formatAngleData()    #run the function to get the files and angle values which match and are present

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

        if self.useWhitening:
            self.image_data, self.img_mean, self.img_std = self.whiten(self.image_data)

        if self.useBinaryClassify:
            self.angle_data = self.threshold(self.angle_data)   #refer to the threshold function to extract the angle data as 1 or 0s
        else:
            if self.useNormalization:
                self.angle_data, self.ang_min, self.ang_max = self.normalize(self.angle_data)
            if self.useWhitening:
                self.angle_data, self.ang_mean, self.ang_std = self.whiten(self.angle_data)

        # Split data into test/training/validation sets
        index1 = int(self.ratio1 * len(self.image_data)) # Split index
        self.x_train = self.image_data[0:index1, :]     #this is 80% training
        index2 = int((self.ratio1+self.ratio2) * len(self.image_data))
        self.x_val = self.image_data[index1:index2, :]  #then 10% validation
        self.x_test = self.image_data[index2:, :]   #and 10% testing

        if self.useBinaryClassify:      #Split the y labels now, the value of the angle
            self.y_train = self.angle_data[0:index1]
            self.y_val = self.angle_data[index1:index2]
            self.y_test = self.angle_data[index2:]

        else:
            self.y_train = self.angle_data[0:index1]
            self.y_test = self.angle_data[index1:index2]
            self.y_test = self.angle_data[index2:]

    def threshold(self, data):
        threshold = (data < self.binaryThreshold) * 1   #if the value of the alpha angle is below 60 the patient has a 1/diseased, if above or equal it has a 0/healthy
        onehot = np.concatenate( (1 - threshold, threshold), axis = 1)  # OneHot Encoding, mainly used for multiple classes
        return onehot[:, 0]     #notice how we only take the first column which corresponds to a 1 for below 60 and a 0 for above

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
    data = DataGenerator2()     #important to use DataGenerator2 for the alpha angle
    xs, ys = data.next_batch(128)

    #plt.figure(1)
    #data.plot(1)

    plt.figure()
    plt.hist(ys)
    plt.show()

    plt.figure()
    plt.hist(data.y_train)
    plt.show()
