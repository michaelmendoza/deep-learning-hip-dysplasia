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
# importing one hot encoder from sklearn
# There are changes in OneHotEncoder class
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

# Constants
COLOR_CHANNELS = 3

class Stats:
    def __init__(self, min, max, mean, std):
        self.min = min
        self.max = max
        self.mean = mean
        self.std = std

class DataGeneratorCrop:
    def __init__(self,
        imagedir = '/home/nealb/Documents/krithika/src/images/', #insert here the directory where you store the hip images
        csvFileCrop =  '/home/nealb/Documents/krithika/final_data.csv', #insert here the file location of the csv with the patient data
        width = 500,    #insert here the image width
        height = 500,   #insert here the image height
        ratio1 = 0.8,   #this is the percentage for training, in this case 80%
        ratio2 = 0.1,   #this is the percentage for validation, 10% and hence the remaining 10% for testing
        #useBinaryClassify = True,   #we will be using a binary classification, 1 or 0 - FIGURE OUT WHAT IT WOULD BE FOR S
        #binaryThreshold = 60.0,     #above 60 is healthy, below 60 is diseased
        useNormalization = False,
        useWhitening = True,
        useRandomOrder = True):

        #also need to have self.GENDER, self.SIDE, self.INDICATION, self.BIRTHWEIGHT, self.ALPHA, self.BETA
        self.imagedir = imagedir
        self.csvFileCrop = csvFileCrop
        self.WIDTH = width
        self.HEIGHT = height
        self.CHANNELS = 1
        self.ratio1 = ratio1
        self.ratio2 = ratio2

        #self.useBinaryClassify = useBinaryClassify
        #self.binaryThreshold = binaryThreshold
        self.useCropping = False

        self.useNormalization = useNormalization
        self.useWhitening = useWhitening
        self.useRandomOrder = useRandomOrder

        print("Loading and formating image data ....")
        self.generateCrop()     #first function that leads onto the rest, go down to find it
        print("Loading and formating image data: Complete")
        print("\nTraining data size: Input Data", self.x_train.shape, " Truth Data for x coordinate:", self.x_coordinate_train.shape, " Truth Data for y coordinate:", self.y_coordinate_train.shape);
        print("\nValidation data size: Input Data", self.x_val.shape, " Truth Data for x coordinate:", self.x_coordinate_val.shape, " Truth Data for y coordinate:", self.y_coordinate_val.shape);
        print("\nTraining data size: Input Data", self.x_test.shape, " Truth Data for x coordinate:", self.x_coordinate_test.shape, " Truth Data for y coordinate:", self.y_coordinate_test.shape);

    #   KRITHIKA - This function loads both the binary outcome values and the patient details. This will be the function used
    def loadCropCSV(self):
        coordinates_dict = {}
        with open(self.csvFileCrop) as csvfile:
            reader = csv.DictReader(csvfile)
            #Note: Here, we are extracting the center point coordinates for the bounding box
            for row in reader:
                key = row['Match 1']    #Match 1 contains the image name of the nonannotated images
                xCoordinate = row['Center point X Coordinate']
                yCoordinate = row['Center point Y Coordinate']
                coordinates_dict[key] = [xCoordinate, yCoordinate]

        return coordinates_dict

    #
    def formatCropData(self):
        files = os.listdir(self.imagedir)

        files_with_crop = []
        crop_data = []
        for f in files:
            if f in self.coordinates_dict:        #this is to ensure all file names in Match 1 are also present in the image directory chosen
                crop_data.append(self.coordinates_dict[f])       #if this is the case the crop data will be saved for that file
                files_with_crop.append(f)      ##and the file name will also be stored

        return np.array(crop_data), files_with_crop   #note: in this case the crop data is converted into a numpy array

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

    #This is the new generate function used, where the outputs are the x and y coordinates of the center point
    #of the bounding box and the input is the scan
    def generateCrop(self):     #this is the initial function used to lead onto the others
        self.coordinates_dict = self.loadCropCSV() #run the function to get the dictionary containing the x and y coordinates of the bounding box
        self.crop_data, self.files = self.formatCropData()  #run the function to get the files and the x and y coordinates of the bounding box which match and are present

        # Load image data
        self.image_data = self.loadImageData()

        # Extract the necessary data from the dictionaries and reshape into column vectors
        self.x_coordinate = np.reshape(self.crop_data[:,0], (-1, 1)) #The (-1, 1) turns the matrix into a column vector
        self.y_coordinate = np.reshape(self.crop_data[:,1], (-1, 1))

        # Randomize data order
        if self.useRandomOrder:
            indices = [_ for _ in range(len(self.crop_data))]
            self.image_data = self.image_data[indices]
            self.x_coordinate = self.x_coordinate[indices]
            self.y_coordinate = self.y_coordinate[indices]

        #Change data type of the x and y coordinate arrays to be floats instead of strings
        self.x_coordinate = self.x_coordinate.astype(np.float64)
        self.y_coordinate = self.y_coordinate.astype(np.float64)

        # Data preprocessing
        if self.useNormalization:
            self.image_data, self.img_min, self.img_max = self.normalize(self.image_data)
            self.x_coordinate, self.x_coordinate_min, self.x_coordinate_max = self.normalize(self.x_coordinate)
            self.y_coordinate, self.y_coordinate_min, self.y_coordinate_max = self.normalize(self.y_coordinate)
            print("HI")

        if self.useWhitening:
            self.image_data, self.img_mean, self.img_std = self.whiten(self.image_data)
            self.x_coordinate, self.x_coordinate_mean, self.x_coordinate_std = self.whiten(self.x_coordinate)
            self.y_coordinate, self.y_coordinate_mean, self.y_coordinate_std = self.whiten(self.y_coordinate)
            print("HI1")

        # Split the image data into test/training/validation sets
        index1 = int(self.ratio1 * len(self.image_data)) # Split index
        self.x_train = self.image_data[0:index1, :]     #this is 80% training
        index2 = int((self.ratio1+self.ratio2) * len(self.image_data))
        self.x_val = self.image_data[index1:index2, :]  #then 10% validation
        self.x_test = self.image_data[index2:, :]   #and 10% testing

        #Split the coordinates details data into test/training/validation sets

        #x coordinates
        self.x_coordinate_train = self.x_coordinate[0:index1, :]
        self.x_coordinate_val = self.x_coordinate[index1:index2, :]
        self.x_coordinate_test = self.x_coordinate[index2:, :]

        #y coordinates
        self.y_coordinate_train = self.y_coordinate[0:index1]
        self.y_coordinate_val = self.y_coordinate[index1:index2]
        self.y_coordinate_test = self.y_coordinate[index2:]

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
