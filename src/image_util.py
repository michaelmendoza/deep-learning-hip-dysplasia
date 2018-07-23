
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
import math

import os
from skimage import io, transform
from tqdm import trange, tqdm

def filterImages():
    imagedir = 'data/images-filtered/'
    files = os.listdir(imagedir)
    for f in tqdm(files):
        img = io.imread(imagedir + f)
        
        removeFile = img.shape[0] < 600 or img.shape[1] < 600
        if removeFile:
            os.remove(imagedir + f)
            print ("File Removed:", imagedir + f, img.shape, removeFile)

def cropImages():
    crop = [600, 600]

    imagedir = 'data/images-filtered/'
    files = os.listdir(imagedir)
    for f in tqdm(files):
        img = io.imread(imagedir + f)

        height =  img.shape[0]
        width = img.shape[1]
        dw = round( (width - crop[0]) / 2.0 )
        dh = round( (height - crop[1]) / 2.0 )
        img = img[dh:crop[0] + dh, dw:crop[1] + dw]
        io.imsave(imagedir + f, img)

if __name__ == '__main__':
    #filterImages()
    cropImages()
