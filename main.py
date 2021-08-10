
#from src.tf.estimator import Estimator #from the file 'estimator', import the class Estimator
from classifier2_PDNet import Classify2_PDNet    #Import this classifier if you want to train the patient details network
from classifier2_croppedscansonly import Classify2_croppedScans    #Import this classifier if you want to train the patient details network
from classifier2_uncroppedscansonly import Classify2_uncroppedScans    #Import this classifier if you want to train the patient details network
from classifier2_mixed_tests import Classify2_mixed
from src.tf2.regression_autocrop import AutoCrop    #Import this regression class if you want to train the auto-cropping network
import tensorflow as tf


if __name__ == '__main__': #This code is run when the main.py file is run

  #Classify2_croppedScans()
  #Classify2_uncroppedScans()
  #AutoCrop() #Use this to train the autocropping network
  #Classify2_PDNet()    #Use this to train the patient details network

  Classify2_mixed()
