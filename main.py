
from src.tf.estimator import Estimator
from src.tf.classifier import Classifier
from src.tf2.classifier import Classify
from src.tf2.classifier import Load
from src.v3.classifier import Train

if __name__ == '__main__':

   import os
   if not os.path.exists('results/tf2/'):
      os.makedirs('results/tf2/')
   
   #estimator = Estimator(num_steps = 20000, batch_size=128, display_step=1000, save_step = 5000, doRestore = False)
   #classifier = Classifier(num_steps = 50000, batch_size=32, display_step=1000, save_step = 10000, doRestore = False) 
   #Classify()
   #Load("./results/tf2/classifier__2020-02-04__05-39.h5")
   Train()