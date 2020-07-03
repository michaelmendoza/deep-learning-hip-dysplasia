
from src.tf.estimator import Estimator
from src.tf.classifier import Classifier
from src.tf2.classifier import Classify      #Import this classifier if you want to use 'outcome' as the diagnostic parameter
from src.tf2.classifier2 import Classify2    #Import this classifier if you want to use the alpha angle as the diagnostic parameter
import tensorflow as tf

if __name__ == '__main__':
   #Classify()
   Classify2()    #in this case we will be using alpha
