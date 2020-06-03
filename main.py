
from src.tf.estimator import Estimator
from src.tf.classifier import Classifier
from src.tf2.classifier import Classify
from src.tf2.classifier2 import Classify2
import tensorflow as tf

if __name__ == '__main__':
   #estimator = Estimator(num_steps = 20000, batch_size=128, display_step=1000, save_step = 5000, doRestore = False)
   #classifier = Classifier(num_steps = 50000, batch_size=32, display_step=1000, save_step = 10000, doRestore = False) 
   Classify2()
   #try:
   		#with tf.device('/device:GPU:1'):
   			#Classify()
   #except RuntimeError as e:
  		#print(e)