
from src.tf.estimator import Estimator
from src.tf.classifier import Classifier
from src.keras.classifier import Classify

if __name__ == '__main__':
   #estimator = Estimator(num_steps = 20000, batch_size=128, display_step=1000, save_step = 5000, doRestore = False)
   #classifier = Classifier(num_steps = 50000, batch_size=32, display_step=1000, save_step = 10000, doRestore = False) 
   Classify()
