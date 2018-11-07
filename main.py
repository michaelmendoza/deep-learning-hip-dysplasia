
from src.tf.estimator import Estimator
from src.tf.classifier import Classifier

if __name__ == '__main__':
   #estimator = Estimator(num_steps = 1000, batch_size=128, display_step=100)
   classifier = Classifier(num_steps = 50000, batch_size=32, display_step=1000, save_step = 10000, doRestore = False) 

