
from src.tf.estimator import Estimator

if __name__ == '__main__':
   estimator = Estimator(num_steps = 1000, batch_size=128, display_step=100)
   