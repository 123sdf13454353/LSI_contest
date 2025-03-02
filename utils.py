import os
import random
import numpy as np
import tensorflow as tf

def seed_all(value=42):
    random.seed(value)
    np.random.seed(value)
    tf.random.set_seed(value)
    os.environ['PYTHONHASHSEED'] = str(value)

seed_all()
