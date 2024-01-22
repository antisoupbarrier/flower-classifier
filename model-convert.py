import os
import tensorflow as tf
from tensorflow import keras
from keras import backend as K


model = keras.saving.load_model('densenet201_v3_15_0.913.h5')
tf.saved_model.save(model, 'flower-model')