import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

# Load in our model
reload_sm_keras = tf.keras.models.load_model(
  "./model",
  custom_objects={'KerasLayer': hub.KerasLayer})

reload_sm_keras.summary()
