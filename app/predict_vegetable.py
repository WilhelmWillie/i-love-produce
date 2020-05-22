import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import requests
import json

# Load in our model
model = tf.keras.models.load_model(
  "../model",
  custom_objects={'KerasLayer': hub.KerasLayer})

# URL for image we want to run on our model
IMAGE_RES = 224

# Method that gets an image from a URL and returns a numpy array we can use to predict
def get_tensor_from_image_url(url):
  img = tf.image.decode_jpeg(requests.get(url).content, channels=3)
  img = tf.image.convert_image_dtype(img, tf.float32)
  img = tf.image.resize(img, [IMAGE_RES, IMAGE_RES])

  return img[np.newaxis, ...]

# Put together a message to send back
def assemble_message(vegetable):
  expiration = {
    "carrot": "14-21 days in the fridge",
    "onion": "7-10 days on a countertop/pantry",
    "tomato": "7-10 days on a countertop/pantry"
  }

  message = "hello - this looks like a(n)... " + vegetable + "! a(n) " + vegetable + " is good for " + expiration[vegetable] + ". hope this helped!"
  return message

# Predict using our model and print out the predicted label
def predict(IMAGE_URL):
  CLASSES = ['carrot', 'onion', 'tomato']

  image = get_tensor_from_image_url(IMAGE_URL)
  prediction_results = model.predict(image)
  prediction_class = np.argmax(prediction_results[0], axis=-1)
  predicted_vegetable = CLASSES[prediction_class]

  return assemble_message(predicted_vegetable)