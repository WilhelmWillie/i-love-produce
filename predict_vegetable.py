import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import requests

# Load in our model
model = tf.keras.models.load_model(
  "./model",
  custom_objects={'KerasLayer': hub.KerasLayer})

model.summary()

# URL for image we want to run on our model
IMAGE_URL = "https://fearlessfresh.com/wp-content/uploads/2015/07/Carrots-in-Hand.jpg"
IMAGE_RES = 224

# Method that gets an image from a URL and returns a numpy array we can use to predict
def get_tensor_from_image_url(url):
  img = tf.image.decode_jpeg(requests.get(url).content, channels=3)
  img = tf.image.convert_image_dtype(img, tf.float32)
  img = tf.image.resize(img, [IMAGE_RES, IMAGE_RES])

  return img[np.newaxis, ...]

CLASSES = ['carrot', 'onion', 'tomato']

# Predict using our model and print out the predicted label
image = get_tensor_from_image_url(IMAGE_URL)
prediction_results = model.predict(image)
prediction_class = np.argmax(prediction_results[0], axis=-1)
print(prediction_results[0])
print("Predicted label: ", CLASSES[prediction_class])