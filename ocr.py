import cv2
import numpy as np
import tensorflow as tf
from PIL import Image

def recognize_handwriting(image):
    # Preprocess the image for the model
    image_array = np.array(image)
    gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (28, 28))  # Resize to 28x28
    normalized = resized / 255.0
    input_data = np.expand_dims(normalized, axis=0)
    input_data = np.expand_dims(input_data, axis=1)

    # Load the pre-trained model
    model = tf.keras.models.load_model('./model.h5')

    # Predict the equation
    predictions = model.predict(input_data)
    print (predictions)
    # equation = decode_predictions(predictions)
    # return equation

if __name__=='__main__':
    image = Image.open('recognized_equation.png')
    recognize_handwriting(image)

