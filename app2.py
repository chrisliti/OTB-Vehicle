## Import Libraries

import numpy as np
import os
import keras
from tensorflow.keras.utils import img_to_array
from keras.models import load_model
from keras.applications.vgg19 import preprocess_input
import streamlit as st
import cv2

import pickle

from PIL import Image

## Load model

model = load_model('model.h5')

## Image dimensions

img_height = 256
img_width = 256

## Class names list
class_names_list = ['Bus', 'Minibuses', 'Saloon', 'Truck', 'Van']

## other class
#other_class = ['Minibuses', 'Saloon','Van']
other_class = { "Van": "other", "Saloon": "other","Minibuses":"other"}

## Class coercision fxn
def multiple_replace(text, wordDict):
    for key in wordDict:
        text = text.replace(key, wordDict[key])
    return text

## Main Fxn
## Classification function
def predictor ():

  ## Load image and resize
  upld_img = st.file_uploader("Upload Image below...", type=["jpg", "jpeg", "png"])

  ## Convert image to array
  i = img_to_array(upld_img)

  ## Pre-process imput using keras
  im = preprocess_input(i)

  ## Expand dimensions
  img = np.expand_dims(im,axis=0)

  ## Classify image
  predictions = model.predict(img)

  ##  Get image label
  predicted_class = class_names_list[np.argmax(predictions)]

  ## Convert van and saloon to other
  predicted_class = multiple_replace(predicted_class,other_class)
  

  ## Return dictionary
  result = {'image': upld_img, 'class': predicted_class}

  return result

#Setting Title of App
#st.title("Vehicle Classification")


html_temp = """
<div style="background-color:dodgerblue;padding:10px">
<h2 style="color:white;text-align:center;">Vehicle Type Classification App </h2>
</div>
    """
st.markdown(html_temp,unsafe_allow_html=True)


image = Image.open('vehicle-types-mage.jpeg')
st.image(image,use_column_width=True)

st.markdown("""
This web page leverages deep learning to classify vehicle images as:
* Saloon
* Bus
* Truck
* Nissan / Van
"""
)


#Uploading the image
vehicle_image = st.file_uploader("Upload Image below...", type=["jpg", "jpeg", "png"])
submit = st.button('Classify Image')

with st.spinner('Classifying...'):
  #On predict button click
  if submit:


      if vehicle_image is not None:
        # Convert the file to an opencv image.
        file_bytes = np.asarray(bytearray(vehicle_image.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)
        
        # Displaying the image
        st.image(opencv_image, channels="BGR",width=512)

        ## Pre-process imput using keras
        opencv_image = preprocess_input(opencv_image)
        
        #Resizing the image
        opencv_image = cv2.resize(opencv_image, (256,256))
    
        #Convert image to 4 Dimension
        opencv_image.shape = (1,256,256,3)

        #Make Prediction
        predictions = model.predict(opencv_image)

        predicted_class = class_names_list[np.argmax(predictions)]

        ## Convert van and saloon to other
        predicted_class = multiple_replace(predicted_class,other_class)
        
        st.subheader(str("Image classified as "+predicted_class))
