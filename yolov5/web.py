import streamlit as st
# import tensorflow as tf
# import numpy as np
# from PIL import Image
# import pydot
# import graphviz
# import pydotplus
from streamlit.delta_generator import DeltaGenerator
# from tensorflow.keras.models import *
# from tensorflow_addons.losses import SigmoidFocalCrossEntropy
import os
import cv2
from PIL import Image

# def load_trained_model():
#    model_1 = load_model (PATH of your trained models)
#    return model_1

# model_1= load_trained_model()

def global_upload_file(uploaded_file):
    global image_file
    image_file= uploaded_file
    # print(image_file)


# def preprocess_image (image):
#   image = np.expand_dims(image, axis=0)
#   image = tf.image.resize(image, size=(256,256))
#   image = tf.cast(image, dtype='float32')/255.0
#   return image

# def predict(image):
#   result_1= model_1.predict(image)
#   return result_1

st.title("""
  Detection and localization abnormalities in chest X-Ray image
""")
st.write("\n")

col_1, col_2= st.beta_columns(2)

with col_1:
  col_1.header("Upload Image")
  st.set_option('deprecation.showfileUploaderEncoding', False)

  uploaded_file: DeltaGenerator = st.file_uploader(" ", type=['png', 'jpg', 'jpeg'])
#   print(DeltaGenerator.name)

  global_upload_file(uploaded_file)

col_2.header('Prediction result')
if col_2.button("Click Here to Classify"):
  with st.spinner('Classifying ...'):
    # image = Image.open(uploaded_file).convert('RGB')
    image = preprocess_image(image)
    prediction = np.array(predict(image)).reshape(1)
    prediction_number = prediction[0]
    col_2.header("Algorithm Predicts: ")
    st.write("Your chance of getting melanoma is", '\n')
    st.write('**Probability: ** {:.2f} %'.format(prediction_number*100))

image = Image.open('/home/huynhmanhhung441/BK/LuanVanTotNghiep/Demo_detect/yolov5/runs/detect/exp4/0a1addecfc432a1b425d61fe57bc29d2.jpg')
st.image(image, caption='predicted image')
# os.listdir("")
