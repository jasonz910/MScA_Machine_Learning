filename = '/home/appuser/venv/lib/python3.9/site-packages/keras_vggface/models.py'
text = open(filename).read()
open(filename, 'w+').write(text.replace('keras.engine.topology', 'tensorflow.keras.utils'))

import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import joblib
import sys
import os
import pandas as pd 
from keras_vggface.utils import preprocess_input
from keras_vggface import VGGFace
from keras.models import Model
from tensorflow.keras.preprocessing.image import load_img
import warnings
warnings.filterwarnings("ignore")
import av
from PIL import Image
import io

@st.cache_resource(show_spinner=False)
def load_svr():
    return joblib.load('best_svr_model.pkl')

@st.cache_resource(show_spinner=False)
def load_vggface():
    vggface = VGGFace(model='senet50')
    vggface_model = Model(inputs=vggface.input, outputs=vggface.get_layer('avg_pool').output)
    return vggface_model


svr_model = load_svr()
vggface_model = load_vggface()

@st.cache_resource(show_spinner=False)
def get_feature(img):
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img, version=2) 
    feature = vggface_model.predict(img)
    return feature[0][0]

cascPath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)

font = cv2.FONT_HERSHEY_SIMPLEX

#@st.cache_data
def predict_bmi(frame):
    pred_bmi = []

    faces = faceCascade.detectMultiScale(
            cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
            scaleFactor = 1.15,
            minNeighbors = 5,
            minSize = (30,30),
            )

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        image = frame[y:y+h, x:x+w]
        img = image.copy()
        img = cv2.resize(img, (224, 224))
        img = np.array(img).astype(np.float64)
        features = get_feature(img)
        preds = svr_model.predict(features)
        pred_bmi.append(preds[0])
        cv2.putText(frame, f'BMI: {preds}', (x+5, y-5), font, 2, (255, 255, 255), 2)

    return pred_bmi

#@st.cache_data
def prepare_download(img):
    buf = io.BytesIO()
    img.save(buf, format='JPEG')
    image_bytes = buf.getvalue()
    return image_bytes

def bmi_segment(bmi):
    if bmi<18.5:
        st.write('**Sorry you are UNDERWEIGHT. Eat More!!🥩**')
    elif 18.5<=bmi<=25:
        st.write('**Hurray! Your BMI looks good! Keep Going!💪**')
    elif 25<bmi<30:
        st.write('**Sorry you are OVERWEIGHT! Be careful about your diet.🥦**')
    elif 30<=bmi<35:
        st.write('**Hey, You are MODERATELY OBESE. Eat healthy and exercise more please.🥗**')
    elif 35<=bmi<=40:
        st.write('**Oh no! You are SEVERELY OBESE. Please eat healthy and exercise more.🏃**')
    elif bmi>40:
        st.write('**Watch out! You are VERY SEVERELY OBESE. Please reach out your doctor for professional advice on your health.😞**')

################################  PAGE CONTENT  ################################

st.markdown("<h1 style='text-align: center; color: #800020;'>Take a Photo to Predict BMI</h1>", unsafe_allow_html=True)

picture = st.camera_input("👇Take Photo here:")

if picture is not None:

    picture_taken = np.array(Image.open(picture))
    bmi_pred = predict_bmi(picture_taken)
    pil_pic_taken = Image.fromarray(picture_taken)

    photo, result = st.columns([1, 1])
    
    with photo:
        st.image(pil_pic_taken, use_column_width='auto', clamp=True)
        photo_download = prepare_download(pil_pic_taken)

    with result:
        if len(bmi_pred)==0:
            st.markdown("Sorry, we don't detect any faces. Please re-take your photo.")
        elif len(bmi_pred)==1:
            st.markdown('1 face is detected')
            st.write(f'The BMI of this face is: **{round(bmi_pred[0],2)}**')
            bmi_segment(bmi_pred[0])
        else:
            st.markdown(f'{len(bmi_pred)} faces are detected')
            for i in range(len(bmi_pred)):
                st.write(f'The BMI for face {i+1} is: **{round(bmi_pred[i],2)}**')
                bmi_segment(bmi_pred[i])

        if len(bmi_pred)!=0:
            st.download_button(
                label="Download Prediction",
                data=photo_download,
                file_name='Photo_with_BMI.jpg',
                mime='image/jpeg',
            )

hide_default_format = """
       <style>
       footer {visibility: hidden;}
       </style>
       """
st.markdown(hide_default_format, unsafe_allow_html=True)
    
    

