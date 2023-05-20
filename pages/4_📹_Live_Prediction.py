filename = '/home/appuser/venv/lib/python3.9/site-packages/keras_vggface/models.py'
text = open(filename).read()
open(filename, 'w+').write(text.replace('keras.engine.topology', 'tensorflow.keras.utils'))

import streamlit as st
from streamlit_webrtc import webrtc_streamer
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
from turn import get_ice_servers
import threading
from typing import Union

@st.cache_resource(show_spinner=False)
def load_svr():
    return joblib.load('svr_model.pkl')

@st.cache_resource(show_spinner=False)
def load_vggface():
    vggface = VGGFace(model='vgg16', include_top=True, input_shape=(224, 224, 3), pooling='avg')
    return Model(inputs=vggface.input, outputs=vggface.get_layer('fc6').output)


svr_model = load_svr()
vggface_model = load_vggface()

@st.cache_resource(show_spinner=False)
def get_fc6_feature(img):
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img, version=2) 
    fc6_feature = vggface_model.predict(img)
    return fc6_feature

cascPath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)

font = cv2.FONT_HERSHEY_SIMPLEX

#@st.cache_resource
def predict_bmi(frame):
    faces = faceCascade.detectMultiScale(
            cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
            scaleFactor = 1.15,
            minNeighbors = 5,
            minSize = (30,30),
            )

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 1)
        image = frame[y:y+h, x:x+w]
        img = image.copy()
        img = cv2.resize(img, (224, 224))
        img = np.array(img).astype(np.float64)
        features = get_fc6_feature(img)
        preds = svr_model.predict(features)
        cv2.putText(frame, f'BMI: {preds}', (x+5, y-5), font, 1, (255, 255, 255), 2)

#@st.cache_data
def prepare_download(img):
    buf = io.BytesIO()
    img.save(buf, format='JPEG')
    image_bytes = buf.getvalue()
    return image_bytes

class VideoProcessor:
    def __init__(self):
        self.frame_lock = threading.Lock()
        self.out_image = None

    def recv(self, frame):
        frm = frame.to_ndarray(format='bgr24')
        predict_bmi(frm)
        with self.frame_lock:
            self.out_image = frm

        return av.VideoFrame.from_ndarray(frm, format='bgr24') 
    
###############################

st.markdown("<h1 style='text-align: center; color: #B92708;'>Predict Your BMI Live</h1>", unsafe_allow_html=True)

ctx = webrtc_streamer(key="live predict", video_transformer_factory=VideoProcessor, sendback_audio=False, rtc_configuration={"iceServers": get_ice_servers()},)

if ctx.video_transformer:
    snap = st.button("Snapshot")
    if snap:
        with ctx.video_transformer.frame_lock:
            out_image = ctx.video_transformer.out_image

        if out_image is not None:
            st.write("Output image:")
            st.image(out_image, channels="BGR")
        else:
            st.warning("No frames available yet.")


hide_default_format = """
       <style>
       footer {visibility: hidden;}
       </style>
       """
st.markdown(hide_default_format, unsafe_allow_html=True)