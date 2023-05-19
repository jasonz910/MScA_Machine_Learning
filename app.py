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

svr_model = joblib.load('svr_model.pkl')
vggface = VGGFace(model='vgg16', include_top=True, input_shape=(224, 224, 3), pooling='avg')
vggface_model = Model(inputs=vggface.input, outputs=vggface.get_layer('fc6').output)

def get_fc6_feature(img):
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img, version=2) 
    fc6_feature = vggface_model.predict(img)
    return fc6_feature

cascPath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)

font = cv2.FONT_HERSHEY_SIMPLEX

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

def prepare_download(img):
    buf = io.BytesIO()
    img.save(buf, format='JPEG')
    image_bytes = buf.getvalue()
    return image_bytes

class VideoProcessor:

    def recv(self, frame):

        frm = frame.to_ndarray(format = 'bgr24')

        predict_bmi(frm)

        return av.VideoFrame.from_ndarray(frm, format = 'bgr24') 
    


###############################


st.title("🎈Wecome to JZ's BMI Prediction📷")

st.caption('The BMI predicted from this site is based on learning your face features. This site does not take responsibility for providing accurate and credible BMI results. Thank you! 😊')

upload, photo, live = st.tabs(['Upload Your Photo','Take a Photo','Video Live'])

with upload:

    upload_files = st.file_uploader("Upload your photo to predict:", accept_multiple_files=True)

    for upload_file in upload_files:

        index = 1

        pic_upload = np.array(Image.open(upload_file))

        predict_bmi(pic_upload)

        pil_pic_upload = Image.fromarray(pic_upload)
        st.image(pil_pic_upload, use_column_width=True, clamp=True)

        pic_download = prepare_download(pil_pic_upload)
    
        st.download_button(
            label="Download Prediction",
            data=pic_download,
            file_name='Image'+str(index)+'_with_BMI.jpg',
            mime='image/jpeg',
        )
        
        index += 1


with photo:

    picture = st.camera_input("Take a photo to predict:")

    if picture is not None:

        picture_taken = np.array(Image.open(picture))

        predict_bmi(picture_taken)

        pil_pic_taken = Image.fromarray(picture_taken)
        st.image(pil_pic_taken, use_column_width=True, clamp=True)

        photo_download = prepare_download(pil_pic_taken)

        st.download_button(
            label="Download Prediction",
            data=photo_download,
            file_name='Photo_with_BMI.jpg',
            mime='image/jpeg',
        )

with live:

    webrtc_streamer(key="example", video_transformer_factory=VideoProcessor, sendback_audio=False)