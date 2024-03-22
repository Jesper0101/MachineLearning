import streamlit as st
import numpy as np
import joblib
import cv2
from sklearn.preprocessing import StandardScaler
    
loaded_model = joblib.load('svm_model.sav')


def processImage(input):
        
    data_in = input.getvalue()

    decode = cv2.imdecode(np.frombuffer(data_in, np.uint8), cv2.IMREAD_GRAYSCALE)
    _, thresholded = cv2.threshold(decode, threshVal, 255, cv2.THRESH_BINARY_INV)
    imgArr = thresholded/255
    resized = cv2.resize(imgArr, (28, 28), interpolation=cv2.INTER_AREA)
    
    #Standardize the image
    scaleImg = StandardScaler()
    fix_the_pic = scaleImg.fit_transform(resized.reshape(-1, 1)).reshape(resized.shape)

    #Display the reformatted image
    st.write("Reformatted image: ")
    st.image(fix_the_pic, width=128 ,output_format="auto", clamp=True)

    if st.button('Make Prediction'):
       prediction(fix_the_pic)

    
def prediction(inputImg):
     
    #Model prediction
    flat = inputImg.flatten().reshape(1, -1)
    st.write("Predicted digit: ", loaded_model.predict(flat))


     
#Build a sidebar
with st.sidebar:

    choice = st.radio(
        "Choose a method",
        ("Upload", "Webcam"),
        captions = [ "Upload an image", "Use your Webcam"]
    )


if choice == 'Upload':
    upload = st.file_uploader("Upload your picture of a digit here:", type=['png', 'jpeg', 'jpg'])
    threshVal = st.slider('Threshold value', 0, 254, 100)

    if upload is not None:
        col1, col2 = st.columns(2)
        with col1:
            st.write("Uploaded image: ")
            st.image(upload, width=128)
        with col2:
            processImage(upload)

else: 
    buffer = st.camera_input("Use your Webcam and take a picture of one digit")
    threshVal = st.slider('Threshold value', 0, 254, 100)

    if buffer is not None:
        processImage(buffer)