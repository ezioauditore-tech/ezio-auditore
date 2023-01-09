import streamlit as st
import cv2
from PIL import Image,ImageEnhance
import numpy as np
import os
import pandas as pd
 

columns=[]
st.title("Traffic flow")
dfheader = ["Vehicle-Id","Vehicle-Number","Vehicle-Type"]
st.sidebar.title("Menu")
df = pd.DataFrame(
        np.random.randn(100, 3),
        columns=('%s' %ele for ele in dfheader))
option = st.sidebar.selectbox("Select one",("Live Video","Database","Analytics"))

#Database menu
if option == "Database":
    st.table(df)
    for ele in columns:
        st.header(ele)

#Analytics dashboard menu
elif option == "Analytics":
    st.subheader("Analysis")
    st.markdown("***")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("No of in Vehicles")

        #--- add the code to calculate number of in vehicles and append to "noofinvehicle"
        noofinvehicle = "__"
        st.subheader(noofinvehicle)

    with col2:
        st.subheader("No of out Vehicles")
        #--- add the code to calculate number of out vehicles and append to "noofoutvehicle"
        noofoutvehicle = "__"
        st.subheader(noofoutvehicle)

    with col2:
        st.subheader("Vehicles inside the parking")
        #--- add the code to calculate number of Remaining vehicles which are inside the campus and append to "noofremainvehicle"
        noofremainvehicle = "__"
        st.subheader(noofoutvehicle)
    
    st.area_chart(df)



    #OpenCV code
elif option == "Live Video":

    img_file_buffer = st.camera_input("Live Streaming")

    if img_file_buffer is not None:
        bytes_data = img_file_buffer.getvalue()
        cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)


        st.write(type(cv2_img))
