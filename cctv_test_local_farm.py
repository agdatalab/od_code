# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 17:28:09 2020

@author: Jang Ikhoon
"""
#%%
import cv2 
import numpy as np
import ftplib
import os
#%%
# DIR setting : initializing when first time
if not os.path.isdir("/home/pi/Downloads/cctv_image"):
    os.mkdir("/home/pi/Downloads/cctv_image")
url = 'rtsp://admin:farmai1234@192.168.0.102:554/Streaming/channels/102' 
#%%
cap = cv2.VideoCapture(url)
ret, frame = cap.read() 
image = np.array(frame)
cv2.imwrite('/home/pi/Downloads/cctv_image/cctv_test.jpg', image)

#%%
