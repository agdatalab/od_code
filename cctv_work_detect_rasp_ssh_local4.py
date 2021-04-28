######## Webcam Object Detection Using Tensorflow-trained Classifier #########
#
# Author: Evan Juras
# Date: 10/27/19
# Description: 
# This program uses a TensorFlow Lite model to perform object detection on a live webcam
# feed. It draws boxes and scores around the objects of interest in each frame from the
# webcam. To improve FPS, the webcam object runs in a separate thread from the main program.
# This script will work with either a Picamera or regular USB webcam.
#
# This code is based off the TensorFlow Lite image classification example at:
# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/examples/python/label_image.py
#
# I added my own method of drawing boxes and labels using OpenCV.
# 
# Modified by: Shawn Hymel
# Date: 09/22/20
# Description:
# Added ability to resize cv2 window and added center dot coordinates of each detected object.
# Objects and center coordinates are printed to console.
#%%
# Import packages
import os
import argparse
import cv2
import numpy as np
import sys
import time
from threading import Thread
import importlib.util
import sqlite3
import ftplib

#%%
# parameter candidate
image_upload_interval_sec = 5 

farm_id = 'F00001'
camera_id = 'C0000101'
if not os.path.isdir("/home/pi/Downloads/cctv_image"):
    os.mkdir("/home/pi/Downloads/cctv_image")
url = 'rtsp://admin:jinong30#@192.168.0.104:554/Streaming/channels/102' 

# status info
status_person = 0
status_detect_cooltime = 0
time_detect_person = cv2.getTickCount()

#%%
# SQLite DB setting : initializing when first time
if not os.path.isdir("/home/pi/Downloads/sqlite"):
    os.mkdir("/home/pi/Downloads/sqlite")
if not os.path.isfile("/home/pi/Downloads/sqlite/od_result.db"):    
    con = sqlite3.connect("/home/pi/Downloads/sqlite/od_result.db")
    cursor = con.cursor()
    cursor.execute("CREATE TABLE OD_LOCAL(MESURE_DT Datetime, FARM_ID text, CAMERA_ID text, OBJECT_NM text, OBJECT_SCORES real, BOX_X_MIN real , BOX_X_MAX real, BOX_Y_MIN real, BOX_Y_MAX real, CENTER_X real, CENTER_Y real)")
    con.close()
# SQLite DB connet if exiting
con = sqlite3.connect("/home/pi/Downloads/sqlite/od_result.db")
cursor = con.cursor()

#%%
# Define VideoStream class to handle streaming of video from webcam in separate processing thread
# Source - Adrian Rosebrock, PyImageSearch: https://www.pyimagesearch.com/2015/12/28/increasing-raspberry-pi-fps-with-python-and-opencv/
class VideoStream:
    """Camera object that controls video streaming from the Picamera"""
    def __init__(self,resolution=(640,480),framerate=30):
        # Initialize the PiCamera and the camera image stream
        self.stream = cv2.VideoCapture(url)
        ret = self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        ret = self.stream.set(3,resolution[0])
        ret = self.stream.set(4,resolution[1])
            
        # Read first frame from the stream
        (self.grabbed, self.frame) = self.stream.read()

	# Variable to control when the camera is stopped
        self.stopped = False

    def start(self):
	# Start the thread that reads frames from the video stream
        Thread(target=self.update,args=()).start()
        return self

    def update(self):
        # Keep looping indefinitely until the thread is stopped
        while True:
            # If the camera is stopped, stop the thread
            if self.stopped:
                # Close camera resources
                self.stream.release()
                return

            # Otherwise, grab the next frame from the stream
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
	# Return the most recent frame
        return self.frame

    def stop(self):
	# Indicate that the camera and thread should be stopped
        self.stopped = True
#%%
# Define and parse input arguments
parser = argparse.ArgumentParser()
parser.add_argument('--modeldir', help='Folder the .tflite file is located in',
                    required=True)
# parser.add_argument('--camaddr', help='Ip address of webcam, rtsp address is required',
#                     required=True)                    
parser.add_argument('--graph', help='Name of the .tflite file, if different than detect.tflite',
                    default='detect.tflite')
parser.add_argument('--labels', help='Name of the labelmap file, if different than labelmap.txt',
                    default='labelmap.txt')
parser.add_argument('--threshold', help='Minimum confidence threshold for displaying detected objects',
                    default=0.5)
parser.add_argument('--resolution', help='Desired webcam resolution in WxH. If the webcam does not support the resolution entered, errors may occur.',
                    default='1280x720')
parser.add_argument('--edgetpu', help='Use Coral Edge TPU Accelerator to speed up detection',
                    action='store_true')
#%%
args = parser.parse_args()
#%%
MODEL_NAME = args.modeldir
GRAPH_NAME = args.graph
LABELMAP_NAME = args.labels
# url = args.camaddr
min_conf_threshold = float(args.threshold)
resW, resH = args.resolution.split('x')
# imW, imH = int(resW), int(resH)
use_TPU = args.edgetpu
#%%
# MODEL_NAME = 'coco_ssd_mobilenet_v1'
# GRAPH_NAME = 'detect.tflite'
# LABELMAP_NAME = 'labelmap.txt'
# min_conf_threshold = float(0.5)
# resW, resH = '1920x1080'.split('x')  # laptom cam : resW, resH = '1280x720'.split('x') 
imW, imH = int(1280*640/int(resW)), int(720*360/int(resH)) # imW, imH = int(resW), int(resH)
# use_TPU = False #'store_true'


#%%
# Import TensorFlow libraries
# If tflite_runtime is installed, import interpreter from tflite_runtime, else import from regular tensorflow
# If using Coral Edge TPU, import the load_delegate library
pkg = importlib.util.find_spec('tflite_runtime')
#%%
if pkg:
    from tflite_runtime.interpreter import Interpreter
    if use_TPU:
        from tflite_runtime.interpreter import load_delegate
else:
    from tensorflow.lite.python.interpreter import Interpreter
    if use_TPU:
        from tensorflow.lite.python.interpreter import load_delegate

# If using Edge TPU, assign filename for Edge TPU model
if use_TPU:
    # If user has specified the name of the .tflite file, use that name, otherwise use default 'edgetpu.tflite'
    if (GRAPH_NAME == 'detect.tflite'):
        GRAPH_NAME = 'edgetpu.tflite'       
#%%
# Get path to current working directory
CWD_PATH = os.getcwd()

# Path to .tflite file, which contains the model that is used for object detection
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,GRAPH_NAME)

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH,MODEL_NAME,LABELMAP_NAME)
#%%
# Load the label map
with open(PATH_TO_LABELS, 'r') as f:
    labels = [line.strip() for line in f.readlines()]
#%%
# Have to do a weird fix for label map if using the COCO "starter model" from
# https://www.tensorflow.org/lite/models/object_detection/overview
# First label is '???', which has to be removed.
if labels[0] == '???':
    del(labels[0])
#%%
# Load the Tensorflow Lite model.
# If using Edge TPU, use special load_delegate argument
if use_TPU:
    interpreter = Interpreter(model_path=PATH_TO_CKPT,
                              experimental_delegates=[load_delegate('libedgetpu.so.1.0')])
    print(PATH_TO_CKPT)
else:
    interpreter = Interpreter(model_path=PATH_TO_CKPT)

interpreter.allocate_tensors()
#%%
# Get model details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

floating_model = (input_details[0]['dtype'] == np.float32)

input_mean = 127.5
input_std = 127.5
#%%
# Initialize frame rate calculation
frame_rate_calc = 1
freq = cv2.getTickFrequency()
#%%
# Initialize video stream
videostream = VideoStream(resolution=(imW,imH),framerate=30).start()
time.sleep(1)
#%%
# Create window
# cv2.namedWindow('Object detector', cv2.WINDOW_NORMAL)
os.chdir('/home/pi/Downloads/cctv_image')
#%%
#for frame1 in camera.capture_continuous(rawCapture, format="bgr",use_video_port=True):
while True:

    # Start timer (for calculating frame rate)
    t1 = cv2.getTickCount()

    # Grab frame from video stream
    frame1 = videostream.read()

    # Acquire frame and resize to expected shape [1xHxWx3]
    frame = frame1.copy()
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (width, height))
    input_data = np.expand_dims(frame_resized, axis=0)

    # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
    if floating_model:
        input_data = (np.float32(input_data) - input_mean) / input_std

    # Perform the actual detection by running the model with the image as input
    interpreter.set_tensor(input_details[0]['index'],input_data)
    interpreter.invoke()

    # Retrieve detection resultsq
    boxes = interpreter.get_tensor(output_details[0]['index'])[0] # Bounding box coordinates of detected objects
    classes = interpreter.get_tensor(output_details[1]['index'])[0] # Class index of detected objects
    scores = interpreter.get_tensor(output_details[2]['index'])[0] # Confidence of detected objects
    #num = interpreter.get_tensor(output_details[3]['index'])[0]  # Total number of detected objects (inaccurate and not needed)

    od_time = time.strftime('%Y-%m-%d %X', time.localtime(time.time()))
    max_i = -1
    # Loop over all detections and draw detection box if confidence is above minimum threshold
    for i in range(len(scores)):
        if ((scores[i] > min_conf_threshold) and (scores[i] <= 1.0)):

            max_i = i

            # Get bounding box coordinates and draw box
            # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
            ymin = int(max(1,(boxes[i][0] * imH)))
            xmin = int(max(1,(boxes[i][1] * imW)))
            ymax = int(min(imH,(boxes[i][2] * imH)))
            xmax = int(min(imW,(boxes[i][3] * imW)))
            
            cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)
            
            # Draw label
            object_name = labels[int(classes[i])] # Look up object name from "labels" array using class index
            label = '%s: %d%%' % (object_name, int(scores[i]*100)) # Example: 'person: 72%'
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) # Get font size
            label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
            cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
            cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text

            # Draw circle in center
            xcenter = xmin + (int(round((xmax - xmin) / 2)))
            ycenter = ymin + (int(round((ymax - ymin) / 2)))
            cv2.circle(frame, (xcenter, ycenter), 5, (0,0,255), thickness=-1)

            insert_tb_row = (od_time, farm_id, camera_id, object_name, int(scores[i]*100), xmin, xmax, ymin, ymax, xcenter, ycenter)
            
            # SQLite DB upload 
            sql = "insert into OD_LOCAL values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
            cursor.executemany(sql, [insert_tb_row])
            
            # Print info
            print('Object ' + str(i) + ': ' + object_name + ' at (' + str(xcenter) + ', ' + str(ycenter) + ')')      

    # DB commit if person is detected 
    selected_objects = [e for i, e in enumerate(labels) if i in list(map(int, classes[range(0,max_i+1)].tolist()))]
    if 'person' in selected_objects :
        if status_detect_cooltime == 0 :
            status_person = 1
            time_detect_person = cv2.getTickCount()
        con.commit()
    else : 
        status_person = 0
        cursor = con.cursor()

    # Draw framerate in corner of frame
    cv2.putText(frame,'FPS: {0:.2f}'.format(frame_rate_calc),(30,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2,cv2.LINE_AA)

    # All the results have been drawn on the frame, so it's time to display it.
    # cv2.imshow('Object detector', frame)

    # write frame to jpg : ?? second interval 
    duration_detect_person = (cv2.getTickCount()-time_detect_person)/freq
    print('duration second is ' + str(int(duration_detect_person)) +' second')

    if (status_person == 1 and status_detect_cooltime == 0) or (status_person == 1 and duration_detect_person >= image_upload_interval_sec) : 
        od_date = time.strftime('%Y_%m_%d', time.localtime(time.time()))  
        od_time1 = time.strftime('%H_%M_%S', time.localtime(time.time()))
        file_name = farm_id + '_' + camera_id + '_' + od_date + '_' + od_time1 + '.jpg'
        cv2.imwrite(file_name,frame) # 용량이 가장 적음
        
        status_detect_cooltime = 1 
        time_detect_person = cv2.getTickCount()

    elif status_person == 0 and duration_detect_person >= image_upload_interval_sec :
        status_detect_cooltime = 0    

    # Calculate framerate
    t2 = cv2.getTickCount()
    time1 = (t2-t1)/freq
    frame_rate_calc= 1/time1

    # Press 'q' to quit
    if cv2.waitKey(1) == ord('q'):
        break

# Clean up/
cv2.destroyAllWindows()
videostream.stop()
con.close()