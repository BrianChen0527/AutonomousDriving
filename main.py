import numpy as np
import tensorflow as tf
import keras
import cv2

cap = cv2.VideoCapture(0)

while(True):
    ret, frame = cap.read()

    #convert from BGR to HSV
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    #isolate the lane color (blue in this case)
    lowest_blue = np.array([65,30,30])
    highest_blue = np.array([145,255,255])
    mask = cv2.inRange(hsv_frame, lowest_blue, highest_blue)

    cv2.imshow('frame', mask)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()