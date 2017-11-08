import cv2
import numpy as np
import os

#Cascades definieren
faceDetection = cv2.CascadeClassifier('cascades/haarcascade_frontalface_default.xml')
eyeDetection = cv2.CascadeClassifier('cascades/haarcascade_eye.xml')

#Webcam auswählen
webcam = cv2.VideoCapture(0)

while(True):
    ret,img = webcam.read()
    imgGrayScale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceDetection.detectMultiScale(imgGrayScale,1.3,5)
    eyes = eyeDetection.detectMultiScale(imgGrayScale,1.3,5)
    for(x,y,width, height) in faces:
        cv2.rectangle(img, (x,y), (x+width, y++height), (0,255,0),2)
    for(x,y,width,height) in eyes:
        cv2.rectangle(img,(x,y),(x+width,y+height), (255,0,0), 1)
    cv2.imshow("Face", img)
    if(cv2.waitKey(1)==ord('q')):
        break;

webcam.release()
cv2.destroyAllWindows()
