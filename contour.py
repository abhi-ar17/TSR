

import cv2
import numpy as np
#import matplotlib.pyplot as plt
#import matplotlib.image as zpimg

img = cv2.imread('sign.jpeg')

hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
lowerRange= np.array([0,100,100]) 
upperRange= np.array([10,255,255]) 
lowerBound= np.array([160,100,100])
upperBound= np.array([179,255,255])
 

#to change size of the image
image = cv2.resize(img,(360,240))

#object=cv2.inRange(hsv,lowerRange,upperRange)
object=cv2.inRange(hsv,lowerBound,upperBound)
#it allows pixels within range and black out other
edged=cv2.Canny(object,30,150)
contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

for c in contours:
	print (c)
	break;

cv2.drawContours(img,contours,0,(255,0,0),3)

#cv2.rectangle(img,(83,64),(150,170),(0,255,0),3)
cv2.imshow('orginal',img)
cv2.imshow('HSV',hsv)
cv2.imshow('mask', object)
cv2.imshow('edged',edged)

cv2.waitKey(0)


