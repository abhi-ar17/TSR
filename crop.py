

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as zpimg

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
edged=cv2.Canny(object,30,200)
contours, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
print ("No : of Shapes {0}",format(len(contours)))
for cnt in contours:
	rect=cv2.minAreaRect(cnt)
	box=cv2.cv.BoxPoints(rect)
	box=np.int0(box)
	img=cv2.drawContours(img,cnt,-1,(255,0,0),3)


cv2.imshow('orginal',img)
cv2.imshow('mask',object)
cv2.imshow('edged',edged)
#cv2.imshow('final',iss)
cv2.waitKey(0)


