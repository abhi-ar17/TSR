

import cv2
import numpy as np
#import matplotlib.pyplot as plt
#import matplotlib.image as zpimg

img = cv2.imread('sign.jpeg')
imgee = cv2.imread('sign.jpeg',0)

hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
lowerRange= np.array([0,100,100]) 
upperRange= np.array([10,255,255]) 
lowerBound= np.array([160,100,100])
upperBound= np.array([179,255,255])
 
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
#create mask
height,width = imgee.shape
mask=np.zeros((height,width),np.uint8)

#to change size of the image
image = cv2.resize(img,(360,240))

#object=cv2.inRange(hsv,lowerRange,upperRange)
object=cv2.inRange(hsv,lowerBound,upperBound)
#it allows pixels within range and black out other
cv2.imshow('first',thresh)
edged=cv2.Canny(object,30,150)

contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

for c in contours:
	print (c)
	break;

final=cv2.drawContours(img,contours,0,(255,0,0),0)
cv2.imshow('wrw',final)
object1=cv2.inRange(img,(255,0,0),(255,0,0))
cv2.imshow('object1',object1)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
edged1=cv2.Canny(thresh,100,200)
cv2.imshow('Edge',edged1)

circles=cv2.HoughCircles(edged1,cv2.HOUGH_GRADIENT,1,50,param1=50,param2=20,minRadius=50,maxRadius=750)
for i in circles[0,:]:
	i[2]=i[2]+4
		#draw on mask
	cv2.circle(mask,(i[0],i[1]),i[2],(255,255,255),-1)
#copy that image using that mask
masked_data=cv2.bitwise_and(img,img,mask=mask)

#apply threshold
_,thresh = cv2.threshold(mask,1,255,cv2.THRESH_BINARY)

#Find Contour
cnts,_=cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
x,y,w,h =cv2.boundingRect(contours[0])

#crop masked data
crop=masked_data[y:y+h,x:x+w]


#cv2.rectangle(img,(83,64),(150,170),(0,255,0),3)
cv2.imshow('orginal',img)
cv2.imshow('HSV',hsv)
cv2.imshow('CROP',crop)
cv2.imshow('mask', object)
cv2.imshow('edged',edged)
cv2.imshow('thresh',thresh)
cv2.imshow('object2',object1)

cv2.waitKey(0)


