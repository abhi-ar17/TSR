import cv2
import numpy as np
image=cv2.imread("sign.jpeg")	
image=cv2.medianBlur(image,5)
image=cv2.resize(image,(200,200))

hsv=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)


lower_red = np.array([0,50,50])
upper_red = np.array([10,255,255])
mask0 = cv2.inRange(hsv, lower_red, upper_red)


lower_red = np.array([170,100,100])
upper_red = np.array([180,255,255])
mask1 = cv2.inRange(hsv, lower_red, upper_red)


mask = mask0+mask1
output_img=image.copy()
output_img[np.where(mask==0)]=0
#cv2.imshow("mask",mask)
#cv2.imshow("asdd",output_img)
#cv2.waitKey(0)
#imh=cv2.inRange(hsv,lowerBound,upperBound)
#cv2.imshow("aqe",imh)
#cv2.waitKey(0)
#imh_gray=cv2.cvtColor(imh,cv2.COLOR_BGR2GRAY)

contours, hier = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#print("-----------")




#ret, thresh = cv2.threshold(cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY) , 127, 255, cv2.THRESH_BINARY)
#contours, hier = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#print(contours)
#for c in contours:
# find bounding box coordinates
areas=[cv2.contourArea(c) for c in contours]
#print(areas)
max_index=np.argmax(areas)
cnt=contours[max_index]
x,y,w,h = cv2.boundingRect(cnt)
cv2.rectangle(image, (x,y), (x+w, y+h), (0, 255, 0), 2)
#cv2.drawContours(image, contours, 0, (255, 0, 0), 1)
#cv2.imshow("contours", image)
crop=image[y:y+h,x:x+w]
cv2.imshow("cropped",crop)
cv2.imwrite("cropped.jpeg",crop)
# find minimum area
#rect = cv2.minAreaRect(c)
# calculate coordinates of the minimum area rectangle
#box = cv2.boxPoints(rect)
# normalize coordinates to integers
#box = np.int0(box)
# draw contours
#cv2.drawContours(image, [box], 0, (0,0, 255), 3)
#cv2.imshow("asd",image)


cv2.waitKey(0)
