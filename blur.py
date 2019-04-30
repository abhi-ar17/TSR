import numpy as np
import cv2
from matplotlib import pyplot as plt
img=cv2.imread('blurr.jpg')  #to input a saved img
#imeg=cv2.cvtColor(img,cv2_COLOR_BGR2GRAY)
img2=cv2.medianBlur(img,5)   #to remove blurness
cv2.imshow('org',img)
cv2.imshow('image',img2)     #o/p new img
cv2.waitKey(0)
cv2.destroyAllWindows()
