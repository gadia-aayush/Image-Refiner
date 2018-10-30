import cv2 as cv
import numpy as np
img= cv.imread(r'input.png',1)

hsv=cv.cvtColor(img,cv.COLOR_BGR2HSV)
h,s,v = cv.split(hsv)
th, threshed = cv.threshold(s, 100, 255, cv.THRESH_OTSU|cv.THRESH_BINARY) #black background
mask_w = cv.bitwise_not(threshed)     #white background
fg_masked = cv.bitwise_and(v, v, mask=mask_w) #masking the image of shirt with mask_w
dst = cv.inpaint(fg_masked,threshed,3, cv.INPAINT_NS) #inpainting 

#Dilation & Erosion.
kernel = np.ones((4, 4),np.uint8)
dilation = cv.dilate(dst,kernel,iterations = 2)
erosion = cv.erode(dilation, kernel, iterations=1)
dilation2= cv.dilate(erosion,kernel,iterations = 1)
dilation3= cv.dilate(dilation2,kernel,iterations = 1)
erosion_final = cv.erode(dilation3, kernel, iterations=3)
cv.imwrite("output_2 [improved].png", erosion_final)



#[written by AAYUSH GADIA]
