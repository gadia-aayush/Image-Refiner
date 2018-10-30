import cv2 as cv
img= cv.imread('C:\InternshipWork\input.png',1)

hsv=cv.cvtColor(img,cv.COLOR_BGR2HSV)
h,s,v = cv.split(hsv)
th, threshed = cv.threshold(s, 100, 255, cv.THRESH_OTSU|cv.THRESH_BINARY) #black background
mask_w = cv.bitwise_not(threshed)     #white background
fg_masked = cv.bitwise_and(v, v, mask=mask_w) #masking the image of shirt with mask_w
dst = cv.inpaint(fg_masked,threshed,3, cv.INPAINT_NS) #inpainting 
cv.imwrite("final.png", dst)


#[written by AAYUSH GADIA]
