import cv2
import numpy as np
#opening an image
img = cv2.imread("C:\\Users\\rishi\\OneDrive\\Desktop\\cat.jpg")
#accessing a pixel
px = img[100,100]
#printing the BGR values of the pixel
print(px)
#accessing the properties of the image
'''
the shape method returns a tuple containing the number of rows, columns and the number of channels contained in an
image. 
'''
print(img.shape)
'''
the split method is used to split the image into it's respective channels
'''
b, g, r = cv2.split(img)
a = 15.1
b = 15
c = a-b
print(c)