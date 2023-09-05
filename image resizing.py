import cv2
import os
import time 
path = "C:/Users/Rishikeshs/Downloads/kagglecatsanddogs_5340/PetImages/Cat"
x = os.listdir(path)
for i in range(len(x)):
    filename = path + "/" + x[i]
    image = cv2.imread(filename)
    if(image is not None):
        #print(image.shape)
        half = cv2.resize(image, (50,50))
        cv2.imwrite(filename, half)
        print(filename)