# import cv2
# from matplotlib import pyplot as plt
# image = cv2.imread("giza.jpg")
# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# plt.imshow(image)
import cv2
import os
import time 
path = "C:\\Users\\Rishikeshs\\Downloads\\kagglecatsanddogs_5340\\PetImages\\Cat\\10125.jpg"
image = cv2.imread(path)
cv2.imshow("cat", image)
cv2.waitKey(0)
# print(image.shape)
# half = cv2.resize(image, (50,50))
# cv2.imwrite(path, half)
# print(path)