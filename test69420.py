import cv2
from matplotlib import pyplot as plt
image = cv2.imread("giza.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.imshow(image)