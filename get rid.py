import os
import time
path = "C:\\Users\\rishi\\Documents\\image processing\\gan output images"
l = []
x = os.listdir(path)
print(len(x))
# for i in range(len(x)):
#     if((i + 1) % 25 != 0):
#         print("deleted file : ", path +"\\" +  x[i])
#         os.remove(path +"\\" +  x[i])